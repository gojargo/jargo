// Package backend is the backend the llm-with-backend examples share: the slow
// half of an engineering assistant.
//
// The backend runs Claude with tools that stand in for real engineering work,
// each taking as long as the real thing plausibly would:
//
//   - Code changes, a loop: search_codebase finds the files, read_file reads
//     them, apply_patch changes one, run_tests checks the result. The first test
//     run after a change to the client trips a neighboring test that wants each
//     retry delay reported to a metrics module, so the backend has to read that
//     test, patch again and run again before it can report the change done. A
//     minute or so end to end, most of it the model's own steps.
//   - Research: search_docs then read_doc, a few seconds each.
//   - Quick lookups: check_ci_status and list_open_prs, a second each.
//   - Long runs: run_integration_tests, twenty seconds, long enough for the user
//     to change their mind while it runs.
//
// The slow tools are asynchronous (not canceled on interruption) and
// cancellable by the LLM, so the backend's model keeps taking new messages
// while they run and cancels them when a message makes their results unwanted.
//
// Each frontend example builds its backend with BuildBackend and gives its own
// model FrontendInstructions, so they differ only in the frontend service.
//
// BACKEND_WORK_SCALE, when set, scales how long the tools take.
package backend

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers/llmworker"
)

// FrontendInstructions is the frontend's system instruction.
const FrontendInstructions = `You are Pip, the voice assistant of the Acme engineering team. Your
responses are spoken aloud, so keep them to one or two natural sentences without any
formatting. You can make code changes and run the tests, research the team's internal
docs, and check CI and the open pull requests; that work takes a moment.`

// BackendInstructions is the backend's system instruction.
const BackendInstructions = `You do the engineering work for the Acme team's assistant. Use your
tools: search the codebase, read files, apply patches and run the tests for code changes;
search and read the internal docs for research; check CI and list pull requests for status.
When asked for a code change, keep going until the tests pass, then tell the user what you
changed and the test result. Change code only when the user asked for a change: a review, a
question or a status check changes nothing, however clear the fix looks; tell the user what
you found and what you would change, and stop until they ask for it. Keep what you tell them
short and concrete: file names, what changed, what the tests said.`

// The simulated codebase and docs.

// file is one file of the simulated codebase.
type file struct {
	path    string
	content string
}

// doc is one of the team's simulated internal docs.
type doc struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Summary string `json:"summary"`
}

// docRef is a doc as search_docs reports it.
type docRef struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

// failure is one failing test.
type failure struct {
	Test    string `json:"test"`
	Message string `json:"message"`
}

// testRun is what a test run reports.
type testRun struct {
	Passed   int       `json:"passed"`
	Failed   int       `json:"failed"`
	Failures []failure `json:"failures"`
}

// pullRequest is one open pull request.
type pullRequest struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Author string `json:"author"`
}

// errorResult is what a tool reports when what it was asked for does not exist.
type errorResult struct {
	Error string `json:"error"`
}

// codebase is the simulated codebase: its files in order, and the patches
// applied to each.
type codebase struct {
	mu      sync.Mutex
	files   []file
	patches map[string][]string
}

//nolint:gochecknoglobals // the simulated codebase the tools work on
var repo = &codebase{
	files: []file{
		{"src/http_client.py", "import time\n\n\n" +
			"class TransientError(Exception):\n" +
			"    pass\n\n\n" +
			"class RetriesExhausted(Exception):\n" +
			"    pass\n\n\n" +
			"class HttpClient:\n" +
			"    RETRIES = 3\n" +
			"    BACKOFF_SECS = 1\n\n" +
			"    def __init__(self, transport):\n" +
			"        self._transport = transport\n\n" +
			"    def get(self, url):\n" +
			"        for attempt in range(self.RETRIES):\n" +
			"            try:\n" +
			"                return self._transport(url)\n" +
			"            except TransientError:\n" +
			"                time.sleep(self.BACKOFF_SECS)\n" +
			"        raise RetriesExhausted(url)\n"},
		{"tests/test_http_client.py", "import time\n\n" +
			"from src.http_client import HttpClient, TransientError\n\n\n" +
			"def flaky(fail_times):\n" +
			"    calls = []\n\n" +
			"    def transport(url):\n" +
			"        calls.append(url)\n" +
			"        if len(calls) <= fail_times:\n" +
			"            raise TransientError()\n" +
			"        return 'ok'\n\n" +
			"    return transport\n\n\n" +
			"def test_retry_backoff(monkeypatch):\n" +
			"    sleeps = []\n" +
			"    monkeypatch.setattr(time, 'sleep', sleeps.append)\n" +
			"    HttpClient(flaky(2)).get('https://example.test/flaky')\n" +
			"    # The backoff between retries must grow, or a busy upstream\n" +
			"    # gets hammered at a fixed rate.\n" +
			"    assert sleeps == sorted(sleeps) and sleeps[0] < sleeps[-1]\n\n\n" +
			"def test_first_retry_waits_one_base_interval(monkeypatch):\n" +
			"    sleeps = []\n" +
			"    monkeypatch.setattr(time, 'sleep', sleeps.append)\n" +
			"    HttpClient(flaky(1)).get('https://example.test/flaky')\n" +
			"    assert sleeps[0] == HttpClient.BACKOFF_SECS\n"},
		{"src/metrics.py", "_backoffs = []\n\n\n" +
			"def record_backoff(attempt, delay):\n" +
			"    # Report one retry delay, so the retry dashboard can chart them.\n" +
			"    _backoffs.append((attempt, delay))\n"},
		{"tests/integration/test_http_client_live.py", "import time\n\n" +
			"from src.http_client import HttpClient\n" +
			"from tests.integration.stack import live_upstream\n\n\n" +
			"DEADLINE_SECS = 2\n\n\n" +
			"def test_retry_on_timeout():\n" +
			"    # The live upstream drops the first two requests on the floor; the\n" +
			"    # client must get through on a retry before the deadline is up.\n" +
			"    with live_upstream(drop_first=2) as upstream:\n" +
			"        started = time.monotonic()\n" +
			"        HttpClient(upstream.transport).get(upstream.url('/health'))\n" +
			"        assert time.monotonic() - started < DEADLINE_SECS\n"},
		{"tests/test_retry_metrics.py", "import time\n\n" +
			"from src import metrics\n" +
			"from src.http_client import HttpClient\n" +
			"from tests.test_http_client import flaky\n\n\n" +
			"def test_every_retry_delay_is_recorded(monkeypatch):\n" +
			"    monkeypatch.setattr(time, 'sleep', lambda secs: None)\n" +
			"    recorded = []\n" +
			"    monkeypatch.setattr(metrics, 'record_backoff', lambda attempt, delay: recorded.append(delay))\n" +
			"    HttpClient(flaky(2)).get('https://example.test/flaky')\n" +
			"    # The dashboard charts what the client actually waited, so the client\n" +
			"    # reports each delay before sleeping it.\n" +
			"    assert len(recorded) == 2\n"},
	},
	patches: map[string][]string{},
}

//nolint:gochecknoglobals // the simulated internal docs
var docs = []doc{
	{
		ID:    "rfc-12",
		Title: "RFC 12: Retry policy for outbound HTTP",
		Summary: "Retries of outbound HTTP calls back off exponentially from a base of one second, " +
			"doubling on each attempt up to a cap of thirty seconds, with full jitter. Fixed " +
			"delays are disallowed because they synchronize clients against a struggling " +
			"upstream.",
	},
	{
		ID:    "guide-http",
		Title: "HTTP client guide",
		Summary: "HttpClient wraps requests with retries, timeouts and tracing. Retry behavior is " +
			"governed by RFC 12; the client's fixed BACKOFF_SECS sleep predates it.",
	},
	{
		ID:    "oncall-runbook",
		Title: "On-call runbook: upstream timeouts",
		Summary: "When the payments upstream times out, check its status page first, then the " +
			"retry dashboard for a spike in RetriesExhausted.",
	},
}

// fixedSleep finds a client that still sleeps a fixed BACKOFF_SECS.
var fixedSleep = regexp.MustCompile(`sleep\(\s*(\w+\.)?BACKOFF_SECS\s*\)`)

// work takes about as long as the real thing would.
func work(ctx context.Context, seconds float64) error {
	scale := 1.0
	if s := os.Getenv("BACKEND_WORK_SCALE"); s != "" {
		parsed, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return fmt.Errorf("BACKEND_WORK_SCALE: %w", err)
		}
		scale = parsed
	}
	select {
	case <-time.After(time.Duration(seconds * scale * float64(time.Second))):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// result reports v, encoded as JSON, as a tool call's result.
func result(ctx context.Context, params llm.FunctionCallParams, v any) error {
	encoded, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return params.Result(ctx, string(encoded), nil)
}

// arguments decodes a tool call's arguments into v.
func arguments(params llm.FunctionCallParams, v any) error {
	if len(params.Arguments) == 0 {
		return nil
	}
	return json.Unmarshal(params.Arguments, v)
}

// searchCodebase searches the codebase for files relevant to a query.
func searchCodebase(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Query string `json:"query"`
	}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 2); err != nil {
		return err
	}
	terms := strings.Fields(strings.ToLower(args.Query))
	repo.mu.Lock()
	hits := []string{}
	all := make([]string, 0, len(repo.files))
	for _, f := range repo.files {
		all = append(all, f.path)
		for _, term := range terms {
			if strings.Contains(strings.ToLower(f.path), term) || strings.Contains(strings.ToLower(f.content), term) {
				hits = append(hits, f.path)
				break
			}
		}
	}
	repo.mu.Unlock()
	if len(hits) == 0 {
		hits = all
	}
	return result(ctx, params, struct {
		Files []string `json:"files"`
	}{hits})
}

// readFile reads a file from the codebase.
func readFile(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Path string `json:"path"`
	}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 1.5); err != nil {
		return err
	}
	content, ok := repo.read(args.Path)
	if !ok {
		return result(ctx, params, errorResult{"no such file: " + args.Path})
	}
	return result(ctx, params, struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}{args.Path, content})
}

// applyPatch replaces a file's content in the codebase.
func applyPatch(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 2); err != nil {
		return err
	}
	if !repo.write(args.Path, args.Content) {
		return result(ctx, params, errorResult{"no such file: " + args.Path})
	}
	return result(ctx, params, struct {
		Path    string `json:"path"`
		Applied bool   `json:"applied"`
		Lines   int    `json:"lines"`
	}{args.Path, true, strings.Count(args.Content, "\n")})
}

// runTests runs the test suite, or one test file. It takes a while.
func runTests(ctx context.Context, params llm.FunctionCallParams) error {
	if err := work(ctx, 5); err != nil {
		return err
	}
	// Two tests read the client as last patched: the backoff test wants the
	// sleep to depend on the attempt, and a neighboring metrics test the
	// backend has not seen wants each delay reported to the metrics module. A
	// fix that does both at once passes both; one that does only the first
	// trips the second, and the backend has to read that test and patch again.
	client := repo.lastPatch("src/http_client.py")
	failures := []failure{}
	if fixedSleep.MatchString(client) {
		failures = append(failures, failure{
			Test:    "tests/test_http_client.py::test_retry_backoff",
			Message: "assert [1, 1] == sorted([1, 1]) and 1 < 1: the retry delays do not grow",
		})
	}
	if !strings.Contains(client, "record_backoff(") {
		failures = append(failures, failure{
			Test: "tests/test_retry_metrics.py::test_every_retry_delay_is_recorded",
			Message: "assert 0 == 2: no retry delay was recorded; the client must call " +
				"metrics.record_backoff(attempt, delay) before each sleep (see " +
				"tests/test_retry_metrics.py)",
		})
	}
	return result(ctx, params, testRun{Passed: 42 - len(failures), Failed: len(failures), Failures: failures})
}

// runIntegrationTests runs the integration test suite, or one of its files,
// against a live stack. It takes a while.
func runIntegrationTests(ctx context.Context, params llm.FunctionCallParams) error {
	if err := work(ctx, 20); err != nil {
		return err
	}
	return result(ctx, params, testRun{
		Passed: 17,
		Failed: 1,
		Failures: []failure{{
			Test:    "tests/integration/test_http_client_live.py::test_retry_on_timeout",
			Message: "TimeoutError: the third retry exceeded the 2s deadline",
		}},
	})
}

// searchDocs searches the team's internal docs. It takes a few seconds.
func searchDocs(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Query string `json:"query"`
	}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 3); err != nil {
		return err
	}
	terms := strings.Fields(strings.ToLower(args.Query))
	hits := []docRef{}
	all := make([]docRef, 0, len(docs))
	for _, d := range docs {
		all = append(all, docRef{ID: d.ID, Title: d.Title})
		for _, term := range terms {
			if strings.Contains(strings.ToLower(d.Title), term) || strings.Contains(strings.ToLower(d.Summary), term) {
				hits = append(hits, docRef{ID: d.ID, Title: d.Title})
				break
			}
		}
	}
	if len(hits) == 0 {
		hits = all
	}
	return result(ctx, params, struct {
		Docs []docRef `json:"docs"`
	}{hits})
}

// readDoc reads one of the team's internal docs.
func readDoc(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		DocID string `json:"doc_id"`
	}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 2); err != nil {
		return err
	}
	for _, d := range docs {
		if d.ID == args.DocID {
			return result(ctx, params, d)
		}
	}
	return result(ctx, params, errorResult{"no such doc: " + args.DocID})
}

// checkCIStatus checks the CI status of a branch.
func checkCIStatus(ctx context.Context, params llm.FunctionCallParams) error {
	args := struct {
		Branch string `json:"branch"`
	}{Branch: "main"}
	if err := arguments(params, &args); err != nil {
		return err
	}
	if err := work(ctx, 1); err != nil {
		return err
	}
	status := "running"
	if args.Branch == "main" {
		status = "green"
	}
	return result(ctx, params, struct {
		Branch  string `json:"branch"`
		Status  string `json:"status"`
		LastRun string `json:"last_run"`
	}{args.Branch, status, time.Now().Format("15:04")})
}

// listOpenPRs lists the team's open pull requests.
func listOpenPRs(ctx context.Context, params llm.FunctionCallParams) error {
	if err := work(ctx, 1); err != nil {
		return err
	}
	return result(ctx, params, struct {
		PullRequests []pullRequest `json:"pull_requests"`
	}{[]pullRequest{
		{Number: 412, Title: "Add tracing to the payments client", Author: "ana"},
		{Number: 415, Title: "Bump httpx to 0.28", Author: "renovate"},
		{Number: 418, Title: "Retry policy: exponential backoff (RFC 12)", Author: "sam"},
	}})
}

// read returns a file's content.
func (c *codebase) read(path string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, f := range c.files {
		if f.path == path {
			return f.content, true
		}
	}
	return "", false
}

// write replaces a file's content and records the patch, reporting whether the
// file exists.
func (c *codebase) write(path, content string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.files {
		if c.files[i].path == path {
			c.files[i].content = content
			c.patches[path] = append(c.patches[path], content)
			return true
		}
	}
	return false
}

// lastPatch is a file as last patched, or as it was when it never was.
func (c *codebase) lastPatch(path string) string {
	c.mu.Lock()
	patches := c.patches[path]
	c.mu.Unlock()
	if len(patches) > 0 {
		return patches[len(patches)-1]
	}
	content, _ := c.read(path)
	return content
}

// tool describes one tool to the model, carrying its handler.
func tool(name, description, parameters string, handler llm.FunctionCallHandler) frames.Tool {
	return frames.Tool{
		Name:        name,
		Description: description,
		Parameters:  json.RawMessage(parameters),
		Handler:     handler,
	}
}

// slow marks a tool as asynchronous and cancellable by the model.
func slow(t frames.Tool) frames.Tool {
	t.CancelOnInterruption = new(false)
	t.CancellableByLLM = new(true)
	return t
}

// tools are the backend's tools.
func tools() []frames.Tool {
	return []frames.Tool{
		tool("search_codebase", "Search the codebase for files relevant to a query.", `{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "What to look for: a symbol, a file name, or a description."
				}
			},
			"required": ["query"]
		}`, searchCodebase),
		tool("read_file", "Read a file from the codebase.", `{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "The file's path, as search_codebase reports it."
				}
			},
			"required": ["path"]
		}`, readFile),
		tool("apply_patch", "Replace a file's content in the codebase.", `{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "The file to change."
				},
				"content": {
					"type": "string",
					"description": "The file's new content, in full."
				}
			},
			"required": ["path", "content"]
		}`, applyPatch),
		slow(tool("run_tests", "Run the test suite, or one test file. Takes a while.", `{
			"type": "object",
			"properties": {
				"path": {
					"anyOf": [{"type": "string"}, {"type": "null"}],
					"description": "A test file to run alone; the whole suite when omitted."
				}
			},
			"required": []
		}`, runTests)),
		slow(tool("run_integration_tests",
			"Run the integration test suite, or one of its files, against a live stack. Takes a while.", `{
			"type": "object",
			"properties": {
				"path": {
					"anyOf": [{"type": "string"}, {"type": "null"}],
					"description": "An integration test file to run alone; the whole suite when omitted."
				}
			},
			"required": []
		}`, runIntegrationTests)),
		slow(tool("search_docs", "Search the team's internal docs. Takes a few seconds.", `{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "What to look for."
				}
			},
			"required": ["query"]
		}`, searchDocs)),
		tool("read_doc", "Read one of the team's internal docs.", `{
			"type": "object",
			"properties": {
				"doc_id": {
					"type": "string",
					"description": "The doc's id, as search_docs reports it."
				}
			},
			"required": ["doc_id"]
		}`, readDoc),
		tool("check_ci_status", "Check the CI status of a branch.", `{
			"type": "object",
			"properties": {
				"branch": {
					"type": "string",
					"description": "The branch; main when omitted."
				}
			},
			"required": []
		}`, checkCIStatus),
		tool("list_open_prs", "List the team's open pull requests.",
			`{"type": "object", "properties": {}, "required": []}`, listOpenPRs),
	}
}

// BuildBackend builds the backend worker the examples share.
func BuildBackend() (*llmworker.BackendWorker, error) {
	model := anthropic.NewLLM(anthropic.Config{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
		// Thinking summaries reach the frontend as silent "Backend (thinking):"
		// messages, so it can say how the work is going if asked. The budget
		// keeps a step to seconds: left to decide for itself, the model can think
		// for a minute over one file.
		Thinking: &anthropic.ThinkingConfig{Type: "enabled", BudgetTokens: 2048, Display: "summarized"},
		// Thinking counts against MaxTokens; the budget plus a patch with a
		// file's whole content must fit.
		MaxTokens: 16384,
	})
	model.SetSystemInstruction(BackendInstructions)

	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	return llmworker.NewBackend(llmworker.BackendConfig{LLM: model, Context: convo})
}
