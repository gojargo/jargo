package eval_test

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/eval"
)

func writeFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func wsOf(s *httptest.Server) string { return "ws" + strings.TrimPrefix(s.URL, "http") }

// runSuite loads the manifest in dir and runs it, returning the suite and the
// results.jsonl records.
func runSuite(t *testing.T, dir string, o eval.ManifestOverrides) (*eval.Suite, []map[string]any) {
	t.Helper()
	m, err := eval.LoadManifest(filepath.Join(dir, "manifest.yaml"), o)
	if err != nil {
		t.Fatal(err)
	}
	suite := eval.NewSuite(m)
	results := filepath.Join(dir, "run", "results.jsonl")
	opts := eval.SuiteRunOptions{ResultsPath: results}
	if err := suite.Run(t.Context(), filepath.Join(dir, "run", "logs"), opts); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(results) //nolint:gosec // a test file
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var records []map[string]any
	for s := bufio.NewScanner(f); s.Scan(); {
		var r map[string]any
		if err := json.Unmarshal(s.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		records = append(records, r)
	}
	return suite, records
}

const passScenario = `name: pass
turns:
  - user: "hello world"
    expect:
      - event: llm_response
        text_contains: "hello world"
`

const failScenario = `name: fail
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "goodbye"
        within_ms: 2000
`

// An entry naming a bot_url connects to a bot already running rather than
// spawning one.
func TestSuiteAgainstRunningBots(t *testing.T) {
	srv1 := httptest.NewServer(eval.Handler(buildFakeBot))
	defer srv1.Close()
	srv2 := httptest.NewServer(eval.Handler(buildFakeBot))
	defer srv2.Close()

	dir := t.TempDir()
	writeFile(t, dir, "pass.yaml", passScenario)
	writeFile(t, dir, "fail.yaml", failScenario)
	writeFile(t, dir, "manifest.yaml", fmt.Sprintf(`concurrency: 2
suite:
  - bot_url: %s
    scenarios: [pass.yaml, fail.yaml]
  - bot_url: %s
    name: second
    scenarios: [pass.yaml]
`, wsOf(srv1), wsOf(srv2)))

	suite, records := runSuite(t, dir, eval.ManifestOverrides{})
	if len(suite.Runs) != 3 || len(records) != 3 {
		t.Fatalf("got %d runs and %d records, want 3", len(suite.Runs), len(records))
	}
	passed := 0
	for _, r := range suite.Runs {
		if r.Error != "" || r.ScriptResult == nil {
			t.Fatalf("run %s: error %q", r.Scenario, r.Error)
		}
		if r.ScriptResult.Passed() {
			passed++
		}
	}
	if passed != 2 { // pass.yaml runs on both bots; fail.yaml fails
		t.Fatalf("want 2 passed of 3, got %d", passed)
	}
	labels := map[string]bool{}
	for _, rec := range records {
		labels[fmt.Sprint(rec["name"])] = true
	}
	if !labels[wsOf(srv1)] || !labels["second"] {
		t.Fatalf("labels = %v, want the URL and the entry's name", labels)
	}
	// The decision trace of each run is kept beside its log.
	if traces, _ := filepath.Glob(filepath.Join(dir, "run", "logs", "*__second__pass.eval.log")); len(traces) != 1 {
		t.Fatalf("traces = %v, want the run's", traces)
	}
}

// overlapBot counts how many scenarios are being played against it at once,
// and records the highest that count ever reached.
type overlapBot struct {
	mu   sync.Mutex
	live int
	peak int
}

func (b *overlapBot) serve(t *testing.T) string {
	t.Helper()
	bot := eval.Handler(buildFakeBot)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.live++
		b.peak = max(b.peak, b.live)
		b.mu.Unlock()
		defer func() {
			b.mu.Lock()
			b.live--
			b.mu.Unlock()
		}()
		bot.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return wsOf(srv)
}

// An entry's concurrency: is how many it may have in flight at once, for a bot
// whose provider rate-limits concurrent connections.
//
// The count can reach twice the cap, because a connection the harness has
// finished with is counted until the bot's own pipeline has torn down behind
// it. The suite's own count is held to the cap exactly by the scheduling tests.
func TestSuiteAnEntrysConcurrencyIsItsOwnCap(t *testing.T) {
	bot := &overlapBot{}
	dir := t.TempDir()
	var names []string
	for i := range 6 {
		body := strings.Replace(passScenario, "name: pass", fmt.Sprintf("name: s%d", i), 1)
		writeFile(t, dir, fmt.Sprintf("s%d.yaml", i), body)
		names = append(names, fmt.Sprintf("s%d.yaml", i))
	}
	writeFile(t, dir, "manifest.yaml", fmt.Sprintf(
		"concurrency: 6\nsuite:\n  - bot_url: %s\n    concurrency: 2\n    scenarios: [%s]\n",
		bot.serve(t), strings.Join(names, ", ")))
	runSuite(t, dir, eval.ManifestOverrides{})
	bot.mu.Lock()
	defer bot.mu.Unlock()
	if bot.peak > 4 {
		t.Fatalf("the bot was played %d scenarios at once, want at most its 2", bot.peak)
	}
}

// freePort is a port nothing listens on, as a base for the suite's.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := (&net.ListenConfig{}).Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port //nolint:forcetypeassert // a TCP listener
}

// The suite builds a bot from its Go package once, spawns it on its own port
// for each run with its runner body, drives it, and stops it.
func TestSuiteBuildsAndSpawnsBots(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a bot")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	testdata, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	writeFile(t, dir, "pass.yaml", passScenario)
	writeFile(t, dir, "body.json", `{"k":"v"}`)
	writeFile(t, dir, "inline.yaml", `name: inline
turns:
  - user: "hi"
    expect:
      - event: llm_response
        text_contains: 'echo ({"model":"m"}): hi'
`)
	writeFile(t, dir, "manifest.yaml", fmt.Sprintf(`bots_dir: %s
suite:
  - bot: suitebot
    scenarios: [pass.yaml]
  - bot: suitebot
    name: with-body
    runner_body: {path: body.json}
    scenarios: [pass.yaml]
  - bot: suitebot
    name: inline
    runner_body: {data: {model: m}}
    scenarios: [inline.yaml]
`, testdata))
	port := freePort(t)
	suite, records := runSuite(t, dir, eval.ManifestOverrides{BasePort: &port})
	for _, r := range suite.Runs {
		if r.Error != "" || r.ScriptResult == nil || !r.ScriptResult.Passed() {
			log, _ := os.ReadFile(filepath.Join( //nolint:gosec // a test file
				dir, "run", "logs", strings.ReplaceAll(r.Bot, "/", "_")+"__"+r.Scenario+".log"))
			t.Fatalf("run %s/%s: error %q result %+v\nbot log:\n%s", r.Label(), r.Scenario, r.Error, r.ScriptResult, log)
		}
	}
	if len(records) != 3 {
		t.Fatalf("got %d records", len(records))
	}
	if _, err := os.Stat(filepath.Join(dir, "run", "bin", "suitebot")); err != nil {
		t.Fatalf("the bot was not built once into bin/: %v", err)
	}
	bodyPath := filepath.Join(dir, "run", "logs", "suitebot__inline__inline.body.json")
	body, err := os.ReadFile(bodyPath) //nolint:gosec // a test file
	if err != nil || string(body) != `{"model":"m"}` {
		t.Fatalf("inline body = %s, %v", body, err)
	}
}
