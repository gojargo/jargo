package interactions

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	jevent "github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

//nolint:gochecknoglobals // test fixtures
var (
	signature   = []byte("\x12\x34signature")
	weatherTool = frames.Tool{
		Name:        "get_weather",
		Description: "Get the weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
)

// evt renders one stream event.
func evt(v map[string]any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func start(index int, step map[string]any) string {
	return evt(map[string]any{"event_type": "step.start", "index": index, "step": step})
}

func delta(index int, d map[string]any) string {
	return evt(map[string]any{"event_type": "step.delta", "index": index, "delta": d})
}

func stop(index int) string {
	return evt(map[string]any{"event_type": "step.stop", "index": index})
}

func completed(status string, usage map[string]any) string {
	in := map[string]any{"id": "", "status": status, "model": "gemini-3.6-flash", "object": "interaction"}
	if usage != nil {
		in["usage"] = usage
	}
	return evt(map[string]any{"event_type": "interaction.completed", "interaction": in})
}

func signedThought(index int) []string {
	return []string{
		start(index, map[string]any{"type": "thought"}),
		delta(index, map[string]any{"type": "thought_signature", "signature": base64.StdEncoding.EncodeToString(signature)}),
		stop(index),
	}
}

func textReply(text string, index int) []string {
	return []string{
		start(index, map[string]any{"type": "model_output"}),
		delta(index, map[string]any{"type": "text", "text": text}),
		stop(index),
	}
}

func functionCall(index int, callID, city string) []string {
	return []string{
		start(index, map[string]any{
			"type": "function_call", "id": callID, "name": "get_weather", "arguments": map[string]any{},
		}),
		delta(index, map[string]any{"type": "arguments_delta", "arguments": `{"city": "` + city + `"}`}),
		stop(index),
	}
}

func eventsOf(groups ...[]string) []string { return slices.Concat(groups...) }

// sse renders events as a server-sent event stream.
func sse(evts []string) string {
	var b strings.Builder
	for _, e := range evts {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

// fakeAPI stands in for the Interactions endpoint, replying to each request in
// turn and recording what it was sent.
type fakeAPI struct {
	*httptest.Server
	mu       sync.Mutex
	requests []map[string]any
	headers  []http.Header
}

// reply is one response of the fake API: a status and a body, or a stream that
// stalls after its jevent.
type reply struct {
	status int
	body   string
	stall  bool
}

func newFakeAPI(t *testing.T, replies ...reply) *fakeAPI {
	t.Helper()
	f := &fakeAPI{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		n := len(f.requests)
		f.requests = append(f.requests, body)
		f.headers = append(f.headers, r.Header.Clone())
		f.mu.Unlock()
		rep := reply{body: sse(nil)}
		if n < len(replies) {
			rep = replies[n]
		}
		if rep.status != 0 {
			w.WriteHeader(rep.status)
			_, _ = w.Write([]byte(rep.body))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(rep.body))
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
		if rep.stall {
			<-r.Context().Done()
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// request is the n-th request the API received.
func (f *fakeAPI) request(t *testing.T, n int) map[string]any {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if n >= len(f.requests) {
		t.Fatalf("the API received %d requests, want at least %d", len(f.requests), n+1)
	}
	return f.requests[n]
}

// testShaper sends the service's requests to the fake API.
type testShaper struct{ url string }

func (s testShaper) Endpoint() string { return s.url }

func (testShaper) Authorize(_ context.Context, req *http.Request) error {
	req.Header.Set("x-goog-api-key", "test-key")
	return nil
}

func makeService(api *fakeAPI, cfg Config) *LLMService {
	cfg.APIKey = "test-key"
	return NewShapedLLM("GoogleInteractionsLLM", testShaper{url: api.URL}, cfg)
}

func userContext(text string) *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage(text)
	return convo
}

// runContext runs one turn of convo through svc and returns the frames that came
// out of it downstream and upstream. wait, when set, is waited on before the
// pipeline is stopped.
func runContext(
	t *testing.T, svc *LLMService, convo *frames.LLMContext, params pipeline.Params, wait <-chan struct{},
) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	noRTVI := false
	ended := make(chan struct{})
	var once sync.Once
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		Params:                  params,
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	jevent.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame:
			return
		case *frames.LLMFullResponseEndFrame:
			once.Do(func() { close(ended) })
		}
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	jevent.On(&worker.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("the response never ended")
	}
	if wait != nil {
		select {
		case <-wait:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting")
		}
	}
	worker.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	return down, up
}

func storedSignatures(fs []frames.Frame) []map[string]any {
	var out []map[string]any
	for _, f := range fs {
		if app, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			for _, m := range app.Messages {
				if native, ok := m.Native.(map[string]any); ok {
					out = append(out, native)
				}
			}
		}
	}
	return out
}

func errorMessages(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e.Error)
		}
	}
	return out
}

// capturedLogs swaps the default logger for one writing level and above into a
// buffer, restoring it when the test ends.
func capturedLogs(t *testing.T, level slog.Level) func() string {
	t.Helper()
	var mu sync.Mutex
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(writerFunc(func(p []byte) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return buf.Write(p)
	}), &slog.HandlerOptions{Level: level})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		return buf.String()
	}
}

type writerFunc func(p []byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

// TestTextReply checks text deltas becoming text frames inside a full response.
func TestTextReply(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		signedThought(0), textReply("Hello there.", 1), []string{completed("completed", nil)},
	))})
	down, _ := runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)

	startAt := slices.IndexFunc(down, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMFullResponseStartFrame)
		return ok
	})
	textAt := slices.IndexFunc(down, func(f frames.Frame) bool { _, ok := f.(*frames.LLMTextFrame); return ok })
	if startAt < 0 || textAt < startAt {
		t.Errorf("frames = %v, want the text inside the response", down)
	}
	if _, ok := down[len(down)-1].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("last frame = %T, want the end of the response", down[len(down)-1])
	}
	var text strings.Builder
	for _, f := range down {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			text.WriteString(tf.Text)
		}
	}
	if text.String() != "Hello there." {
		t.Errorf("text = %q", text.String())
	}
}

// TestTextSignatureStoredWithItsText checks a reply's signature is stored as raw
// bytes, bookmarked by the reply's text.
func TestTextSignatureStoredWithItsText(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		signedThought(0), textReply("Hello there.", 1), []string{completed("completed", nil)},
	))})
	down, _ := runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)

	want := []map[string]any{{
		"type": "thought_signature", "signature": signature, "bookmark": map[string]any{"text": "Hello there."},
	}}
	if got := storedSignatures(down); !reflect.DeepEqual(got, want) {
		t.Errorf("stored = %v, want %v", got, want)
	}
}

// callRecorder is a get_weather handler recording each call it gets.
type callRecorder struct {
	mu    sync.Mutex
	calls []llm.FunctionCallParams
	done  chan struct{}
	want  int
}

func newCallRecorder(want int) *callRecorder {
	return &callRecorder{done: make(chan struct{}), want: want}
}

func (r *callRecorder) handle(ctx context.Context, params llm.FunctionCallParams) error {
	r.mu.Lock()
	r.calls = append(r.calls, params)
	if len(r.calls) == r.want {
		close(r.done)
	}
	r.mu.Unlock()
	return params.Result(ctx, `{"temperature_f": 72}`, nil)
}

func toolContext() *frames.LLMContext {
	convo := userContext("Hi")
	convo.SetTools([]frames.Tool{weatherTool})
	return convo
}

// TestFunctionCall checks a function call's arguments are collected and the call
// run.
func TestFunctionCall(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		signedThought(0), functionCall(1, "call_1", "Paris"), []string{completed("requires_action", nil)},
	))})
	svc := makeService(api, Config{})
	rec := newCallRecorder(1)
	svc.RegisterFunction("get_weather", rec.handle)

	down, _ := runContext(t, svc, toolContext(), pipeline.Params{}, rec.done)

	started := 0
	for _, f := range down {
		if _, ok := f.(*frames.FunctionCallsStartedFrame); ok {
			started++
		}
	}
	if started != 1 {
		t.Errorf("FunctionCallsStartedFrame = %d, want 1", started)
	}
	if rec.calls[0].ToolCallID != "call_1" || !jsonEqual(rec.calls[0].Arguments, `{"city":"Paris"}`) {
		t.Errorf("call = %s %s", rec.calls[0].ToolCallID, rec.calls[0].Arguments)
	}
	if got := storedSignatures(down)[0]["bookmark"]; !reflect.DeepEqual(got, map[string]any{"function_call": "call_1"}) {
		t.Errorf("bookmark = %v, want the call", got)
	}
}

func jsonEqual(raw json.RawMessage, want string) bool {
	var a, b any
	_ = json.Unmarshal(raw, &a)
	_ = json.Unmarshal([]byte(want), &b)
	return reflect.DeepEqual(a, b)
}

// TestParallelCallsSignedOnce checks parallel calls all run, with the signature
// stored for the first.
func TestParallelCallsSignedOnce(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		signedThought(0), functionCall(1, "call_1", "Paris"), functionCall(2, "call_2", "Tokyo"),
		[]string{completed("requires_action", nil)},
	))})
	svc := makeService(api, Config{})
	rec := newCallRecorder(2)
	svc.RegisterFunction("get_weather", rec.handle)

	down, _ := runContext(t, svc, toolContext(), pipeline.Params{}, rec.done)

	var cities []string
	for _, c := range rec.calls {
		var args struct{ City string }
		_ = json.Unmarshal(c.Arguments, &args)
		cities = append(cities, args.City)
	}
	slices.Sort(cities)
	if !reflect.DeepEqual(cities, []string{"Paris", "Tokyo"}) {
		t.Errorf("cities = %v", cities)
	}
	stored := storedSignatures(down)
	if len(stored) != 1 || !reflect.DeepEqual(stored[0]["bookmark"], map[string]any{"function_call": "call_1"}) {
		t.Errorf("stored = %v, want one signature for call_1", stored)
	}
}

// TestThoughtSummary checks a thought summary is bracketed as one thought.
func TestThoughtSummary(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		[]string{
			start(0, map[string]any{"type": "thought"}),
			delta(0, map[string]any{"type": "thought_summary", "content": map[string]any{"type": "text", "text": "Hmm."}}),
			stop(0),
		},
		textReply("Five cents.", 1), []string{completed("completed", nil)},
	))})
	down, _ := runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)

	var thought []frames.Frame
	for _, f := range down {
		switch f.(type) {
		case *frames.LLMThoughtStartFrame, *frames.LLMThoughtTextFrame, *frames.LLMThoughtEndFrame:
			thought = append(thought, f)
		}
	}
	if len(thought) != 3 {
		t.Fatalf("thought frames = %v, want start, text, end", thought)
	}
	_, isStart := thought[0].(*frames.LLMThoughtStartFrame)
	text, isText := thought[1].(*frames.LLMThoughtTextFrame)
	_, isEnd := thought[2].(*frames.LLMThoughtEndFrame)
	if !isStart || !isText || !isEnd || text.Text != "Hmm." {
		t.Errorf("thought frames = %v", thought)
	}
}

// TestTokenUsage checks thought tokens count as output and tool-use tokens as
// input.
func TestTokenUsage(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(textReply("Hi.", 1), []string{completed("completed", map[string]any{
		"total_input_tokens":    10,
		"total_tool_use_tokens": 4,
		"total_output_tokens":   3,
		"total_thought_tokens":  20,
		"total_cached_tokens":   2,
	})}))})
	down, _ := runContext(t, makeService(api, Config{}), userContext("Hi"),
		pipeline.Params{EnableMetrics: true, EnableUsageMetrics: true}, nil)

	var usage []frames.LLMTokenUsage
	for _, f := range down {
		if m, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range m.Data {
				if u, ok := d.(frames.LLMUsageMetricsData); ok {
					usage = append(usage, u.Value)
				}
			}
		}
	}
	if len(usage) != 1 {
		t.Fatalf("usage reports = %d, want 1", len(usage))
	}
	u := usage[0]
	if u.PromptTokens != 14 || u.CompletionTokens != 23 || u.TotalTokens != 37 ||
		u.ReasoningTokens == nil || *u.ReasoningTokens != 20 || u.CacheReadTokens == nil || *u.CacheReadTokens != 2 {
		t.Errorf("usage = %+v", u)
	}
}

// TestIncompleteResponseWarns checks a response cut short (by the token limit,
// say) is logged as incomplete.
func TestIncompleteResponseWarns(t *testing.T) {
	logged := capturedLogs(t, slog.LevelWarn)
	api := newFakeAPI(t, reply{body: sse(eventsOf(textReply("Han", 1), []string{completed("incomplete", nil)}))})
	runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)

	if !strings.Contains(logged(), "response incomplete, the interaction ended incomplete") {
		t.Errorf("logs = %q", logged())
	}
}

// TestEmptyResponseReported checks a completed response with no text or function
// call is reported as an error.
func TestEmptyResponseReported(t *testing.T) {
	logged := capturedLogs(t, slog.LevelWarn)
	api := newFakeAPI(t, reply{body: sse(eventsOf(
		signedThought(0), textReply(" ", 1), []string{completed("completed", nil)},
	))})
	_, up := runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)

	if !strings.Contains(logged(), "completed with no text or function call") ||
		!strings.Contains(logged(), "steps=\"[thought model_output]\"") {
		t.Errorf("logs = %q", logged())
	}
	want := []string{"LLM response completed with no text or function call"}
	if got := errorMessages(up); !reflect.DeepEqual(got, want) {
		t.Errorf("errors = %v", got)
	}
}

// TestStalledStreamTimesOut checks a stream that stops producing ends the
// response with a timeout error.
func TestStalledStreamTimesOut(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse([]string{start(0, map[string]any{"type": "model_output"})}), stall: true})
	idle := 50 * time.Millisecond
	svc := makeService(api, Config{StreamIdleTimeout: &idle})
	down, up := runContext(t, svc, userContext("Hi"), pipeline.Params{}, nil)

	if _, ok := down[len(down)-1].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("last frame = %T, want the end of the response", down[len(down)-1])
	}
	if got := errorMessages(up); !reflect.DeepEqual(got, []string{"LLM completion timeout"}) {
		t.Errorf("errors = %v", got)
	}
}

// TestRetryOnTimeoutReissuesTheRequest checks a request whose first event never
// arrives is re-issued once.
func TestRetryOnTimeoutReissuesTheRequest(t *testing.T) {
	api := newFakeAPI(t,
		reply{body: "", stall: true},
		reply{body: sse(eventsOf(textReply("Hi.", 1), []string{completed("completed", nil)}))},
	)
	svc := makeService(api, Config{RetryOnTimeout: true, RetryTimeout: 100 * time.Millisecond})
	down, _ := runContext(t, svc, userContext("Hi"), pipeline.Params{}, nil)

	var text strings.Builder
	for _, f := range down {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			text.WriteString(tf.Text)
		}
	}
	if text.String() != "Hi." || len(api.requests) != 2 {
		t.Errorf("text = %q after %d requests, want the retry's answer", text.String(), len(api.requests))
	}
}

// TestStreamErrorEventReported checks an error event is pushed as an error.
func TestStreamErrorEventReported(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse([]string{
		evt(map[string]any{"event_type": "error", "error": map[string]any{"code": "x", "message": "overloaded"}}),
	})})
	_, up := runContext(t, makeService(api, Config{}), userContext("Hi"), pipeline.Params{}, nil)
	if got := errorMessages(up); !reflect.DeepEqual(got, []string{"LLM response error: overloaded"}) {
		t.Errorf("errors = %v", got)
	}
}

// requestFor runs one plain turn and returns the request it sent.
func requestFor(t *testing.T, cfg Config, convo *frames.LLMContext, setup func(*LLMService)) map[string]any {
	t.Helper()
	api := newFakeAPI(t, reply{body: sse(eventsOf(textReply("Hi.", 1), []string{completed("completed", nil)}))})
	svc := makeService(api, cfg)
	if setup != nil {
		setup(svc)
	}
	if convo == nil {
		convo = userContext("Hi")
	}
	runContext(t, svc, convo, pipeline.Params{}, nil)
	return api.request(t, 0)
}

func genConfig(request map[string]any) map[string]any {
	g, _ := request["generation_config"].(map[string]any)
	return g
}

// TestStatelessStreamedRequest checks the request streams, stores nothing, and
// names the model.
func TestStatelessStreamedRequest(t *testing.T) {
	request := requestFor(t, Config{}, nil, nil)
	if request["stream"] != true || request["store"] != false || request["model"] != "gemini-3.6-flash" {
		t.Errorf("request = %v", request)
	}
	input, _ := request["input"].([]any)
	first, _ := input[0].(map[string]any)
	if first["type"] != "user_input" {
		t.Errorf("input = %v", input)
	}
	if _, ok := request["tools"]; ok {
		t.Errorf("tools = %v, want none", request["tools"])
	}
}

// TestGenerationSettings checks sampling settings and the system instruction
// reach the request.
func TestGenerationSettings(t *testing.T) {
	temperature, seed := 0.2, 7
	request := requestFor(t, Config{Temperature: &temperature, MaxTokens: 100, Seed: &seed}, nil,
		func(s *LLMService) { s.SetSystemInstruction("Be brief.") })
	g := genConfig(request)
	if request["system_instruction"] != "Be brief." || g["temperature"] != 0.2 ||
		g["max_output_tokens"] != float64(100) || g["seed"] != float64(7) {
		t.Errorf("request = %v", request)
	}
}

// TestThinkingDefault checks a listed model gets its lowest thinking level when
// none is set.
func TestThinkingDefault(t *testing.T) {
	if got := genConfig(requestFor(t, Config{}, nil, nil))["thinking_level"]; got != "minimal" {
		t.Errorf("thinking_level = %v, want minimal", got)
	}
}

// thinkingFor is the request's generation config for a model, and the warnings
// logged building it.
func thinkingFor(t *testing.T, model string) (map[string]any, string) {
	t.Helper()
	logged := capturedLogs(t, slog.LevelWarn)
	g := genConfig(requestFor(t, Config{Model: model}, nil, nil))
	return g, logged()
}

// TestUnlistedModelThatCanThinkWarnedAbout checks an alias, which has no
// default, is warned about as the Gemini service does.
func TestUnlistedModelThatCanThinkWarnedAbout(t *testing.T) {
	g, warnings := thinkingFor(t, "gemini-flash-latest")
	if _, ok := g["thinking_level"]; ok {
		t.Errorf("generation_config = %v, want no thinking level", g)
	}
	if !strings.Contains(warnings, "no thinking default") || !strings.Contains(warnings, "model=gemini-flash-latest") {
		t.Errorf("logs = %q", warnings)
	}
}

// TestGemini25RunsAtGooglesDefaultWithoutWarning checks Gemini 2.5, whose
// default is a budget, gets no level and no warning.
func TestGemini25RunsAtGooglesDefaultWithoutWarning(t *testing.T) {
	for _, model := range []string{"gemini-2.5-flash", "gemini-2.5-flash-preview"} {
		t.Run(model, func(t *testing.T) {
			g, warnings := thinkingFor(t, model)
			if _, ok := g["thinking_level"]; ok {
				t.Errorf("generation_config = %v, want no thinking level", g)
			}
			if strings.Contains(warnings, "thinking default") {
				t.Errorf("logs = %q, want no warning", warnings)
			}
		})
	}
}

// TestThinkingSettings checks an explicit thinking level and thought summaries
// replace the default.
func TestThinkingSettings(t *testing.T) {
	g := genConfig(requestFor(t, Config{Thinking: &ThinkingConfig{Level: "high", IncludeThoughts: true}}, nil, nil))
	if g["thinking_level"] != "high" || g["thinking_summaries"] != "auto" {
		t.Errorf("generation_config = %v", g)
	}
}

// TestProviderThinkingDefault checks use_provider_thinking_default leaves
// thinking to Google.
func TestProviderThinkingDefault(t *testing.T) {
	g := genConfig(requestFor(t, Config{}, nil, func(s *LLMService) { s.SetUseProviderThinkingDefault(true) }))
	if _, ok := g["thinking_level"]; ok {
		t.Errorf("generation_config = %v, want no thinking level", g)
	}
}

// TestUnsupportedSettingsIgnoredWithWarning checks top_k and a thinking budget
// are left out, each warned about.
func TestUnsupportedSettingsIgnoredWithWarning(t *testing.T) {
	logged := capturedLogs(t, slog.LevelWarn)
	topK, budget := 5, 100
	g := genConfig(requestFor(t, Config{TopK: &topK, Thinking: &ThinkingConfig{Budget: &budget}}, nil, nil))
	if _, ok := g["top_k"]; ok {
		t.Errorf("generation_config = %v, want no top_k", g)
	}
	if !strings.Contains(logged(), "ignoring top_k") || !strings.Contains(logged(), "ignoring thinking_budget") {
		t.Errorf("logs = %q", logged())
	}
}

// TestSafetySettingsConverted checks generateContent safety settings take the
// Interactions API's names.
func TestSafetySettingsConverted(t *testing.T) {
	request := requestFor(t, Config{SafetySettings: []SafetySetting{{
		Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH",
	}}}, nil, nil)
	want := []any{map[string]any{"type": "harassment", "threshold": "block_only_high"}}
	if !reflect.DeepEqual(request["safety_settings"], want) {
		t.Errorf("safety_settings = %v", request["safety_settings"])
	}
}

// TestToolConfigConverted checks a function_calling_config becomes the
// Interactions API's tool_choice.
func TestToolConfigConverted(t *testing.T) {
	g := genConfig(requestFor(t, Config{ToolConfig: map[string]any{
		"function_calling_config": map[string]any{"mode": "ANY", "allowed_function_names": []any{"a"}},
	}}, nil, nil))
	want := map[string]any{"allowed_tools": map[string]any{"mode": "any", "tools": []any{"a"}}}
	if !reflect.DeepEqual(g["tool_choice"], want) {
		t.Errorf("tool_choice = %v", g["tool_choice"])
	}
}

// TestContextTools checks the conversation's tools are sent as function tools.
func TestContextTools(t *testing.T) {
	request := requestFor(t, Config{}, toolContext(), nil)
	tools, _ := request["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v, want one", request["tools"])
	}
	if first, _ := tools[0].(map[string]any); first["type"] != "function" {
		t.Errorf("tools = %v", request["tools"])
	}
}

// TestServiceToolsUsedWithoutContextTools checks tools given to the service, in
// generateContent format, fill in for the conversation's.
func TestServiceToolsUsedWithoutContextTools(t *testing.T) {
	request := requestFor(t, Config{Tools: []map[string]any{{"google_search": map[string]any{}}}}, nil, nil)
	want := []any{map[string]any{"type": "google_search"}}
	if !reflect.DeepEqual(request["tools"], want) {
		t.Errorf("tools = %v", request["tools"])
	}
}

// TestExtra checks extra request fields are merged in, generation_config field
// by field.
func TestExtra(t *testing.T) {
	request := requestFor(t, Config{Extra: map[string]any{
		"service_tier": "flex", "generation_config": map[string]any{"stop_sequences": []any{"END"}},
	}}, nil, nil)
	g := genConfig(request)
	if request["service_tier"] != "flex" || !reflect.DeepEqual(g["stop_sequences"], []any{"END"}) ||
		g["thinking_level"] != "minimal" {
		t.Errorf("request = %v", request)
	}
}

// TestRequestIsAuthorized checks the request carries the API key and asks for an
// event stream.
func TestRequestIsAuthorized(t *testing.T) {
	api := newFakeAPI(t, reply{body: sse(eventsOf(textReply("Hi.", 1), []string{completed("completed", nil)}))})
	shaper := apiKeyShaperAt{apiKeyShaper{apiKey: "secret"}, api.URL}
	svc := NewShapedLLM("GoogleInteractionsLLM", shaper, Config{APIKey: "secret"})
	runContext(t, svc, userContext("Hi"), pipeline.Params{}, nil)
	if h := api.headers[0]; h.Get("x-goog-api-key") != "secret" || h.Get("Accept") != "text/event-stream" {
		t.Errorf("headers = %v", h)
	}
}

// apiKeyShaperAt is the Gemini API's authorization, sent to the fake API.
type apiKeyShaperAt struct {
	apiKeyShaper
	url string
}

func (s apiKeyShaperAt) Endpoint() string { return s.url }

// TestNewLLMDefaults checks the service's label, endpoint and defaults.
func TestNewLLMDefaults(t *testing.T) {
	svc := NewLLM(Config{APIKey: "k"})
	providertest.Service(t, "GoogleInteractionsLLM", svc)
	if svc.Model() != defaultModel || svc.cfg.MaxTokens != 4096 {
		t.Errorf("model = %s, max tokens = %d", svc.Model(), svc.cfg.MaxTokens)
	}
	if got := svc.shaper.Endpoint(); got != "https://generativelanguage.googleapis.com/v1beta/interactions" {
		t.Errorf("endpoint = %s", got)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Error("a config without an API key validated")
	}
}
