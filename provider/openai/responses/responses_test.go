package responses

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	responsesadapter "github.com/gojargo/jargo/adapter/responses"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// mustRequest builds a request and fails the test if the conversion did.
func mustRequest(t *testing.T, cfg Config, convo *frames.LLMContext) request {
	t.Helper()
	req, err := cfg.newRequest(&responsesadapter.Adapter{}, convo, adapter.Options{}, false)
	if err != nil {
		t.Fatalf("newRequest: %v", err)
	}
	cfg.applyThinkingDefault(&req, nil)
	return req
}

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: Config{}, Valid: false},
		{Name: "API key only", Cfg: Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "OpenAIResponsesLLM", NewLLM(Config{APIKey: "k"}))
	providertest.Service(t, "OpenAIResponsesHTTPLLM", NewHTTPLLM(Config{APIKey: "k"}))
}

// recordedBody starts a fake Responses endpoint that answers every request with
// a short streamed reply, or a one-shot one for an inference, and returns the
// service pointed at it with a reader for the last request body.
func recordedBody(t *testing.T, cfg Config) (*HTTPService, func() map[string]any) {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] == false {
			_, _ = w.Write([]byte(`{"output":[{"content":[{"type":"output_text","text":"ok"}]}]}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_text.delta","delta":"hi"}`,
			`{"type":"response.completed","response":{"id":"resp_1"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	}))
	t.Cleanup(srv.Close)
	cfg.APIKey, cfg.BaseURL = "k", srv.URL
	svc := NewHTTPLLM(cfg)
	svc.http = srv.Client()
	return svc, func() map[string]any { return body }
}

// generateHTTP runs one streamed turn and fails the test if it failed.
func generateHTTP(t *testing.T, svc *HTTPService, convo *frames.LLMContext) {
	t.Helper()
	if err := svc.Generate(context.Background(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
}

// inputOf returns a request body's input list as JSON, for comparison. The
// body was decoded into a map, so each item's keys come out sorted.
func inputOf(t *testing.T, body map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(body["input"])
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// TestConfiguredInstructionsAreTheBaseInstruction checks the configured prompt
// is the base the service composes its own guidance onto, so an addition is
// sent after it rather than in its place.
func TestConfiguredInstructionsAreTheBaseInstruction(t *testing.T) {
	svc, body := recordedBody(t, Config{Instructions: "You are a booking assistant."})
	svc.AppendSystemInstruction("Answer in one sentence.")
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")

	generateHTTP(t, svc, convo)
	if got := body()["instructions"]; got != "You are a booking assistant.\n\nAnswer in one sentence." {
		t.Errorf("instructions = %q, want the configured prompt followed by the addition", got)
	}
}

// TestContextPromptKeptBesideComposedInstruction checks the conversation's
// prompt still reaches the model when the service composes an instruction of
// its own: it is the developer message at the head of the input.
func TestContextPromptKeptBesideComposedInstruction(t *testing.T) {
	svc, body := recordedBody(t, Config{Instructions: "from config"})
	svc.AppendSystemInstruction("Answer in one sentence.")
	convo := frames.NewLLMContext("from context")
	convo.AddUserMessage("hi")

	generateHTTP(t, svc, convo)
	b := body()
	if b["instructions"] != "from config\n\nAnswer in one sentence." {
		t.Errorf("instructions = %q, want the composed instruction", b["instructions"])
	}
	want := `[{"content":"from context","role":"developer","type":"message"},` +
		`{"content":"hi","role":"user","type":"message"}]`
	if got := inputOf(t, b); got != want {
		t.Errorf("input = %s, want %s", got, want)
	}
}

// TestInferenceUsesTheServiceInstruction checks an inference given no
// instruction of its own sends the service's, and one given an instruction
// sends that in its place.
func TestInferenceUsesTheServiceInstruction(t *testing.T) {
	svc, body := recordedBody(t, Config{Instructions: "Original instruction"})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")

	if _, err := svc.RunInference(context.Background(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got := body()["instructions"]; got != "Original instruction" {
		t.Errorf("instructions = %v, want the service's own", got)
	}

	if _, err := svc.RunInference(context.Background(), convo,
		llm.InferenceOptions{SystemInstruction: "New system instruction"}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	b := body()
	if b["instructions"] != "New system instruction" {
		t.Errorf("instructions = %v, want the call's own", b["instructions"])
	}
	if got := inputOf(t, b); got != `[{"content":"Hello","role":"user","type":"message"}]` {
		t.Errorf("input = %s, want the user message alone", got)
	}
}

// TestInferenceOnAnEmptyConversation checks an inference over a conversation
// with nothing in it sends its instruction once, as the developer message the
// API needs, and no instructions field beside it. That holds for the service's
// instruction and for the call's own alike.
func TestInferenceOnAnEmptyConversation(t *testing.T) {
	for _, c := range []struct {
		name, configured, given, want string
	}{
		{name: "service instruction", configured: "You are helpful", want: "You are helpful"},
		{name: "call instruction", given: "Summarize the conversation", want: "Summarize the conversation"},
	} {
		t.Run(c.name, func(t *testing.T) {
			svc, body := recordedBody(t, Config{Instructions: c.configured})
			if _, err := svc.RunInference(context.Background(), frames.NewLLMContext(""),
				llm.InferenceOptions{SystemInstruction: c.given}); err != nil {
				t.Fatalf("RunInference: %v", err)
			}
			b := body()
			if _, ok := b["instructions"]; ok {
				t.Errorf("instructions = %v, want none", b["instructions"])
			}
			want := `[{"content":"` + c.want + `","role":"developer","type":"message"}]`
			if got := inputOf(t, b); got != want {
				t.Errorf("input = %s, want %s", got, want)
			}
		})
	}
}

// TestServiceOffersCancelTools checks the service holds the adapter it converts
// through, which is where the base puts the tools the service implements
// itself (a cancel tool, say), so they reach the request.
func TestServiceOffersCancelTools(t *testing.T) {
	var _ llm.AdapterHolder = (*HTTPService)(nil)
	var _ llm.AdapterHolder = (*Service)(nil)

	svc, body := recordedBody(t, Config{})
	svc.LLMAdapter().SetBuiltin(adapter.Builtin{Tool: frames.Tool{
		Name: "cancel_lookup", Parameters: json.RawMessage(`{"type":"object"}`),
	}})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")

	if err := svc.GenerateWithTools(context.Background(), convo, &recordingSink{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	tools, _ := body()["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v, want the built-in tool", body()["tools"])
	}
	if tool, _ := tools[0].(map[string]any); tool["name"] != "cancel_lookup" {
		t.Errorf("tool = %v, want the built-in tool", tool)
	}
}

// TestEncodeBodyExtraOverrides checks configured extra fields reach the request
// and win over the modeled ones.
func TestEncodeBodyExtraOverrides(t *testing.T) {
	req := request{Model: "gpt-4.1", Stream: true}
	raw, err := encodeBody(req, map[string]any{"model": "gpt-5", "custom": 7})
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["model"] != "gpt-5" {
		t.Errorf("model = %v, want the extra field to win", m["model"])
	}
	if m["custom"] != float64(7) {
		t.Errorf("custom = %v, want the extra field carried through", m["custom"])
	}
}

// TestNewRequestOmitsUnset checks the optional controls are absent from the wire
// when unset, leaving the API defaults in place.
func TestNewRequestOmitsUnset(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")

	raw, err := encodeBody(mustRequest(t, Config{Model: "gpt-4.1"}, convo), nil)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{
		"temperature", "top_p", "max_output_tokens", "service_tier", "tools",
		"instructions", "previous_response_id",
	} {
		if _, ok := m[key]; ok {
			t.Errorf("%s is present when unset, want it omitted", key)
		}
	}
	if m["store"] != false {
		t.Errorf("store = %v, want false so conversations are not retained", m["store"])
	}
	if m["stream"] != true {
		t.Errorf("stream = %v, want true", m["stream"])
	}
}

// TestHTTPRetriesARateLimitedTurn checks a turn the API refuses for now (a rate
// limit, asking to wait a moment) is sent again rather than lost.
func TestHTTPRetriesARateLimitedTurn(t *testing.T) {
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		if requests == 1 {
			w.Header().Set("Retry-After-Ms", "10")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_text.delta","delta":"hi"}`,
			`{"type":"response.completed","response":{"id":"resp_1"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	}))
	t.Cleanup(srv.Close)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL})
	svc.http = srv.Client()

	var text string
	err := svc.Generate(context.Background(), userConvo("hi"), func(s string) error {
		text += s
		return nil
	})
	if err != nil || text != "hi" || requests != 2 {
		t.Errorf("turn = (%q, %v) after %d requests, want hi after a retry", text, err, requests)
	}
}

// TestFailedResponseReportsItsUsage checks a response that failed still reports
// the tokens it was billed for, beside the error.
func TestFailedResponseReportsItsUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"type":"response.failed","response":{"id":"r","status":"failed",` +
			`"error":{"code":"server_error","message":"server_error"},` +
			`"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}}}` + "\n\n"))
	}))
	t.Cleanup(srv.Close)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL})

	var mu sync.Mutex
	var usage []frames.LLMTokenUsage
	var errs []string
	ended := make(chan struct{}, 1)
	noInitial := false
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics: true, EnableUsageMetrics: true, SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.MetricsFrame:
			for _, d := range fr.Data {
				if u, ok := d.(frames.LLMUsageMetricsData); ok {
					usage = append(usage, u.Value)
				}
			}
		case *frames.LLMFullResponseEndFrame:
			ended <- struct{}{}
		}
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errs = append(errs, ef.Error)
			mu.Unlock()
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(userConvo("hi")))
	select {
	case <-ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the response did not end")
	}
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if len(usage) != 1 || usage[0].PromptTokens != 12 || usage[0].CompletionTokens != 3 {
		t.Errorf("usage = %+v, want the failed response's 12 and 3 tokens", usage)
	}
	if len(errs) != 1 || errs[0] != "LLM response error: server_error" {
		t.Errorf("errors = %v, want the failure reported once", errs)
	}
}
