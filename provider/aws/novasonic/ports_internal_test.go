package novasonic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Ports of TestNovaSonicIntermediateResults in
// tests/test_realtime_intermediate_results.py: Nova Sonic takes a tool's
// intermediate results as text beside the call, and appended messages as text,
// interactive as asked.

const weather = `{"temperature":75}`

// toolResults counts the tool results sent.
func (f *fakeStream) toolResults() int {
	n := 0
	for _, e := range f.events() {
		if e.name == "toolResult" {
			n++
		}
	}
	return n
}

// liveStream is a harness whose session is set up, with the stream it runs on.
func liveStream(t *testing.T) (*harness, *fakeStream) {
	t.Helper()
	h := newHarness(t, Config{})
	h.seed(t, frames.NewLLMContext(""))
	return h, h.client.stream(t, 0)
}

// textsAfterSetup are the text blocks sent after the session was set up.
func textsAfterSetup(st *fakeStream, before int) []sentText { return st.texts()[before:] }

func novaResult(final, run bool) *frames.FunctionCallResultFrame {
	f := frames.NewFunctionCallResultFrame("call_1", "delegate", json.RawMessage(`{}`), weather)
	f.Properties = &frames.FunctionCallResultProperties{IsFinal: &final, RunLLM: &run}
	return f
}

func TestAnIntermediateResultGoesInAsInteractiveText(t *testing.T) {
	h, st := liveStream(t)
	before := len(st.texts())
	h.s.handleFunctionCallResult(context.Background(), novaResult(false, true))

	texts := textsAfterSetup(st, before)
	if len(texts) != 1 || !texts[0].interactive {
		t.Fatalf("texts = %+v, want one interactive", texts)
	}
	p, ok := frames.ParseAsyncToolMessage(frames.Message{Role: frames.RoleDeveloper, Text: texts[0].text})
	if !ok || p.Kind != frames.AsyncToolIntermediate {
		t.Errorf("text = %q, want the intermediate result envelope", texts[0].text)
	}
	if n := st.toolResults(); n != 0 {
		t.Errorf("tool results sent = %d, want none", n)
	}
}

func TestASilentResultGoesInWithoutAskingForAReply(t *testing.T) {
	h, st := liveStream(t)
	before := len(st.texts())
	h.s.handleFunctionCallResult(context.Background(), novaResult(false, false))
	if texts := textsAfterSetup(st, before); len(texts) != 1 || texts[0].interactive {
		t.Fatalf("texts = %+v, want one not interactive", texts)
	}
}

func TestAFinalResultSettlesTheCall(t *testing.T) {
	h, st := liveStream(t)
	before := len(st.texts())
	h.s.handleFunctionCallResult(context.Background(), novaResult(true, true))
	if texts := textsAfterSetup(st, before); len(texts) != 0 {
		t.Errorf("texts = %+v, want none", texts)
	}
	if n := st.toolResults(); n != 1 {
		t.Errorf("tool results sent = %d, want one", n)
	}
	if !h.s.completed("call_1") {
		t.Error("the call is not settled")
	}
}

func TestAnAppendedMessageGoesInAsTextInteractiveAsAsked(t *testing.T) {
	h, st := liveStream(t)
	before := len(st.texts())
	run := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleDeveloper, Text: "Backend: On it."}})
	run.RunLLM = true
	quiet := frames.NewLLMMessagesAppendFrame(
		[]frames.Message{{Role: frames.RoleDeveloper, Text: "Backend (working): read_file()"}})
	h.s.handleMessagesAppend(context.Background(), run)
	h.s.handleMessagesAppend(context.Background(), quiet)

	want := []sentText{
		{role: "USER", text: "Backend: On it.", interactive: true},
		{role: "USER", text: "Backend (working): read_file()", interactive: false},
	}
	got := textsAfterSetup(st, before)
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("texts = %+v, want %+v", got, want)
	}
}

// Ports of tests/test_nova_sonic_usage_metrics.py: a usage event reports its
// delta, the speech and text tokens of each direction added together.

func TestCanGenerateMetrics(t *testing.T) {
	if !New(Config{}).CanGenerateMetrics() {
		t.Error("the service should report metrics")
	}
}

// usage is the token usage the service reported.
func (r *recorder) usage() []frames.LLMTokenUsage {
	var out []frames.LLMTokenUsage
	for _, f := range r.frames() {
		m, ok := f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range m.Data {
			if u, ok := d.(frames.LLMUsageMetricsData); ok {
				out = append(out, u.Value)
			}
		}
	}
	return out
}

func TestUsageEventReportsCollapsedDelta(t *testing.T) {
	h := newHarness(t, Config{})
	h.drive(t, "usageEvent", map[string]any{"details": map[string]any{
		"delta": map[string]any{
			"input":  map[string]any{"speechTokens": 0, "textTokens": 3},
			"output": map[string]any{"speechTokens": 20, "textTokens": 0},
		},
		"total": map[string]any{
			"input":  map[string]any{"speechTokens": 288, "textTokens": 3443},
			"output": map[string]any{"speechTokens": 694, "textTokens": 203},
		},
	}})
	got := h.down.usage()
	if len(got) != 1 {
		t.Fatalf("usage = %+v, want one report", got)
	}
	if u := got[0]; u.PromptTokens != 3 || u.CompletionTokens != 20 || u.TotalTokens != 23 {
		t.Errorf("usage = %+v, want 3 prompt, 20 completion, 23 total", u)
	}
}

func TestUsageEventWithNoTokensIsSkipped(t *testing.T) {
	h := newHarness(t, Config{})
	h.drive(t, "usageEvent", map[string]any{"details": map[string]any{
		"delta": map[string]any{
			"input":  map[string]any{"speechTokens": 0, "textTokens": 0},
			"output": map[string]any{"speechTokens": 0, "textTokens": 0},
		},
	}})
	if got := h.down.usage(); len(got) != 0 {
		t.Errorf("usage = %+v, want none", got)
	}
}

func TestUsageEventWithMissingDetailsIsSafe(t *testing.T) {
	h := newHarness(t, Config{})
	h.drive(t, "usageEvent", map[string]any{})
	if got := h.down.usage(); len(got) != 0 {
		t.Errorf("usage = %+v, want none", got)
	}
}

// Ports of the Nova Sonic binding of tests/test_realtime_service_tool_sync.py.
// A plain list of standard tools is the same Go type as a schema's, so the case
// normalizing one has no counterpart.

func sampleHandler(ctx context.Context, p llm.FunctionCallParams) error {
	return p.Result(ctx, "{}", nil)
}

func sampleTools() frames.ToolsSchema {
	return frames.ToolsSchema{Standard: []frames.Tool{{
		Name: "sample", Description: "A sample tool.", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: llm.FunctionCallHandler(sampleHandler),
	}}}
}

func TestServiceSchemaHandlerRegisters(t *testing.T) {
	s := New(Config{Tools: sampleTools()})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.HasFunction("sample") {
		t.Error("the configured tool's handler was not registered")
	}
}

func TestServiceAsyncToolOptionHonored(t *testing.T) {
	keep := false
	tools := sampleTools()
	tools.Standard[0].CancelOnInterruption = &keep
	s := New(Config{Tools: tools})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.FunctionIsAsync("sample") {
		t.Error("a configured tool not canceled on interruption should register as async")
	}
}

func TestNoServiceToolsIsSafe(t *testing.T) {
	s := New(Config{})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if s.HasFunction("sample") {
		t.Error("nothing should be registered")
	}
}
