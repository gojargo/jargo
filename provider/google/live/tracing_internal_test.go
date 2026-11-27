package live

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// The llm_setup span is the one span the service raises itself. The model's
// replies and the function calls it makes are traced from the frames the
// service pushes, which the service-level tests below drive with server
// messages.

// tracedService is a service set up on a pipeline traced or not, with its
// spans recorded. Its session is one that is never acknowledged: the tests
// drive the message handler themselves.
func tracedService(t *testing.T, cfg Config, traced bool) (*LLMService, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	cfg.APIKey = "k"
	s := New(cfg)
	silent := newSilentSession(t)
	s.connector = apiKeyConnector{baseURL: "ws" + strings.TrimPrefix(silent.URL, "http"), apiKey: "k"}
	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem(), TracingEnabled: traced, Tracing: tracing.NewContext()}
	if err := s.Setup(ctx, setup); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = s.Cleanup(ctx) })
	return s, rec
}

// spansNamed are the attributes of the ended spans named name.
func spansNamed(rec *tracetest.SpanRecorder, name string) []map[string]string {
	var out []map[string]string
	for _, sp := range rec.Ended() {
		if sp.Name() != name {
			continue
		}
		attrs := map[string]string{}
		for _, kv := range sp.Attributes() {
			attrs[string(kv.Key)] = kv.Value.String()
		}
		out = append(out, attrs)
	}
	return out
}

func TestSetupSpanRecordsToolsAndSystemInstruction(t *testing.T) {
	s, rec := tracedService(t, Config{
		Model:        "gemini-live-test",
		Instructions: "Be brief.",
		Tools: frames.ToolsSchema{Standard: []frames.Tool{{
			Name:        "get_weather",
			Description: "Weather lookup",
			Parameters: json.RawMessage(
				`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}}},
	}, true)

	s.traceSetup(context.Background())

	spans := spansNamed(rec, "llm_setup")
	if len(spans) != 1 {
		t.Fatalf("llm_setup spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.operation.name":      "llm_setup",
		"gen_ai.request.model":       "gemini-live-test",
		"gen_ai.provider.name":       "gcp.gemini",
		"tools.names":                "get_weather",
		"gen_ai.system_instructions": "Be brief.",
		"tools.definitions": `[{"description":"Weather lookup","name":"get_weather",` +
			`"properties":{"city":{"type":"string"}},"required":["city"]}]`,
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
}

func TestSetupSpanTracingDisabledRecordsNothing(t *testing.T) {
	s, rec := tracedService(t, Config{Model: "gemini-live-test"}, false)

	s.traceSetup(context.Background())

	if got := rec.Ended(); len(got) != 0 {
		t.Errorf("spans = %d, want none on a pipeline that is not traced", len(got))
	}
}

// midReply is a traced service in the middle of a spoken reply.
func midReply(t *testing.T, model string) (*LLMService, *tracetest.SpanRecorder) {
	t.Helper()
	s, rec := tracedService(t, Config{Model: model}, true)
	s.handle(context.Background(), decode(t, audioMessage([]byte{1, 2})))
	return s, rec
}

func TestTurnCompleteProducesOneSpanWithUsage(t *testing.T) {
	s, rec := midReply(t, "models/gemini-3.8-live")

	s.handle(context.Background(), decode(t, `{"serverContent":{"turnComplete":true},"usageMetadata":{`+
		`"promptTokenCount":300,"responseTokenCount":40,`+
		`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":250}]}}`))

	spans := spansNamed(rec, "llm_response")
	if len(spans) != 1 {
		t.Fatalf("llm_response spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.provider.name":            "gcp.gemini",
		"gen_ai.request.model":            "models/gemini-3.8-live",
		"gen_ai.usage.input_tokens":       "300",
		"gen_ai.usage.output_tokens":      "40",
		"gen_ai.usage.audio.input_tokens": "250",
		"llm.interrupted":                 "false",
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
	if got := spansNamed(rec, "stt"); len(got) != 0 {
		t.Errorf("stt spans = %v, want none", got)
	}
}

// A thinking model reports usage per model call; the reply's span adds them
// up. This needs usage to be reported before the turn is completed: the second
// report lands on the reply only while it is still open.
func TestHeldThinkingTurnSumsUsageOfEveryChunk(t *testing.T) {
	s, rec := midReply(t, "models/gemini-3.8-live-extended-thinking")
	ctx := context.Background()

	s.handle(ctx, decode(t, `{"serverContent":{"turnComplete":true,"interactionStatus":"IN_PROGRESS"},`+
		`"usageMetadata":{"promptTokenCount":3228,"responseTokenCount":61,"thoughtsTokenCount":104}}`))
	if got := spansNamed(rec, "llm_response"); len(got) != 0 {
		t.Fatalf("llm_response spans = %v, want none while the turn is held", got)
	}

	s.handle(ctx, decode(t, `{"serverContent":{"turnComplete":true,"interactionStatus":"IDLE"},`+
		`"usageMetadata":{"promptTokenCount":3368,"responseTokenCount":20,"thoughtsTokenCount":10}}`))

	spans := spansNamed(rec, "llm_response")
	if len(spans) != 1 {
		t.Fatalf("llm_response spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.usage.input_tokens":            "6596",
		"gen_ai.usage.output_tokens":           "195",
		"gen_ai.usage.reasoning.output_tokens": "114",
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
}

// (Not a port of an upstream test.) A function call the model makes is traced
// as one llm_tool_call span from the frames the service pushes, and the spans
// this service used to raise for it are gone.
func TestToolCallIsTracedFromThePushedFrames(t *testing.T) {
	s, rec := tracedService(t, Config{Model: "gemini-live-test"}, true)
	ctx := context.Background()
	args := json.RawMessage(`{"city":"Rome"}`)
	call := frames.NewFunctionCallInProgressFrame("call-1", "get_weather", args, true, "")
	if err := s.PushFrame(ctx, call, processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}
	result := frames.NewFunctionCallResultFrame("call-1", "get_weather", args, `{"temp_c":21}`)
	if err := s.PushFrame(ctx, result, processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}

	spans := spansNamed(rec, "llm_tool_call")
	if len(spans) != 1 {
		t.Fatalf("llm_tool_call spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.operation.name":      "execute_tool",
		"gen_ai.tool.name":           "get_weather",
		"gen_ai.tool.call.id":        "call-1",
		"gen_ai.tool.call.arguments": `{"city":"Rome"}`,
		"gen_ai.tool.call.result":    `{"temp_c":21}`,
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
	for k := range spans[0] {
		if strings.HasPrefix(k, "tool.") {
			t.Errorf("attr %q recorded, want none of the old tool attributes", k)
		}
	}
	if got := spansNamed(rec, "llm_tool_result"); len(got) != 0 {
		t.Errorf("llm_tool_result spans = %v, want none", got)
	}
}
