package live

import (
	"context"
	"strings"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// Ports of tests/test_gemini_live_tracing.py: the llm_tool_result span carries
// the call, the function and the result.

// tracedService is a service set up to trace, with its spans recorded.
func tracedService(t *testing.T) (*Service, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	s := New(Config{APIKey: "k", Model: "gemini-live-test"})
	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem(), TracingEnabled: true, Tracing: tracing.NewTracingContext()}
	if err := s.Setup(ctx, setup); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = s.Cleanup(ctx) })
	return s, rec
}

// toolResultSpan is the one llm_tool_result span recorded, as attributes.
func toolResultSpan(t *testing.T, rec *tracetest.SpanRecorder) map[attribute.Key]attribute.Value {
	t.Helper()
	spans := rec.Ended()
	if len(spans) != 1 || spans[0].Name() != "llm_tool_result" {
		t.Fatalf("spans = %d, want exactly one llm_tool_result span", len(spans))
	}
	attrs := map[attribute.Key]attribute.Value{}
	for _, kv := range spans[0].Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

func TestCapturesToolResultAttributes(t *testing.T) {
	s, rec := tracedService(t)

	s.traceToolResult(context.Background(), "call-123", "get_weather", map[string]any{"temperature": 72})

	attrs := toolResultSpan(t, rec)
	if attrs["tool.call_id"].AsString() != "call-123" {
		t.Errorf("tool.call_id = %v", attrs["tool.call_id"])
	}
	if attrs["tool.function_name"].AsString() != "get_weather" {
		t.Errorf("tool.function_name = %v", attrs["tool.function_name"])
	}
	if !strings.Contains(attrs["tool.result"].AsString(), `"temperature":72`) {
		t.Errorf("tool.result = %v, want the payload", attrs["tool.result"])
	}
	if attrs["tool.result_status"].AsString() != "completed" {
		t.Errorf("tool.result_status = %v, want completed", attrs["tool.result_status"])
	}
}

func TestResultStatusError(t *testing.T) {
	s, rec := tracedService(t)
	s.traceToolResult(context.Background(), "call-err", "lookup", map[string]any{"error": "not found"})
	if got := toolResultSpan(t, rec)["tool.result_status"].AsString(); got != "error" {
		t.Errorf("tool.result_status = %q, want error", got)
	}
}

func TestResultStatusSuccess(t *testing.T) {
	s, rec := tracedService(t)
	s.traceToolResult(context.Background(), "call-ok", "lookup", map[string]any{"success": true, "value": 1})
	if got := toolResultSpan(t, rec)["tool.result_status"].AsString(); got != "success" {
		t.Errorf("tool.result_status = %q, want success", got)
	}
}

func TestLongResultIsTruncated(t *testing.T) {
	s, rec := tracedService(t)
	s.traceToolResult(context.Background(), "call-big", "dump", map[string]any{"blob": strings.Repeat("x", 5000)})
	result := toolResultSpan(t, rec)["tool.result"].AsString()
	if !strings.HasSuffix(result, "...") || len(result) > 2003 {
		t.Errorf("tool.result is %d long, want it cut to 2000 and marked", len(result))
	}
}

func TestNonDictResultOmitsResultAttribute(t *testing.T) {
	s, rec := tracedService(t)
	s.traceToolResult(context.Background(), "call-str", "echo", "just a string")
	attrs := toolResultSpan(t, rec)
	if attrs["tool.call_id"].AsString() != "call-str" || attrs["tool.function_name"].AsString() != "echo" {
		t.Errorf("attrs = %v, want the call and the function", attrs)
	}
	if _, ok := attrs["tool.result"]; ok {
		t.Error("a result that is not an object was recorded")
	}
	if _, ok := attrs["tool.result_status"]; ok {
		t.Error("a result that is not an object was given a status")
	}
}

// (Not a port of an upstream test.) The user's speech as the session
// transcribed it is traced as a transcription segment, closed with the user's
// turn.
func TestUserTranscriptIsTracedAsSTT(t *testing.T) {
	s, rec := tracedService(t)
	ctx := context.Background()
	s.pushUserTranscript(ctx, "hi there")
	if err := s.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}
	var got map[string]string
	for _, sp := range rec.Ended() {
		if sp.Name() != "stt" {
			continue
		}
		got = map[string]string{}
		for _, kv := range sp.Attributes() {
			got[string(kv.Key)] = kv.Value.String()
		}
	}
	if got["transcript"] != "hi there" || got["gen_ai.request.model"] != "gemini-live-test" ||
		got["stt.incomplete"] != "true" {
		t.Errorf("stt span = %v, want the user's transcript", got)
	}
}
