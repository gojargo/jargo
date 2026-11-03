package realtime

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// (Not a port of an upstream test: upstream has no tests of these spans.)

// tracedService is an offline service with tracing on, and the recorder its
// spans end in.
func tracedService(t *testing.T) (*Service, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	s, _ := offlineServiceWith(t, Config{}, processor.Setup{
		Clock: clock.NewSystem(), TracingEnabled: true, Tracing: tracing.NewTracingContext(),
	})
	s.Link(newRecorder())
	src := newRecorder()
	src.Link(s)
	return s, rec
}

// spanAttrs are the attributes of the ended spans named name.
func spanAttrs(rec *tracetest.SpanRecorder, name string) []map[string]string {
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

// Asking for a response records the conversation as it stands.
func TestRequestSpanRecordsTheConversation(t *testing.T) {
	s, rec := tracedService(t)
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	s.mu.Lock()
	s.convo, s.sessionReady = convo, true
	s.mu.Unlock()
	s.createResponse(context.Background())
	spans := spanAttrs(rec, "llm_request")
	if len(spans) != 1 {
		t.Fatalf("request spans = %v, want one", spans)
	}
	if got := spans[0]["input"]; got != `[{"content":"Hello","role":"user"}]` {
		t.Errorf("input = %q, want the conversation", got)
	}
	if spans[0]["gen_ai.operation.name"] != "llm_request" {
		t.Errorf("attrs = %v, want the operation named", spans[0])
	}
}

// The setup span records the conversation's tools, and nothing of the
// session's properties.
func TestSetupSpanRecordsTheConversationTools(t *testing.T) {
	s, rec := tracedService(t)
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{Name: "get_weather", Description: "Weather."}})
	s.mu.Lock()
	s.convo = convo
	s.mu.Unlock()
	s.traceSetup(context.Background())
	spans := spanAttrs(rec, "llm_setup")
	if len(spans) != 1 || spans[0]["tools.names"] != "get_weather" || spans[0]["tools.count"] != "1" {
		t.Fatalf("setup spans = %v, want the conversation's tool", spans)
	}
	for k := range spans[0] {
		if len(k) > 8 && k[:8] == "session." {
			t.Errorf("attr %q recorded, want no session properties", k)
		}
	}
}

// A completed response records the time to first byte measured for it.
func TestResponseSpanRecordsTTFB(t *testing.T) {
	s, rec := tracedService(t)
	s.StartTTFBMetrics()
	s.ReportTTFBMetrics(context.Background())
	drive(t, s, responseDone("resp"))
	spans := spanAttrs(rec, "llm_response")
	if len(spans) != 1 || spans[0]["metrics.ttfb"] == "" {
		t.Errorf("response spans = %v, want the time to first byte", spans)
	}
}

// The user's speech as the session transcribed it is traced as a transcription
// segment, closed with the user's turn.
func TestUserTranscriptIsTracedAsSTT(t *testing.T) {
	s, rec := tracedService(t)
	drive(t, s, map[string]any{
		"type": "conversation.item.input_audio_transcription.completed", "event_id": "ev",
		"item_id": "item_1", "content_index": 0, "transcript": "hi there",
	})
	if err := s.ProcessFrame(context.Background(), frames.NewUserStoppedSpeakingFrame(),
		processor.Downstream); err != nil {
		t.Fatalf("process: %v", err)
	}
	spans := spanAttrs(rec, "stt")
	if len(spans) != 1 || spans[0]["transcript"] != "hi there" || spans[0]["stt.incomplete"] != "true" {
		t.Errorf("stt spans = %v, want the user's transcript", spans)
	}
}
