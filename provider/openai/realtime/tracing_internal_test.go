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

// spanRecorder installs a tracer provider recording the spans that end.
func spanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return rec
}

// tracedSetup sets a processor up on a traced pipeline.
func tracedSetup() processor.Setup {
	return processor.Setup{Clock: clock.NewSystem(), TracingEnabled: true, Tracing: tracing.NewTracingContext()}
}

// tracedService is an offline service with tracing on, and the recorder its
// spans end in.
func tracedService(t *testing.T) (*Service, *tracetest.SpanRecorder) {
	t.Helper()
	rec := spanRecorder(t)
	s, _ := offlineServiceWith(t, Config{Model: "gpt-realtime"}, tracedSetup())
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

// responseDoneWithUsage is a response.done event billing text, audio and
// cached tokens.
func responseDoneWithUsage() map[string]any {
	return map[string]any{
		"type": "response.done", "event_id": "ev_1",
		"response": map[string]any{
			"id": "resp_1", "object": "realtime.response", "status": "completed", "output": []any{},
			"usage": map[string]any{
				"total_tokens": 100, "input_tokens": 60, "output_tokens": 40,
				"input_token_details": map[string]any{
					"cached_tokens": 30, "text_tokens": 20, "audio_tokens": 40,
					"cached_tokens_details": map[string]any{"text_tokens": 10, "audio_tokens": 20},
				},
				"output_token_details": map[string]any{"text_tokens": 15, "audio_tokens": 25},
			},
		},
	}
}

// midReply puts the service in the middle of a reply: a response it has
// started and not yet closed.
func midReply(t *testing.T, s *Service) {
	t.Helper()
	if err := s.PushFrame(context.Background(), frames.NewLLMFullResponseStartFrame(),
		processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}
}

// A response completing produces one llm_response span carrying the usage it
// was billed on, with its audio breakdown, and no stt, llm_request or
// per-handler span beside it.
func TestResponseDoneProducesOneSpanWithAudioBreakdown(t *testing.T) {
	s, rec := tracedService(t)
	midReply(t, s)

	drive(t, s, responseDoneWithUsage())

	spans := spanAttrs(rec, "llm_response")
	if len(spans) != 1 {
		t.Fatalf("llm_response spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.provider.name":                       "openai",
		"gen_ai.request.model":                       "gpt-realtime",
		"gen_ai.usage.input_tokens":                  "60",
		"gen_ai.usage.output_tokens":                 "40",
		"gen_ai.usage.cache_read.input_tokens":       "30",
		"gen_ai.usage.audio.input_tokens":            "40",
		"gen_ai.usage.audio.output_tokens":           "25",
		"gen_ai.usage.audio.cache_read.input_tokens": "20",
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
	for _, removed := range []string{"stt", "llm_request"} {
		if got := spanAttrs(rec, removed); len(got) != 0 {
			t.Errorf("%s spans = %v, want none", removed, got)
		}
	}
}

// The Azure service is this one under another name, and its spans report the
// provider that name stands for.
func TestAzureRealtimeReportsItsProviderName(t *testing.T) {
	rec := spanRecorder(t)
	s := NewWithConnector("AzureRealtime", bearerConnector{}, Config{Model: "gpt-realtime"})
	ctx := context.Background()
	if err := s.Setup(ctx, tracedSetup()); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = s.Cleanup(ctx) })
	midReply(t, s)

	drive(t, s, responseDoneWithUsage())

	spans := spanAttrs(rec, "llm_response")
	if len(spans) != 1 || spans[0]["gen_ai.provider.name"] != "azure.ai.openai" {
		t.Fatalf("llm_response spans = %v, want one from azure.ai.openai", spans)
	}
}

// (Not a port of an upstream test.) The setup span records the conversation's
// tools, and nothing of the session's properties.
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
	if spans[0]["gen_ai.provider.name"] != "openai" {
		t.Errorf("provider = %q, want openai", spans[0]["gen_ai.provider.name"])
	}
	for k := range spans[0] {
		if len(k) > 8 && k[:8] == "session." {
			t.Errorf("attr %q recorded, want no session properties", k)
		}
	}
}

// (Not a port of an upstream test.) The user's transcript is the input of the
// reply it is answered by, not a transcription span of its own.
func TestUserTranscriptIsTheReplyInput(t *testing.T) {
	s, rec := tracedService(t)
	ctx := context.Background()
	if err := s.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}
	midReply(t, s)
	drive(t, s, map[string]any{
		"type": "conversation.item.input_audio_transcription.completed", "event_id": "ev",
		"item_id": "item_1", "content_index": 0, "transcript": "hi there",
	})
	drive(t, s, responseDoneWithUsage())
	// The reply answers speech, so it is exported when the service is closed
	// rather than when its usage arrives.
	if err := s.PushFrame(ctx, frames.NewEndFrame(), processor.Downstream); err != nil {
		t.Fatalf("push: %v", err)
	}

	spans := spanAttrs(rec, "llm_response")
	if len(spans) != 1 || spans[0]["input"] != "hi there" || spans[0]["input_modality"] != "AUDIO" {
		t.Errorf("llm_response spans = %v, want the user's transcript as the input", spans)
	}
	if got := spanAttrs(rec, "stt"); len(got) != 0 {
		t.Errorf("stt spans = %v, want none", got)
	}
}
