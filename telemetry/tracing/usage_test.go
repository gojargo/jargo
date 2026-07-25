package tracing_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// TestSetTokenUsageWritesAttributes checks that token usage lands on the span
// under both the legacy llm.tokens.* and the standard gen_ai.usage.* keys, and
// that the audio/text/cache breakdowns are written when nonzero.
func TestSetTokenUsageWritesAttributes(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	ctx, span := tracing.Tracer().Start(context.Background(), "llm")
	tracing.SetTokenUsage(ctx, frames.LLMTokenUsage{
		PromptTokens:      150,
		CompletionTokens:  50,
		TotalTokens:       200,
		CacheReadTokens:   20,
		InputAudioTokens:  50,
		OutputAudioTokens: 40,
		InputTextTokens:   100,
		OutputTextTokens:  10,
	})
	span.End()

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	attrs := map[string]int64{}
	for _, kv := range spans[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInt64()
	}

	want := map[string]int64{
		"llm.tokens.input":                     150,
		"llm.tokens.output":                    50,
		"llm.tokens.total":                     200,
		"gen_ai.usage.input_tokens":            150,
		"gen_ai.usage.output_tokens":           50,
		"gen_ai.usage.input_audio_tokens":      50,
		"gen_ai.usage.output_audio_tokens":     40,
		"gen_ai.usage.input_text_tokens":       100,
		"gen_ai.usage.output_text_tokens":      10,
		"gen_ai.usage.cache_read.input_tokens": 20,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Fatalf("attr %q = %d, want %d (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetTokenUsageOmitsZeroBreakdowns confirms a text-only generation carries
// no empty realtime/cache attributes.
func TestSetTokenUsageOmitsZeroBreakdowns(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	ctx, span := tracing.Tracer().Start(context.Background(), "llm")
	tracing.SetTokenUsage(ctx, frames.LLMTokenUsage{
		PromptTokens:     12,
		CompletionTokens: 3,
		TotalTokens:      15,
	})
	span.End()

	for _, kv := range rec.Ended()[0].Attributes() {
		switch string(kv.Key) {
		case "gen_ai.usage.input_audio_tokens",
			"gen_ai.usage.output_audio_tokens",
			"gen_ai.usage.input_text_tokens",
			"gen_ai.usage.output_text_tokens",
			"gen_ai.usage.cache_read.input_tokens",
			"gen_ai.usage.cache_creation.input_tokens":
			t.Fatalf("unexpected breakdown attribute %q on a text-only generation", kv.Key)
		}
	}
}
