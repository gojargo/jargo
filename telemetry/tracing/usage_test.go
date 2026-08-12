package tracing_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// TestSetTokenUsageWritesAttributes checks that token usage lands on the span
// under the standard gen_ai.usage.* keys, cache and per-modality breakdowns
// included.
func TestSetTokenUsageWritesAttributes(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	ctx, span := tracing.Tracer().Start(context.Background(), "llm")
	tracing.SetTokenUsage(ctx, frames.LLMTokenUsage{
		PromptTokens:         150,
		CompletionTokens:     50,
		TotalTokens:          200,
		CacheReadTokens:      new(int64(20)),
		CacheCreationTokens:  new(int64(5)),
		ReasoningTokens:      new(int64(12)),
		InputAudioTokens:     new(int64(50)),
		OutputAudioTokens:    new(int64(40)),
		CacheReadAudioTokens: new(int64(8)),
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
		"gen_ai.usage.input_tokens":                  150,
		"gen_ai.usage.output_tokens":                 50,
		"gen_ai.usage.cache_read.input_tokens":       20,
		"gen_ai.usage.cache_creation.input_tokens":   5,
		"gen_ai.usage.reasoning.output_tokens":       12,
		"gen_ai.usage.audio.input_tokens":            50,
		"gen_ai.usage.audio.output_tokens":           40,
		"gen_ai.usage.audio.cache_read.input_tokens": 8,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Fatalf("attr %q = %d, want %d (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetTokenUsageRecordsReportedZero checks that a count the service measured
// as zero lands on the span. On a model that caches, a generation that read
// nothing from the cache is a measurement, and it has to be distinguishable
// from a model that does not cache at all.
func TestSetTokenUsageRecordsReportedZero(t *testing.T) {
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
		CacheReadTokens:  new(int64(0)),
	})
	span.End()

	var found bool
	for _, kv := range rec.Ended()[0].Attributes() {
		if string(kv.Key) == "gen_ai.usage.cache_read.input_tokens" {
			found = true
			if got := kv.Value.AsInt64(); got != 0 {
				t.Fatalf("cache-read tokens = %d, want the reported 0", got)
			}
		}
	}
	if !found {
		t.Fatal("a cache read the service reported as zero should still be recorded")
	}
}

// TestSetTokenUsageOmitsUnreportedBreakdowns confirms a generation whose service
// accounts for none of the cache or per-modality counts carries no attribute for
// them, rather than a run of zeroes.
func TestSetTokenUsageOmitsUnreportedBreakdowns(t *testing.T) {
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
		case "gen_ai.usage.cache_read.input_tokens",
			"gen_ai.usage.cache_creation.input_tokens",
			"gen_ai.usage.reasoning.output_tokens",
			"gen_ai.usage.audio.input_tokens",
			"gen_ai.usage.audio.output_tokens",
			"gen_ai.usage.audio.cache_read.input_tokens":
			t.Fatalf("unexpected breakdown attribute %q for a service that reported none", kv.Key)
		}
	}
}

// spanOf is the span carried by a context handed to a stringAttrs callback.
func spanOf(ctx context.Context) trace.Span { return trace.SpanFromContext(ctx) }

// stringAttrs runs fn against a live span and returns the attributes it wrote.
func stringAttrs(t *testing.T, fn func(ctx context.Context)) map[string]string {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	ctx, span := tracing.Tracer().Start(context.Background(), "test")
	fn(ctx)
	span.End()

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	attrs := map[string]string{}
	for _, kv := range spans[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.String()
	}
	return attrs
}

// TestSetTTSUsage checks that a synthesis is marked as a priceable generation
// carrying its character count.
func TestSetTTSUsage(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetTTSUsage(ctx, "eleven_flash_v2_5", 42)
	})

	want := map[string]string{
		"gen_ai.operation.name":              "tts",
		"gen_ai.request.model":               "eleven_flash_v2_5",
		"langfuse.observation.type":          "generation",
		"langfuse.observation.usage_details": `{"characters":42}`,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetSTTUsage checks that a transcription reports its audio in whole
// milliseconds.
func TestSetSTTUsage(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetSTTUsage(ctx, "nova-3", 2500*time.Millisecond)
	})

	want := map[string]string{
		"gen_ai.operation.name":              "stt",
		"gen_ai.request.model":               "nova-3",
		"langfuse.observation.type":          "generation",
		"langfuse.observation.usage_details": `{"milliseconds":2500}`,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetUsageWithoutModel confirms a service that reports no model leaves the
// span a plain span: there is nothing to price the usage against.
func TestSetUsageWithoutModel(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetTTSUsage(ctx, "", 42)
	})

	if got := attrs["gen_ai.operation.name"]; got != "tts" {
		t.Errorf("gen_ai.operation.name = %q, want tts", got)
	}
	for _, k := range []string{
		"gen_ai.request.model",
		"langfuse.observation.type",
		"langfuse.observation.usage_details",
	} {
		if v, ok := attrs[k]; ok {
			t.Errorf("attr %q = %q, want unset", k, v)
		}
	}
}
