// Package tracing wires OpenTelemetry tracing into a jargo voice agent.
//
// The service processors emit spans through the global tracer, so
// instrumentation costs nothing until a TracerProvider is installed — without
// one, Tracer returns a no-op. Call Init at startup to export spans over OTLP,
// and let the task trace the session so the conversation, its turns and the STT,
// LLM and TTS calls of each turn nest under a single trace:
//
//	shutdown, err := tracing.Init(ctx, tracing.Config{ServiceName: "voicebot"})
//	defer shutdown(context.Background())
//	...
//	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
//		EnableTracing:  true,
//		ConversationID: sessionID,
//	})
//	task.Run(ctx)
package tracing

import (
	"context"
	"strconv"
	"time"

	"github.com/gojargo/jargo/frames"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// instrumentationName identifies jargo's spans.
const instrumentationName = "github.com/gojargo/jargo"

// Tracer returns jargo's tracer from the global TracerProvider. Before Init (or
// any other provider) is installed, this is a no-op tracer.
func Tracer() trace.Tracer {
	return otel.Tracer(instrumentationName)
}

// Config configures OTLP export.
type Config struct {
	// ServiceName labels the traces; defaults to "jargo".
	ServiceName string
	// ServiceVersion is an optional version label.
	ServiceVersion string
	// Endpoint overrides the OTLP HTTP endpoint (host:port). Empty honors the
	// standard OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
	Endpoint string
	// Insecure sends over plain HTTP instead of HTTPS.
	Insecure bool
	// SampleRatio is the head-sampling ratio in (0,1]. Zero (or less) always
	// samples — jargo traces are low-volume, one trace per session.
	SampleRatio float64
}

// Init installs a global TracerProvider that batches spans to an OTLP HTTP
// collector, and returns a shutdown function that flushes and stops it. Call the
// returned function on exit. The service processors begin emitting spans as soon
// as the provider is installed.
func Init(ctx context.Context, cfg Config) (func(context.Context) error, error) {
	var opts []otlptracehttp.Option
	if cfg.Endpoint != "" {
		opts = append(opts, otlptracehttp.WithEndpoint(cfg.Endpoint))
	}
	if cfg.Insecure {
		opts = append(opts, otlptracehttp.WithInsecure())
	}
	exp, err := otlptracehttp.New(ctx, opts...)
	if err != nil {
		return nil, err
	}

	name := cfg.ServiceName
	if name == "" {
		name = "jargo"
	}
	attrs := []attribute.KeyValue{attribute.String("service.name", name)}
	if cfg.ServiceVersion != "" {
		attrs = append(attrs, attribute.String("service.version", cfg.ServiceVersion))
	}

	sampler := sdktrace.AlwaysSample()
	if cfg.SampleRatio > 0 {
		sampler = sdktrace.TraceIDRatioBased(cfg.SampleRatio)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(resource.NewSchemaless(attrs...)),
		sdktrace.WithSampler(sampler),
	)
	otel.SetTracerProvider(tp)
	return tp.Shutdown, nil
}

// SetTokenUsage records LLM token usage on the span in ctx, under the
// OpenTelemetry GenAI gen_ai.usage.* keys.
//
// The input and output counts are always recorded. The cache and per-modality
// breakdowns are recorded whenever the service accounted for them, a reported
// zero included: on a model that caches, no cache read this generation is a
// measurement, and one worth telling apart from a model that does not cache at
// all. A count the service left unreported carries no attribute.
func SetTokenUsage(ctx context.Context, u frames.LLMTokenUsage) {
	span := trace.SpanFromContext(ctx)
	span.SetAttributes(
		attribute.Int64("gen_ai.usage.input_tokens", u.PromptTokens),
		attribute.Int64("gen_ai.usage.output_tokens", u.CompletionTokens),
	)
	setReported(span, "gen_ai.usage.cache_read.input_tokens", u.CacheReadTokens)
	setReported(span, "gen_ai.usage.cache_creation.input_tokens", u.CacheCreationTokens)
	setReported(span, "gen_ai.usage.reasoning.output_tokens", u.ReasoningTokens)
	setReported(span, "gen_ai.usage.audio.input_tokens", u.InputAudioTokens)
	setReported(span, "gen_ai.usage.audio.output_tokens", u.OutputAudioTokens)
	setReported(span, "gen_ai.usage.audio.cache_read.input_tokens", u.CacheReadAudioTokens)
}

// setReported sets an int64 span attribute for a count the service reported,
// zero included, and leaves the attribute off one it did not report.
func setReported(span trace.Span, key string, v *int64) {
	if v != nil {
		span.SetAttributes(attribute.Int64(key, *v))
	}
}

// Attribute keys that let a cost-tracking backend price a span. The
// OpenTelemetry GenAI conventions model usage only as token counts, so speech
// billed per character or per second of audio has no standard key to report
// under; these carry the billable units and mark the span as the kind of
// observation a price applies to.
const (
	observationTypeKey = "langfuse.observation.type"
	usageDetailsKey    = "langfuse.observation.usage_details"
	generationType     = "generation"
)

// SetTTSUsage records one synthesis's billable usage on the span in ctx: the
// provider model and the number of characters handed to it. Characters are
// counted in runes rather than bytes, because that is the unit TTS providers
// bill in — an accented character is one character, not the two bytes it
// occupies in UTF-8.
func SetTTSUsage(ctx context.Context, model string, characters int) {
	setUsage(ctx, "tts", model, usageDetails("characters", int64(characters)))
}

// SetSTTUsage records billable transcription usage on the span in ctx: the
// provider model and the duration of audio sent for transcription. The unit is
// milliseconds because usage values have to be whole numbers — a fractional one
// is discarded — and rounding to whole seconds would throw away most of a short
// turn. Configure the model's price per millisecond accordingly: a rate quoted
// per minute is that rate divided by 60000.
func SetSTTUsage(ctx context.Context, model string, audio time.Duration) {
	setUsage(ctx, "stt", model, usageDetails("milliseconds", audio.Milliseconds()))
}

// setUsage marks the span in ctx as a priceable generation carrying usage. The
// operation is always recorded; the model and the usage are recorded only when
// the service reported a model, since without one there is nothing to price the
// usage against and the span is better left a plain span than an unpriceable
// generation.
func setUsage(ctx context.Context, operation, model, usage string) {
	span := trace.SpanFromContext(ctx)
	span.SetAttributes(attribute.String("gen_ai.operation.name", operation))
	if model == "" {
		return
	}
	span.SetAttributes(
		attribute.String("gen_ai.request.model", model),
		attribute.String(observationTypeKey, generationType),
		attribute.String(usageDetailsKey, usage),
	)
}

// usageDetails renders a single-unit usage object. Both the unit and the count
// are constrained to values that need no escaping, so the JSON is built
// directly rather than marshaled.
func usageDetails(unit string, n int64) string {
	return `{"` + unit + `":` + strconv.FormatInt(n, 10) + `}`
}
