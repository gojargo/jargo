// Package tracing wires OpenTelemetry tracing into a jargo voice agent.
//
// The service processors emit spans through the global tracer, so
// instrumentation costs nothing until a TracerProvider is installed — without
// one, Tracer returns a no-op. Call Init at startup to export spans over OTLP,
// and wrap each session in StartConversation so the per-turn LLM and TTS spans
// nest under a single trace:
//
//	shutdown, err := tracing.Init(ctx, tracing.Config{ServiceName: "voicebot"})
//	defer shutdown(context.Background())
//	...
//	ctx, span := tracing.StartConversation(ctx, sessionID)
//	defer span.End()
//	task.Run(ctx) // LLM/TTS spans nest under the conversation span
package tracing

import (
	"context"

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

// StartConversation begins the root span for one session. The returned context
// carries the span, so service spans created while processing the session's
// frames nest under it; pass it to pipeline.Task.Run. Call End on the returned
// span when the session ends.
func StartConversation(ctx context.Context, id string) (context.Context, trace.Span) {
	ctx, span := Tracer().Start(ctx, "conversation")
	if id != "" {
		span.SetAttributes(attribute.String("conversation.id", id))
	}
	return ctx, span
}
