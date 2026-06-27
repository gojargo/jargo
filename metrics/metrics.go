// Package metrics exports jargo's service measurements — time-to-first-byte,
// processing time, LLM token usage and TTS characters — as OpenTelemetry metrics
// over OTLP. It is the aggregate, fleet-level counterpart to the per-turn
// frames.MetricsFrame the services emit in-band (and the RTVI metrics messages
// the client sees).
//
// The service processors record through the global meter, so instrumentation
// costs nothing until a MeterProvider is installed. Call Init at startup to
// export over OTLP; an OpenTelemetry Collector forwards the metrics to
// Prometheus or any other backend.
package metrics

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
)

// meterName identifies jargo's instruments.
const meterName = "github.com/gojargo/jargo"

// Instruments are created lazily from the global meter, which delegates to the
// installed provider (or no-ops). once guards their creation.
//
//nolint:gochecknoglobals // process-wide instrument singletons
var (
	once       sync.Once
	ttfbHist   metric.Float64Histogram
	procHist   metric.Float64Histogram
	tokCounter metric.Int64Counter
	charCount  metric.Int64Counter
)

func instruments() {
	once.Do(func() {
		m := otel.Meter(meterName)
		ttfbHist, _ = m.Float64Histogram("jargo.ttfb",
			metric.WithUnit("s"), metric.WithDescription("service time to first byte"))
		procHist, _ = m.Float64Histogram("jargo.processing",
			metric.WithUnit("s"), metric.WithDescription("service processing time"))
		tokCounter, _ = m.Int64Counter("jargo.llm.tokens",
			metric.WithDescription("LLM tokens, by direction"))
		charCount, _ = m.Int64Counter("jargo.tts.characters",
			metric.WithDescription("characters synthesized by TTS"))
	})
}

// serviceAttrs builds the common attribute set, omitting an empty model.
func serviceAttrs(kind, service, model string, extra ...attribute.KeyValue) metric.MeasurementOption {
	attrs := make([]attribute.KeyValue, 0, 3+len(extra))
	if kind != "" {
		attrs = append(attrs, attribute.String("service.kind", kind))
	}
	attrs = append(attrs, attribute.String("service", service))
	if model != "" {
		attrs = append(attrs, attribute.String("model", model))
	}
	attrs = append(attrs, extra...)
	return metric.WithAttributes(attrs...)
}

// RecordTTFB records a service's time to first byte, in seconds.
func RecordTTFB(ctx context.Context, kind, service, model string, seconds float64) {
	instruments()
	ttfbHist.Record(ctx, seconds, serviceAttrs(kind, service, model))
}

// RecordProcessing records a service's processing time, in seconds.
func RecordProcessing(ctx context.Context, kind, service, model string, seconds float64) {
	instruments()
	procHist.Record(ctx, seconds, serviceAttrs(kind, service, model))
}

// RecordTokens records LLM token usage, split by direction (input/output).
func RecordTokens(ctx context.Context, service, model string, input, output int64) {
	instruments()
	tokCounter.Add(ctx, input, serviceAttrs("llm", service, model, attribute.String("direction", "input")))
	tokCounter.Add(ctx, output, serviceAttrs("llm", service, model, attribute.String("direction", "output")))
}

// RecordTTSCharacters records the number of characters a TTS service synthesized.
func RecordTTSCharacters(ctx context.Context, service string, n int64) {
	instruments()
	charCount.Add(ctx, n, serviceAttrs("tts", service, ""))
}

// Config configures OTLP metric export.
type Config struct {
	// ServiceName labels the metrics; defaults to "jargo".
	ServiceName string
	// ServiceVersion is an optional version label.
	ServiceVersion string
	// Endpoint overrides the OTLP HTTP endpoint (host:port). Empty honors the
	// standard OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
	Endpoint string
	// Insecure sends over plain HTTP instead of HTTPS.
	Insecure bool
}

// Init installs a global MeterProvider that periodically exports to an OTLP HTTP
// collector, and returns a shutdown function that flushes and stops it. Call the
// returned function on exit. The service processors begin recording as soon as
// the provider is installed.
func Init(ctx context.Context, cfg Config) (func(context.Context) error, error) {
	var opts []otlpmetrichttp.Option
	if cfg.Endpoint != "" {
		opts = append(opts, otlpmetrichttp.WithEndpoint(cfg.Endpoint))
	}
	if cfg.Insecure {
		opts = append(opts, otlpmetrichttp.WithInsecure())
	}
	exp, err := otlpmetrichttp.New(ctx, opts...)
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

	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(exp)),
		sdkmetric.WithResource(resource.NewSchemaless(attrs...)),
	)
	otel.SetMeterProvider(mp)
	return mp.Shutdown, nil
}
