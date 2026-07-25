package metrics_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/telemetry/metrics"
	"go.opentelemetry.io/otel"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

func TestRecordsInstruments(t *testing.T) {
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	prev := otel.GetMeterProvider()
	otel.SetMeterProvider(mp)
	defer otel.SetMeterProvider(prev)

	ctx := context.Background()
	metrics.RecordTTFB(ctx, "llm", "AnthropicLLM", "haiku", 0.3)
	metrics.RecordProcessing(ctx, "llm", "AnthropicLLM", "haiku", 1.2)
	metrics.RecordTokens(ctx, "AnthropicLLM", "haiku", 100, 40)
	metrics.RecordTTSCharacters(ctx, "ElevenLabsTTS", "eleven_flash_v2_5", 25)
	metrics.RecordSTTAudio(ctx, "DeepgramSTT", "nova-3", 4.5)

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(ctx, &rm); err != nil {
		t.Fatalf("collect: %v", err)
	}
	names := map[string]bool{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			names[m.Name] = true
		}
	}
	for _, want := range []string{
		"jargo.ttfb", "jargo.processing", "jargo.llm.tokens",
		"jargo.tts.characters", "jargo.stt.audio",
	} {
		if !names[want] {
			t.Fatalf("missing instrument %q; got %v", want, names)
		}
	}
}
