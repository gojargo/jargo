package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// The service a metric is recorded under is the kind of processor, not the
// instance: the instance number depends on how many processors were built
// before this one, so it changes across restarts and pipeline edits, and every
// value of it would be a series of its own.
func TestMetricsNameTheServiceWithoutItsInstanceNumber(t *testing.T) {
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	prev := otel.GetMeterProvider()
	otel.SetMeterProvider(mp)
	defer otel.SetMeterProvider(prev)

	ctx := context.Background()
	for range 2 {
		p := processor.New("FakeLLM", nil)
		_ = p.PushTokenUsage(ctx, "model-a", frames.LLMTokenUsage{PromptTokens: 10, CompletionTokens: 2})
	}

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(ctx, &rm); err != nil {
		t.Fatalf("collect: %v", err)
	}
	services := map[string]bool{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			sum, ok := m.Data.(metricdata.Sum[int64])
			if m.Name != "jargo.llm.tokens" || !ok {
				continue
			}
			for _, dp := range sum.DataPoints {
				v, _ := dp.Attributes.Value(attribute.Key("service"))
				services[v.AsString()] = true
			}
		}
	}
	if len(services) != 1 || !services["FakeLLM"] {
		t.Fatalf("service labels = %v, want only %q", services, "FakeLLM")
	}
}
