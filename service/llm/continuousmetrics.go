package llm

import (
	"context"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// A service generating continuously has no generation for the base to time, so
// it times its own responses with the methods below and reports each
// measurement as it completes: the time to first byte when the first output
// arrives, the processing time when the response ends.

// ReportTTFBMetrics stops the time-to-first-byte clock started with
// StartTTFBMetrics and reports the measurement at once. It is for a service
// generating continuously, which calls it on the first output of a response.
// Only the first call after a start reports.
func (b *Base) ReportTTFBMetrics(ctx context.Context) {
	b.ttfbMu.Lock()
	if b.hasTTFB || b.ttfbStart.IsZero() {
		b.ttfbMu.Unlock()
		return
	}
	ttfb := time.Since(b.ttfbStart)
	b.ttfb, b.hasTTFB = ttfb, true
	b.ttfbMu.Unlock()

	model := b.modelName()
	trace.SpanFromContext(ctx).SetAttributes(attribute.Float64("metrics.ttfb", ttfb.Seconds()))
	metrics.RecordTTFB(ctx, "llm", b.TypeName(), model, ttfb.Seconds())
	if !b.MetricsEnabled() {
		return
	}
	var data frames.TTFBMetricsData
	data.Processor, data.Model, data.Value = b.Name(), model, ttfb
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(data), processor.Downstream)
}

// StartProcessingMetrics starts the processing clock of a service generating
// continuously, which calls it when it starts on a response.
func (b *Base) StartProcessingMetrics() {
	b.ttfbMu.Lock()
	defer b.ttfbMu.Unlock()
	b.processingStart = time.Now()
}

// StopProcessingMetrics stops the processing clock and reports the measurement.
// It does nothing when the clock is not running.
func (b *Base) StopProcessingMetrics(ctx context.Context) {
	b.ttfbMu.Lock()
	start := b.processingStart
	b.processingStart = time.Time{}
	b.ttfbMu.Unlock()
	if start.IsZero() {
		return
	}
	processing := time.Since(start)
	model := b.modelName()
	metrics.RecordProcessing(ctx, "llm", b.TypeName(), model, processing.Seconds())
	if !b.MetricsEnabled() {
		return
	}
	var data frames.ProcessingMetricsData
	data.Processor, data.Model, data.Value = b.Name(), model, processing
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(data), processor.Downstream)
}

// StopAllMetrics stops and reports whatever is being timed: the time to first
// byte, if no output has arrived yet, and the processing time.
func (b *Base) StopAllMetrics(ctx context.Context) {
	b.ReportTTFBMetrics(ctx)
	b.StopProcessingMetrics(ctx)
}
