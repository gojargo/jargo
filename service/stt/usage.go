package stt

import (
	"context"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// pushUsageMetrics reports the audio a service was given, downstream as a
// MetricsFrame for an in-band consumer such as an RTVI client. It is raw usage,
// not cost, and each report covers the audio since the one before it, so a
// consumer sums them across a session.
//
// It is gated on usage metrics being enabled, the same as the token usage an LLM
// reports; the span and the OpenTelemetry counters are recorded either way.
func pushUsageMetrics(ctx context.Context, p *processor.Base, model string, audio time.Duration) {
	if !p.UsageMetricsEnabled() {
		return
	}
	f := frames.NewMetricsFrame(frames.STTUsageMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: p.Name(), Model: model},
		Value:           frames.STTUsage{AudioSeconds: audio.Seconds()},
	})
	_ = p.PushFrame(ctx, f, processor.Downstream)
}

// pushUsageMetrics reports the audio this service was given.
func (s *StreamService) pushUsageMetrics(ctx context.Context, audio time.Duration) {
	pushUsageMetrics(ctx, s.Base.Base, s.model, audio)
}

// pushUsageMetrics reports the audio this service was given.
func (s *SegmentService) pushUsageMetrics(ctx context.Context, audio time.Duration) {
	pushUsageMetrics(ctx, s.Base.Base, s.model, audio)
}
