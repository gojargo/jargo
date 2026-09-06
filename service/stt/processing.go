package stt

import (
	"context"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
)

// processingMeter measures how long a service spent on one discrete piece of
// work: a segment of audio handed over whole and the transcript that came back.
//
// It answers a different question from the time to first byte, which is the wait
// the conversation felt. Only a service that transcribes in discrete units
// reports it: a streaming service is fed audio continuously and performs no such
// unit, so the number it would report is either meaningless or the wait under
// another name.
type processingMeter struct {
	svc   *processor.Base
	model func() string
}

// newProcessingMeter builds a meter reporting for svc, labeling what it reports
// with the model the service names at the time it reports.
func newProcessingMeter(svc *processor.Base, model func() string) *processingMeter {
	return &processingMeter{svc: svc, model: model}
}

// reportElapsed emits a measurement the caller timed itself, for a service that
// brackets its own work rather than a stretch of frames.
func (m *processingMeter) reportElapsed(ctx context.Context, elapsed time.Duration) {
	model := m.model()
	slog.Debug("stt processing time", "service", m.svc.Name(), "elapsed", elapsed)
	metrics.RecordProcessing(ctx, "stt", m.svc.Name(), model, elapsed.Seconds())
	if !m.svc.MetricsEnabled() {
		return
	}
	data := frames.ProcessingMetricsData{
		Processor: m.svc.Name(), Model: model,
		Value: elapsed,
	}
	_ = m.svc.PushFrame(ctx, frames.NewMetricsFrame(data), processor.Downstream)
}
