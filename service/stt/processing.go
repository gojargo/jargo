package stt

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
)

// processingMeter measures how long a service spent on one utterance, from the
// point it began work to the transcript it produced.
//
// It answers a different question from the time to first byte. That is the wait
// the conversation felt, anchored to the moment the speech ended; this is the
// work the service did, anchored to the moment it started doing it. A streaming
// service is at it for the length of the utterance, so its processing time runs
// long where its wait is short.
type processingMeter struct {
	svc   *processor.Base
	model func() string

	mu sync.Mutex
	// start is when the work began, zero when nothing is being measured.
	start time.Time
}

// newProcessingMeter builds a meter reporting for svc, labeling what it reports
// with the model the service names at the time it reports.
func newProcessingMeter(svc *processor.Base, model func() string) *processingMeter {
	return &processingMeter{svc: svc, model: model}
}

// begin starts the clock, replacing whatever it was on. An utterance nothing was
// produced for is dropped rather than folded into the next one.
func (m *processingMeter) begin() {
	m.mu.Lock()
	m.start = time.Now()
	m.mu.Unlock()
}

// report emits the time spent since the clock started and stops it, so only the
// first transcript out of an utterance is measured against it. It reports
// nothing when the clock was never started.
func (m *processingMeter) report(ctx context.Context) {
	m.mu.Lock()
	start := m.start
	m.start = time.Time{}
	m.mu.Unlock()
	if start.IsZero() {
		return
	}
	m.reportElapsed(ctx, time.Since(start))
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
		BaseMetricsData: frames.BaseMetricsData{Processor: m.svc.Name(), Model: model},
		Value:           elapsed,
	}
	_ = m.svc.PushFrame(ctx, frames.NewMetricsFrame(data), processor.Downstream)
}
