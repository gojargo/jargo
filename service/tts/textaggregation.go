package tts

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// textAggregationMetrics times how long grouping text into sentences takes: from
// a model's first token to the sentence it completes, which is the delay before
// synthesis of that sentence can start at all.
type textAggregationMetrics struct {
	mu      sync.Mutex
	start   time.Time
	started bool
}

// isStreamingTokens reports whether the service passes text straight through as
// it arrives rather than grouping it into sentences. Per-token aggregation time
// is not a meaningful measurement, so it is not taken.
func (b *Base) isStreamingTokens() bool {
	return b.aggregator != nil && b.aggregator.Type() == frames.AggregationToken
}

// startTextAggregationMetrics starts the clock, unless it is already running.
// Text keeps arriving while a sentence is being waited for, and the measurement
// runs from the first of it, not the most recent. The clock is stopped by the
// sentence it was waiting for, so the text after that starts it again: what is
// measured is the wait before each sentence.
func (b *Base) startTextAggregationMetrics() {
	if !b.CanGenerateMetrics() || !b.MetricsEnabled() {
		return
	}
	if b.isStreamingTokens() {
		return
	}
	b.textAgg.mu.Lock()
	defer b.textAgg.mu.Unlock()
	if b.textAgg.started {
		return
	}
	b.textAgg.started = true
	b.textAgg.start = time.Now()
}

// stopTextAggregationMetrics stops the clock and reports what it measured. It
// does nothing when the clock was not running, so it is safe to call again at
// the end of a response after the first sentence already stopped it.
func (b *Base) stopTextAggregationMetrics(ctx context.Context) {
	b.textAgg.mu.Lock()
	started, start := b.textAgg.started, b.textAgg.start
	b.textAgg.started = false
	b.textAgg.start = time.Time{}
	b.textAgg.mu.Unlock()

	if !started {
		return
	}
	if !b.CanGenerateMetrics() || !b.MetricsEnabled() {
		return
	}

	value := time.Since(start)
	slog.Debug("text aggregation time", "service", b.Name(), "took", value)
	f := frames.NewMetricsFrame(frames.TextAggregationMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: b.Name(), Model: b.meta.Model},
		Value:           value,
	})
	_ = b.PushFrame(ctx, f, processor.Downstream)
}
