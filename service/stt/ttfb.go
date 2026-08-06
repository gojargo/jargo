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

// DefaultTTFBTimeout is how long a service waits after the speech ends for the
// transcript that closes it, before reporting the latency against whatever
// arrived in the meantime.
const DefaultTTFBTimeout = 2 * time.Second

// ttfbTracker measures the wait between the user finishing an utterance and its
// transcript being available.
//
// It is not the time-to-first-byte of a request and its answer. A transcription
// service is fed continuously and there is no request to measure from, so what
// is measured is the interval that matters to a conversation: from the moment
// the speech ended to the moment the words could be acted on.
//
// The moment the speech ended is the VAD's determination less the silence it
// required before making it, not the moment it decided. The transcript that
// closes the utterance ends the measurement. When none arrives the deadline
// reports the wait to the last transcript that did, and reports nothing at all
// when there was none: a service that did not answer has no latency to give.
type ttfbTracker struct {
	svc   *processor.Base
	model func() string

	mu      sync.Mutex
	timeout time.Duration
	// start is when the speech ended, zero when nothing is being measured.
	start time.Time
	// lastTranscript is when the last transcript arrived, zero when none has
	// since the measurement began.
	lastTranscript time.Time
	// cancel ends the deadline waiting on the closing transcript, nil when none
	// is running.
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// newTTFBTracker builds a tracker reporting for svc, labeling what it reports
// with the model the service names at the time it reports.
func newTTFBTracker(svc *processor.Base, model func() string) *ttfbTracker {
	return &ttfbTracker{svc: svc, model: model, timeout: DefaultTTFBTimeout}
}

// setTimeout replaces how long the deadline waits. Zero restores the default.
func (t *ttfbTracker) setTimeout(d time.Duration) {
	if d <= 0 {
		d = DefaultTTFBTimeout
	}
	t.mu.Lock()
	t.timeout = d
	t.mu.Unlock()
}

// speechStarted drops what was being measured. The utterance it belonged to is
// over, and a transcript arriving now belongs to the new one.
func (t *ttfbTracker) speechStarted() {
	t.stopDeadline()
	t.mu.Lock()
	t.lastTranscript = time.Time{}
	t.mu.Unlock()
}

// interrupted drops the deadline but leaves what has arrived, since an
// interruption can land while the user is still speaking.
func (t *ttfbTracker) interrupted() {
	t.stopDeadline()
}

// speechEnded starts measuring from the moment the speech itself ended. A VAD
// that does not say how long it waited, or when it decided, is not measured
// against: without both there is no telling the wait for the transcript from the
// wait for the silence that confirmed the speech was over.
func (t *ttfbTracker) speechEnded(ctx context.Context, f *frames.VADUserStoppedSpeakingFrame) {
	if f.StopSecs == 0 || f.Timestamp.IsZero() {
		return
	}
	silence := time.Duration(f.StopSecs * float64(time.Second))
	t.mu.Lock()
	t.start = f.Timestamp.Add(-silence)
	t.mu.Unlock()
	t.startDeadline(ctx)
}

// transcript records that a transcript arrived and, when it is the one that
// closes the utterance, reports the wait it ended.
func (t *ttfbTracker) transcript(ctx context.Context, finalized bool) {
	now := time.Now()
	t.mu.Lock()
	t.lastTranscript = now
	t.mu.Unlock()
	if !finalized {
		return
	}
	// The transcript the wait was for has arrived, so there is nothing left to
	// wait out.
	t.stopDeadline()
	t.report(ctx, now)
}

// answered reports the wait now rather than leaving it to the deadline, for a
// provider that has confirmed it has nothing further to send. The transcript
// went out ahead of the confirmation, so the wait is already over and there is
// nothing to be gained by waiting the deadline out.
func (t *ttfbTracker) answered(ctx context.Context) {
	t.stopDeadline()
	t.report(ctx, time.Now())
}

// close ends the deadline for good, for a service being torn down.
func (t *ttfbTracker) close() { t.stopDeadline() }

// report emits the wait from the end of speech to end, and clears the
// measurement so only the first transcript to close an utterance counts.
func (t *ttfbTracker) report(ctx context.Context, end time.Time) {
	t.mu.Lock()
	start := t.start
	t.start = time.Time{}
	t.mu.Unlock()
	if start.IsZero() {
		// Nothing was being measured: the transcript arrived without the VAD
		// having reported the speech ending, so there is no end of speech to
		// measure it against.
		return
	}
	ttfb := end.Sub(start)

	model := t.model()
	slog.Debug("stt ttfb", "service", t.svc.Name(), "ttfb", ttfb)
	metrics.RecordTTFB(ctx, "stt", t.svc.Name(), model, ttfb.Seconds())
	if !t.svc.MetricsEnabled() {
		return
	}
	data := frames.TTFBMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: t.svc.Name(), Model: model},
		Value:           ttfb,
	}
	_ = t.svc.PushFrame(ctx, frames.NewMetricsFrame(data), processor.Downstream)
}

// startDeadline replaces the running deadline with one for the utterance that
// has just ended.
func (t *ttfbTracker) startDeadline(ctx context.Context) {
	t.stopDeadline()
	dctx, cancel := context.WithCancel(ctx)
	t.mu.Lock()
	t.cancel = cancel
	timeout := t.timeout
	t.mu.Unlock()
	t.wg.Go(func() { t.waitForTranscript(dctx, timeout) })
}

// stopDeadline ends the running deadline and waits for it to finish.
func (t *ttfbTracker) stopDeadline() {
	t.mu.Lock()
	cancel := t.cancel
	t.cancel = nil
	t.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	t.wg.Wait()
}

// waitForTranscript reports the wait once the deadline passes, measured to the
// last transcript that arrived. Nothing is reported when none did: a latency
// measured to a transcript that never came would be reporting the deadline
// rather than the service.
func (t *ttfbTracker) waitForTranscript(ctx context.Context, timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return
	case <-timer.C:
	}
	t.mu.Lock()
	last := t.lastTranscript
	t.mu.Unlock()
	if last.IsZero() {
		return
	}
	t.report(ctx, last)
}
