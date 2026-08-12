package stt

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// segmentTracer owns the span covering one segment of transcription: from the
// moment the user began speaking to the transcript that finalizes what they
// said. A turn that produces several finalized transcripts produces several
// spans, each anchored where its own speech began, so a trace shows what was
// heard and when rather than one span covering the whole session.
//
// The span opens on the first transcript rather than on the speech itself. The
// turn it belongs under is opened by an observer reacting to the same speech,
// and opening here would race it and parent the segment to the turn before. By
// the time a service has any words to report, the turn is open.
//
// The zero value is not usable; build one with newSegmentTracer. Safe for
// concurrent use: transcripts arrive on a provider's receive goroutine while the
// frame path is closing the segment they belong to.
type segmentTracer struct {
	svc *processor.Base
	// baseline supplies the attributes every span of this service opens with,
	// read when the span opens because the model and settings can change while
	// the pipeline runs.
	baseline func() tracing.STTAttributes

	mu      sync.Mutex
	span    trace.Span
	spanCtx context.Context //nolint:containedctx // the span outlives the call that opened it
	// anchor is when the speech this segment covers began, zero when the VAD has
	// not reported speech since the last segment closed.
	anchor time.Time
	// segments are the pieces of transcript recorded so far. A service that
	// re-sends a growing transcript replaces the last piece; one that sends each
	// piece once appends.
	segments []string
	// usage accumulates what the service reported for this segment, attached
	// when the span closes. It accumulates rather than being written as it
	// arrives because a service reports it just before the transcript that ends
	// the segment, when there may be no span open yet.
	usage *frames.STTUsage
}

// newSegmentTracer builds a tracer for svc, taking each span's opening
// attributes from baseline.
func newSegmentTracer(svc *processor.Base, baseline func() tracing.STTAttributes) *segmentTracer {
	return &segmentTracer{svc: svc, baseline: baseline}
}

// sttIncomplete marks a segment whose transcript never arrived.
//
//nolint:gochecknoglobals // a constant attribute
var sttIncomplete = attribute.Bool("stt.incomplete", true)

// audioSecondsAttr reports the audio a segment was transcribed from.
func audioSecondsAttr(seconds float64) attribute.KeyValue {
	return attribute.Float64("metrics.audio_seconds", seconds)
}

// speechStarted anchors the next span at the moment the speech began, which is
// earlier than the VAD's determination by the delay it took to confirm it.
//
// An anchor already set is kept, and so is one belonging to a span already open:
// a VAD that re-triggers inside a turn is reporting more of the same speech, not
// the start of a new segment.
func (t *segmentTracer) speechStarted(at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.span == nil && t.anchor.IsZero() {
		t.anchor = at
	}
}

// open starts the segment's span if it is not already open. It is called before
// a transcript is pushed, so that the metrics frame the push raises finds the
// span it belongs to already open.
func (t *segmentTracer) open() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.openLocked()
}

// openLocked opens the span. The caller holds t.mu.
func (t *segmentTracer) openLocked() {
	if t.span != nil {
		return
	}
	var opts []trace.SpanStartOption
	if !t.anchor.IsZero() {
		opts = append(opts, trace.WithTimestamp(t.anchor))
	}
	t.spanCtx, t.span = t.svc.StartSpan(context.Background(), "stt", opts...)
	tracing.SetSTTAttributes(t.span, t.baseline())
}

// record attaches a transcript to the open span, and closes the span when the
// transcript is the one that finalizes the segment. A transcript arriving with
// no span open is one the service produced without speech having been reported,
// and is left untraced rather than opening a span with nothing to anchor it.
func (t *segmentTracer) record(tf *frames.TranscriptionFrame) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.span == nil {
		return
	}
	if tf.Text != "" {
		t.appendLocked(tf.Text)
	}
	final := tf.Finalized
	attrs := tracing.STTAttributes{
		Service:    t.baseline().Service,
		Model:      t.baseline().Model,
		Transcript: strings.TrimSpace(strings.Join(t.segments, " ")),
		Final:      &final,
		Language:   tf.Language,
		UserID:     tf.UserID,
		VADEnabled: t.baseline().VADEnabled,
	}
	tracing.SetSTTAttributes(t.span, attrs)
	if tf.Finalized {
		t.closeLocked(time.Time{})
	}
}

// appendLocked folds a transcript into the segment. A service that re-sends the
// utterance as it grows extends the piece it is growing; one that sends each
// piece of the utterance once adds to them. Without the distinction a service of
// the second kind would report only its last piece, and the start of what the
// user said would be lost. The caller holds t.mu.
func (t *segmentTracer) appendLocked(text string) {
	if n := len(t.segments); n > 0 && strings.HasPrefix(text, t.segments[n-1]) {
		t.segments[n-1] = text
		return
	}
	t.segments = append(t.segments, text)
}

// recordTTFB attaches the measured latency to the open span.
func (t *segmentTracer) recordTTFB(d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.span == nil {
		return
	}
	seconds := d.Seconds()
	tracing.SetSTTAttributes(t.span, tracing.STTAttributes{
		Service:    t.baseline().Service,
		Model:      t.baseline().Model,
		VADEnabled: t.baseline().VADEnabled,
		TTFB:       &seconds,
	})
}

// addUsage accumulates what the service reported for the segment being
// transcribed, to be attached when its span closes.
func (t *segmentTracer) addUsage(u frames.STTUsage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.usage == nil {
		t.usage = &frames.STTUsage{}
	}
	t.usage.AudioSeconds += u.AudioSeconds
}

// abandon closes an open span that no finalized transcript ever closed, ending
// it at end (or now, when end is zero). The segment is marked incomplete: the
// service was still expected to say what it heard, and stopped without doing so.
func (t *segmentTracer) abandon(end time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.span == nil {
		return
	}
	t.span.SetAttributes(sttIncomplete)
	t.closeLocked(end)
}

// closeLocked attaches the segment's usage and ends its span, at end when one is
// given and where it stands otherwise, then clears the state so the next segment
// starts clean. The caller holds t.mu.
func (t *segmentTracer) closeLocked(end time.Time) {
	if t.usage != nil {
		t.span.SetAttributes(audioSecondsAttr(t.usage.AudioSeconds))
		tracing.SetSTTUsage(t.spanCtx, t.baseline().Model,
			time.Duration(t.usage.AudioSeconds*float64(time.Second)))
	}
	if end.IsZero() {
		t.span.End()
	} else {
		t.span.End(trace.WithTimestamp(end))
	}
	t.span, t.spanCtx = nil, nil
	t.anchor = time.Time{}
	t.segments = nil
	t.usage = nil
}

// close ends whatever is open, for a service being torn down.
func (t *segmentTracer) close() { t.abandon(time.Time{}) }

// recordError marks the segment's span with a transcription that failed. The
// span is opened for it when none was, so a segment the service never managed to
// transcribe is still recorded, with the reason it produced no words.
func (t *segmentTracer) recordError(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.openLocked()
	t.span.RecordError(err)
	t.span.SetAttributes(sttIncomplete)
	t.closeLocked(time.Time{})
}
