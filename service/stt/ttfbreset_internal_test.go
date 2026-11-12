package stt

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// ttfbCounter counts the time-to-first-byte measurements reaching the end of the
// pipeline, and whether the frame it watches for has reached it.
type ttfbCounter struct {
	mu      sync.Mutex
	seen    []time.Duration
	watch   frames.Frame
	watched bool
}

func (c *ttfbCounter) observe(_ context.Context, f frames.Frame) {
	if f == c.watch {
		c.mu.Lock()
		c.watched = true
		c.mu.Unlock()
	}
	mf, ok := f.(*frames.MetricsFrame)
	if !ok {
		return
	}
	for _, d := range mf.Data {
		if td, ok := d.(frames.TTFBMetricsData); ok {
			c.mu.Lock()
			c.seen = append(c.seen, td.Value)
			c.mu.Unlock()
		}
	}
}

func (c *ttfbCounter) sawWatched() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.watched
}

func (c *ttfbCounter) reported() []time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]time.Duration(nil), c.seen...)
}

// runMeasured runs svc with metrics on, handing every frame reaching the end of
// the pipeline to the counter and the transcript watcher.
func runMeasured(
	t *testing.T, svc *StreamService, c *ttfbCounter, w *transcriptWatcher,
) (task *pipeline.Worker, stop func()) {
	t.Helper()
	no := false
	task = pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableMetrics: true, SendInitialEmptyMetrics: &no},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(ctx context.Context, f frames.Frame) {
		c.observe(ctx, f)
		w.observe(ctx, f)
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

// measureAfter reports a speech end, queues reset, and then has the provider
// answer with a transcript that closes an utterance. It returns the measurements
// reported for it.
func measureAfter(t *testing.T, reset frames.Frame) []time.Duration {
	t.Helper()

	st := &answeringStream{out: make(chan []Result, 1)}
	svc := NewStream("FakeSTT", &answeringConnector{stream: st}, 16000)
	c := &ttfbCounter{watch: reset}
	w := &transcriptWatcher{}
	task, stop := runMeasured(t, svc, c, w)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(reset)
	// The transcript goes out once the reset has been through the service, so
	// it is ordered after it.
	waitFor(t, "the reset to pass", c.sawWatched)
	st.out <- []Result{{Text: "hello", Final: true, EndOfTurn: true}}
	waitFor(t, "the transcript", func() bool { return len(w.transcripts()) == 1 })
	return c.reported()
}

// A user starting to speak again drops the measurement the last speech end
// started, so a transcript the provider sends on its own endpointing in the new
// utterance is not measured against a speech end that no longer stands.
func TestStreamServiceResetsTTFBWhenTheUserStartsSpeaking(t *testing.T) {
	t.Parallel()

	if seen := measureAfter(t, frames.NewVADUserStartedSpeakingFrame(0.2, time.Now())); len(seen) != 0 {
		t.Errorf("reported %v after the user started speaking again, want nothing", seen)
	}
}

// An interruption stops the clock where it stands: what was measured so far is
// reported, once, and the transcript after it is not measured against the
// speech end before it.
func TestStreamServiceStopsTTFBOnAnInterruption(t *testing.T) {
	t.Parallel()

	seen := measureAfter(t, frames.NewInterruptionFrame())
	if len(seen) != 1 {
		t.Fatalf("reported %v around an interruption, want the one the interruption stopped", seen)
	}
	if seen[0] > time.Second {
		t.Errorf("reported %v, want the time from the speech end to the interruption", seen[0])
	}
}

// Without a reset between them, the transcript closes the wait the speech end
// started and it is reported, which is what the tests above rely on.
func TestStreamServiceMeasuresWithoutAReset(t *testing.T) {
	t.Parallel()

	if seen := measureAfter(t, frames.NewTextFrame("unrelated")); len(seen) != 1 {
		t.Errorf("reported %v, want one measurement", seen)
	}
}
