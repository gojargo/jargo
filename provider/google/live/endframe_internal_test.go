package live

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// These tests are not ports of upstream tests: upstream has none for holding
// the EndFrame while the model answers.

// runningService runs a service for srv in a pipeline, returning the worker and
// a channel closed when the run ends.
func runningService(t *testing.T, srv *recordingSession) (*Service, *pipeline.Worker, <-chan struct{}) {
	t.Helper()
	s := New(Config{APIKey: "k", BaseURL: wsURL(srv)})
	w := pipeline.NewWorker(pipeline.New(s), pipeline.WorkerConfig{})
	started := make(chan struct{})
	events.On(&w.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) { close(started) })
	done := make(chan struct{})
	go func() {
		_ = w.Run(context.Background())
		close(done)
	}()
	t.Cleanup(func() {
		w.Cancel(context.Background(), "")
		<-done
	})
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline never started")
	}
	waitFor(t, "the session to be ready", s.ready.Load)
	return s, w, done
}

// An EndFrame arriving while the model answers is held until the answer
// finishes, so the pipeline does not end mid-reply.
func TestAnEndFrameWaitsForTheTurnToFinish(t *testing.T) {
	s, w, done := runningService(t, newRecordingSession(t))
	s.startResponse(t.Context(), true)

	w.StopWhenDone()
	select {
	case <-done:
		t.Fatal("the pipeline ended while the model was still answering")
	case <-time.After(200 * time.Millisecond):
	}

	s.handleTurnComplete(t.Context(), nil)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not end once the turn finished")
	}
}

// A held EndFrame is released anyway if the turn never finishes.
func TestAHeldEndFrameIsReleasedAfterTheDeferral(t *testing.T) {
	s, w, done := runningService(t, newRecordingSession(t))
	s.startResponse(t.Context(), true)

	w.StopWhenDone()
	var timer *time.Timer
	waitFor(t, "the EndFrame to be held", func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		timer = s.endDeferral
		return timer != nil
	})
	// Bring the deferral forward rather than waiting it out.
	timer.Reset(time.Millisecond)

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the held EndFrame was never released")
	}
}

// An EndFrame with no answer in progress is processed at once.
func TestAnEndFrameWithNoAnswerInProgressIsNotHeld(t *testing.T) {
	_, w, done := runningService(t, newRecordingSession(t))
	w.StopWhenDone()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not end")
	}
}
