package pipeline

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// passthrough forwards every frame, standing for an ordinary pipeline.
type passthrough struct{ *processor.Base }

func newPassthrough() *passthrough {
	p := &passthrough{}
	p.Base = processor.New("Passthrough", p)
	return p
}

func (p *passthrough) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// startedWorker runs a worker over a one-processor pipeline and returns it once
// the pipeline is up, along with the frames its finished event carried and a
// function that ends the run.
func startedWorker(t *testing.T) (w *Worker, finished func() []frames.Frame, stop func()) {
	t.Helper()
	w = NewWorker(New(newPassthrough()), WorkerConfig{
		IdleTimeout:   -1,
		CancelTimeout: time.Second,
	})

	var mu sync.Mutex
	var got []frames.Frame
	ready := make(chan struct{})
	var once sync.Once
	events.On(&w.Registry, EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		once.Do(func() { close(ready) })
	})
	events.On(&w.Registry, EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})

	runCtx, cancelRun := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = w.Run(runCtx); close(done) }()

	select {
	case <-ready:
	case <-time.After(5 * time.Second):
		cancelRun()
		t.Fatal("the pipeline never started")
	}

	return w, func() []frames.Frame {
			// The handlers run off the frame path, so let them catch up the way
			// the run does before it returns.
			w.Registry.Cleanup(context.Background())
			mu.Lock()
			defer mu.Unlock()
			return append([]frames.Frame(nil), got...)
		}, func() {
			cancelRun()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Error("the run never returned")
			}
		}
}

// TestContextEndReportsFinishedWhenACancelWasAlreadyAskedFor covers the run's
// context ending after something else asked the worker to cancel.
//
// The run loop is what drains the worker's queue, so once it has stopped, the
// CancelFrame that request queued is stranded there: the pipeline is never told
// the run is over, and neither is anything watching the worker. The frame goes
// straight into the pipeline instead, carrying the reason that was asked for.
func TestContextEndReportsFinishedWhenACancelWasAlreadyAskedFor(t *testing.T) {
	w, finished, stop := startedWorker(t)
	defer stop()

	// Exactly the state the runner leaves behind when it cancels a worker as the
	// process is shutting down: the request is in, its frame is in the queue, and
	// the run loop has stopped draining.
	w.mu.Lock()
	w.canceling = true
	w.cancelReason = "hung up"
	w.mu.Unlock()

	ended := w.cancelOnContextEnd(context.Background())

	cancelFrame, ok := ended.(*frames.CancelFrame)
	if !ok {
		t.Fatalf("the run ended on %T, want a CancelFrame", ended)
	}
	if cancelFrame.Reason != "hung up" {
		t.Errorf("reason = %q, want the one the cancellation was asked for", cancelFrame.Reason)
	}
	got := finished()
	if len(got) != 1 {
		t.Fatalf("the pipeline reported finished %d times, want once", len(got))
	}
	if _, ok := got[0].(*frames.CancelFrame); !ok {
		t.Errorf("finished on %T, want a CancelFrame", got[0])
	}
}

// TestContextEndDoesNotSendASecondCancelFrame covers the run's context ending
// while a CancelFrame it already sent is still traveling: the pipeline has been
// told, so it is waited out rather than told again.
func TestContextEndDoesNotSendASecondCancelFrame(t *testing.T) {
	w, finished, stop := startedWorker(t)
	defer stop()

	// What the run loop leaves behind when its own wait is cut short.
	sent := frames.NewCancelFrame()
	sent.Reason = "already going"
	w.mu.Lock()
	w.canceling = true
	w.cancelFrame = sent
	w.mu.Unlock()

	ended := w.cancelOnContextEnd(context.Background())

	if ended != frames.Frame(sent) {
		t.Errorf("the run ended on %v, want the frame already sent", ended)
	}
	if got := finished(); len(got) != 1 {
		t.Fatalf("the pipeline reported finished %d times, want once", len(got))
	}
}

// TestThePipelineFinishesOnce covers the report being made once however often
// the paths that make it are reached.
func TestThePipelineFinishesOnce(t *testing.T) {
	w, finished, stop := startedWorker(t)
	defer stop()
	ctx := context.Background()

	w.callPipelineFinished(ctx, frames.NewCancelFrame())
	w.callPipelineFinished(ctx, frames.NewCancelFrame())
	w.callPipelineFinished(ctx, frames.NewEndFrame())
	w.Registry.Cleanup(ctx)

	if got := finished(); len(got) != 1 {
		t.Errorf("the pipeline reported finished %d times, want once", len(got))
	}
}
