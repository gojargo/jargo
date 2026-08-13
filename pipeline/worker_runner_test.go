package pipeline_test

import (
	"context"
	"os"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// A pipeline worker under a runner. The runner ends when its only root worker
// does, a signal cancels the worker so the pipeline drains a CancelFrame rather
// than dying where it stands, and a canceled parent context does the same.
//
// These cover the runner's signal and shutdown paths, which the ported upstream
// suite does not reach.

// runnerWorker builds a one-processor worker that reports when its pipeline has
// come up and which frame ended it.
func runnerWorker() (w *pipeline.Worker, ready chan struct{}, ended func() frames.Frame) {
	ready = make(chan struct{})
	var once sync.Once

	var mu sync.Mutex
	var last frames.Frame

	w = pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Name:        "worker",
		IdleTimeout: -1,
	})
	events.On(w.Events(), pipeline.EventPipelineStarted,
		func(_ context.Context, _ *frames.StartFrame) { once.Do(func() { close(ready) }) })
	events.On(w.Events(), pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		last = f
		mu.Unlock()
	})

	return w, ready, func() frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// runUnderRunner adds the worker to a runner and runs it, reporting a channel
// closed when the run returns.
func runUnderRunner(ctx context.Context, t *testing.T, r *workers.Runner, w *pipeline.Worker) chan struct{} {
	t.Helper()
	r.AddWorkers(ctx, w)
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := r.Run(ctx, workers.RunOptions{}); err != nil {
			t.Errorf("run: %v", err)
		}
	}()
	return done
}

func TestRunnerEndsWhenTheWorkerDoes(t *testing.T) {
	t.Parallel()
	noSignals := false
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &noSignals})
	w, ready, ended := runnerWorker()

	done := runUnderRunner(t.Context(), t, r, w)
	awaitClosed(t, ready, "the pipeline starting")
	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")

	if _, ok := ended().(*frames.EndFrame); !ok {
		t.Errorf("the pipeline ended on %T, want an EndFrame", ended())
	}
	if !w.HasFinished() {
		t.Error("the worker has not finished")
	}
}

func TestRunnerStopsTheWorkerWhenTheParentContextEnds(t *testing.T) {
	t.Parallel()
	noSignals := false
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &noSignals})
	w, ready, ended := runnerWorker()

	ctx, cancel := context.WithCancel(context.Background())
	done := runUnderRunner(ctx, t, r, w)
	awaitClosed(t, ready, "the pipeline starting")
	cancel()
	awaitClosed(t, done, "the run returning")

	// Cut short, but through the pipeline: every processor was told, so each
	// closed what it had open rather than being dropped where it stood.
	if _, ok := ended().(*frames.CancelFrame); !ok {
		t.Errorf("the pipeline ended on %T, want a CancelFrame", ended())
	}
}

func TestRunnerCancelsTheWorkerOnASignal(t *testing.T) {
	t.Parallel()
	handleTerm := true
	noInterrupt := false
	r := workers.NewRunner(workers.RunnerConfig{
		HandleInterrupt: &noInterrupt,
		HandleTerminate: handleTerm,
	})
	w, ready, ended := runnerWorker()

	done := runUnderRunner(t.Context(), t, r, w)

	// The runner registers its handler before it starts the workers, so the
	// pipeline being up means the signal is caught rather than killing the test
	// binary.
	awaitClosed(t, ready, "the pipeline starting")

	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("could not find the test process: %v", err)
	}
	if err := self.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("could not signal the test process: %v", err)
	}

	awaitClosed(t, done, "the run returning")

	// The signal cancels rather than ending, so the pipeline drains a
	// CancelFrame and the run finishes cleanly instead of reporting the
	// interrupt as a failure.
	deadline := time.After(3 * time.Second)
	for {
		if _, ok := ended().(*frames.CancelFrame); ok {
			return
		}
		select {
		case <-time.After(10 * time.Millisecond):
		case <-deadline:
			t.Fatalf("the pipeline ended on %T, want a CancelFrame", ended())
		}
	}
}
