package pipeline_test

import (
	"context"
	"errors"
	"os"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// Tests for the Runner, which drives one task to completion. With signal
// handling on, the first interrupt cancels the task so the pipeline drains a
// CancelFrame rather than dying where it stands, and the run keeps going on the
// parent context so that cleanup can finish.

// runnerTask builds a one-processor task that reports when the pipeline has
// come up and which frame ended it.
func runnerTask() (task *pipeline.Task, ready chan struct{}, ended func() frames.Frame) {
	ready = make(chan struct{})

	var mu sync.Mutex
	var last frames.Frame

	task = pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		OnPipelineStarted: func(*frames.StartFrame) { close(ready) },
		OnPipelineFinished: func(f frames.Frame) {
			mu.Lock()
			last = f
			mu.Unlock()
		},
	})

	return task, ready, func() frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// runRunner runs task on runner and returns a channel carrying the run error.
func runRunner(ctx context.Context, r *pipeline.Runner, task *pipeline.Task) chan error {
	done := make(chan error, 1)
	go func() { done <- r.Run(ctx, task) }()
	return done
}

// runnerErr waits for the run to finish and returns its error, unlike waitDone
// which fails the test on one.
func runnerErr(t *testing.T, done chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("the runner did not finish")
		return nil
	}
}

// TestRunnerRunsTaskWithoutSignalHandling checks that WithoutSignalHandling
// leaves shutdown to the caller: the run is the task's own, and it ends when the
// task does.
func TestRunnerRunsTaskWithoutSignalHandling(t *testing.T) {
	task, ready, ended := runnerTask()
	runner := pipeline.NewRunner(pipeline.WithoutSignalHandling())

	done := runRunner(context.Background(), runner, task)
	started(t, ready)
	task.StopWhenDone()
	waitDone(t, done)

	if _, ok := ended().(*frames.EndFrame); !ok {
		t.Fatalf("the pipeline ended with %T, want an EndFrame", ended())
	}
	if !task.HasFinished() {
		t.Fatal("the task did not finish")
	}
}

// TestRunnerRunsTaskToCompletion checks that a runner watching for signals still
// returns as soon as the task ends on its own.
func TestRunnerRunsTaskToCompletion(t *testing.T) {
	task, ready, ended := runnerTask()
	runner := pipeline.NewRunner()

	done := runRunner(context.Background(), runner, task)
	started(t, ready)
	task.StopWhenDone()
	waitDone(t, done)

	if _, ok := ended().(*frames.EndFrame); !ok {
		t.Fatalf("the pipeline ended with %T, want an EndFrame", ended())
	}
}

// TestRunnerStopsWhenTheParentContextEnds checks that a canceled parent stops
// the task the same way, and that the runner reports the context error rather
// than swallowing it.
func TestRunnerStopsWhenTheParentContextEnds(t *testing.T) {
	task, ready, ended := runnerTask()
	runner := pipeline.NewRunner()

	ctx, cancel := context.WithCancel(context.Background())
	done := runRunner(ctx, runner, task)
	started(t, ready)
	cancel()

	if err := runnerErr(t, done); !errors.Is(err, context.Canceled) {
		t.Fatalf("the runner returned %v, want context.Canceled", err)
	}
	if _, ok := ended().(*frames.CancelFrame); !ok {
		t.Fatalf("the pipeline ended with %T, want a CancelFrame", ended())
	}
}

// TestRunnerCancelsTheTaskOnAnInterrupt checks the signal path: the interrupt
// cancels the task, so the pipeline shuts down on a CancelFrame and the run
// finishes cleanly rather than reporting the interrupt as an error.
func TestRunnerCancelsTheTaskOnAnInterrupt(t *testing.T) {
	task, ready, ended := runnerTask()
	runner := pipeline.NewRunner()

	done := runRunner(context.Background(), runner, task)

	// The runner registers its handler before it starts the task, so the
	// pipeline being up means the signal will be caught rather than kill the
	// test binary.
	started(t, ready)

	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("could not find the test process: %v", err)
	}
	if err := self.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("could not signal the test process: %v", err)
	}

	waitDone(t, done)

	if _, ok := ended().(*frames.CancelFrame); !ok {
		t.Fatalf("the pipeline ended with %T, want a CancelFrame", ended())
	}
}
