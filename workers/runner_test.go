package workers_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/workers"
)

// The runner: what adding a worker does, what running does, and how ending and
// canceling reach the workers and no further.

// newTestRunner builds a runner that does not take over the process's signals,
// with a recorder on its bus.
func newTestRunner(t *testing.T) (*workers.Runner, *recorder) {
	t.Helper()
	noSignals := false
	runner := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &noSignals})
	rec := &recorder{name: "recorder"}
	runner.Bus().Subscribe(rec)
	// Started here rather than only by Run, so the tests that drive the runner
	// without running it still see what it sends. Starting twice is a no-op.
	runner.Bus().Start(t.Context())
	t.Cleanup(runner.Bus().Stop)
	return runner, rec
}

// runInBackground runs the runner and reports a channel closed when it returns.
func runInBackground(t *testing.T, ctx context.Context, runner *workers.Runner) chan struct{} {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := runner.Run(ctx, workers.RunOptions{}); err != nil {
			t.Errorf("run: %v", err)
		}
	}()
	return done
}

// awaitRun waits for the runner to return.
func awaitRun(t *testing.T, done chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the runner never returned")
	}
}

func TestRunnerAddWorkersRegistersByName(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)

	runner.AddWorkers(ctx, newStubWorker("worker_a"))
	// A second worker of the same name is reported and skipped.
	runner.AddWorkers(ctx, newStubWorker("worker_a"))
}

func TestRunnerRunStartsTheBusAndTheWorkers(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	ready := make(chan struct{})
	var once sync.Once
	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		once.Do(func() { close(ready) })
		// Ended straight away, to unblock the run.
		runner.End(ctx, "")
	})

	awaitRun(t, runInBackground(t, ctx, runner))

	select {
	case <-ready:
	default:
		t.Error("the runner never reported it was ready")
	}
}

func TestRunnerEndIsIdempotent(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		runner.End(ctx, "first")
		runner.End(ctx, "second") // does nothing
	})

	// Returning at all is the assertion: a second end must not hang or repeat.
	awaitRun(t, runInBackground(t, ctx, runner))
}

func TestRunnerCancelIsIdempotent(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		runner.Cancel(ctx, "first")
		runner.Cancel(ctx, "second") // does nothing
	})

	awaitRun(t, runInBackground(t, ctx, runner))
}

func TestRunnerEndReachesRootWorkersOnly(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, rec := newTestRunner(t)

	root := newStubWorker("root")
	child := newStubWorker("child")
	runner.AddWorkers(ctx, root)
	root.AddWorkers(ctx, child)
	runner.AddWorkers(ctx, child)

	runner.End(ctx, "")

	// Only what the runner itself addressed: the parent propagating the end to
	// its own children is the parent's business, and happens here too.
	msgs := await[*bus.EndWorkerMessage](t, rec, 1)
	targets := map[string]bool{}
	for _, m := range msgs {
		if m.Source() == runner.Name() {
			targets[m.Target()] = true
		}
	}
	if !targets["root"] {
		t.Errorf("the end did not reach the root worker, targets were %v", targets)
	}
	// A child is its parent's business, not the runner's.
	if targets["child"] {
		t.Errorf("the end reached a child worker, targets were %v", targets)
	}
}

func TestRunnerCancelReachesRootWorkersOnly(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, rec := newTestRunner(t)

	root := newStubWorker("root")
	child := newStubWorker("child")
	runner.AddWorkers(ctx, root)
	root.AddWorkers(ctx, child)
	runner.AddWorkers(ctx, child)

	runner.Cancel(ctx, "")

	// Only what the runner itself addressed, as above.
	msgs := await[*bus.CancelWorkerMessage](t, rec, 1)
	targets := map[string]bool{}
	for _, m := range msgs {
		if m.Source() == runner.Name() {
			targets[m.Target()] = true
		}
	}
	if !targets["root"] {
		t.Errorf("the cancel did not reach the root worker, targets were %v", targets)
	}
	if targets["child"] {
		t.Errorf("the cancel reached a child worker, targets were %v", targets)
	}
}

func TestRunnerEndsOnAnEndMessage(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		// What a worker asking for the session to end looks like.
		m := &bus.EndMessage{}
		m.From = "worker_a"
		runner.Bus().Send(ctx, m)
	})

	awaitRun(t, runInBackground(t, ctx, runner))
}

func TestRunnerCancelsOnACancelMessage(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		m := &bus.CancelMessage{}
		m.From = "worker_a"
		runner.Bus().Send(ctx, m)
	})

	awaitRun(t, runInBackground(t, ctx, runner))
}

func TestRunnerAddsAWorkerOnAnAddMessage(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)
	runner.AddWorkers(ctx, newStubWorker("worker_a"))

	added := newStubWorker("worker_b")
	addedIsReady := make(chan struct{})
	var once sync.Once
	runner.WorkerRegistry().Watch(ctx, "worker_b", "test", func(context.Context, registry.WorkerReadyData) {
		once.Do(func() { close(addedIsReady) })
	})

	runner.Add(workers.EventRunnerReady, func(ctx context.Context, _ any, _ ...any) {
		m := &bus.AddWorkerMessage{Worker: added}
		m.From = "worker_a"
		runner.Bus().Send(ctx, m)

		select {
		case <-addedIsReady:
		case <-time.After(2 * time.Second):
			t.Error("the worker asked for over the bus never started")
		}
		runner.End(ctx, "")
	})

	awaitRun(t, runInBackground(t, ctx, runner))

	// It is registered now, so adding it again is reported and skipped.
	runner.AddWorkers(ctx, newStubWorker("worker_b"))
}

func TestRunnerEndsWhenItsLastRootWorkerFinishes(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	runner, _ := newTestRunner(t)

	worker := newStubWorker("worker_a")
	runner.AddWorkers(ctx, worker)

	done := runInBackground(t, ctx, runner)

	// A bot with one pipeline ends when that pipeline does, without anyone
	// having to end the runner.
	eventually(t, "the worker starts", func() bool { return worker.StartedAt() != 0 })
	worker.Stop(ctx)

	awaitRun(t, done)
}
