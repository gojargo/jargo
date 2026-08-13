package pipeline_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// A pipeline worker taking part in a session: it becomes ready when its
// pipeline is up, it is activated and deactivated over the bus, and it is ended
// or canceled by driving a frame through the pipeline rather than stopping
// where it stands.
//
// Ported from the upstream worker suite. The cases that drive frames through a
// bridged pipeline wait for the bus edges.

const runner = "test-runner"

// newSessionWorker builds a worker on an identity pipeline, attached to a bus
// and a registry as a runner would attach it.
func newSessionWorker(t *testing.T, cfg pipeline.WorkerConfig) (*pipeline.Worker, *bus.AsyncQueueBus) {
	t.Helper()
	if cfg.Name == "" {
		cfg.Name = "test"
	}
	// Nothing in these tests speaks, so the pipeline must not be canceled out
	// from under them for being quiet.
	cfg.IdleTimeout = -1
	w := pipeline.NewWorker(pipeline.New(newEcho()), cfg)

	msgBus := bus.NewAsyncQueueBus()
	reg := registry.New(runner)
	w.Attach(t.Context(), reg, msgBus.Bus)
	msgBus.Start(t.Context())
	t.Cleanup(msgBus.Stop)
	return w, msgBus
}

// runWorker runs a worker in the background and reports a channel closed when
// its run returns.
func runWorker(t *testing.T, w *pipeline.Worker) chan struct{} {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := w.Run(context.Background()); err != nil {
			t.Errorf("run: %v", err)
		}
	}()
	return done
}

func awaitClosed(t *testing.T, c <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-c:
	case <-time.After(5 * time.Second):
		t.Fatalf("%s never happened", what)
	}
}

func TestWorkerStartedAtUnsetBeforeItRuns(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{})
	if got := w.StartedAt(); got != 0 {
		t.Errorf("started at = %v before the pipeline runs, want zero", got)
	}
}

func TestWorkerIsReadyOncePipelineStarts(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{})

	started := make(chan struct{})
	events.On(w.Events(), pipeline.EventPipelineStarted,
		func(_ context.Context, _ *frames.StartFrame) { close(started) })

	done := runWorker(t, w)
	awaitClosed(t, started, "the pipeline starting")

	// Becoming ready is what the worker does when its pipeline is up, so a
	// worker waiting on this one can address it from here on.
	eventuallyTrue(t, "the worker registers as ready", func() bool { return w.StartedAt() != 0 })

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerActiveByDefaultAndFiresActivated(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{})

	activated := make(chan struct{})
	w.Add(workers.EventActivated, func(context.Context, any, ...any) { close(activated) })

	done := runWorker(t, w)
	awaitClosed(t, activated, "the activation")
	if !w.Active() {
		t.Error("the worker is not active after starting")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerActivatedOverTheBus(t *testing.T) {
	t.Parallel()
	inactive := false
	w, msgBus := newSessionWorker(t, pipeline.WorkerConfig{Active: &inactive})

	activated := make(chan map[string]any, 1)
	w.Add(workers.EventActivated, func(_ context.Context, _ any, args ...any) {
		if len(args) > 0 {
			a, _ := args[0].(map[string]any)
			activated <- a
		}
	})

	started := make(chan struct{})
	events.On(w.Events(), pipeline.EventPipelineStarted,
		func(_ context.Context, _ *frames.StartFrame) { close(started) })

	done := runWorker(t, w)
	awaitClosed(t, started, "the pipeline starting")
	if w.Active() {
		t.Error("the worker started active, want it waiting to be activated")
	}

	args := map[string]any{"messages": []string{"hello"}}
	m := &bus.ActivateWorkerMessage{Args: args}
	m.From = "other"
	m.To = w.Name()
	msgBus.Send(t.Context(), m)

	select {
	case got := <-activated:
		if len(got) != 1 {
			t.Errorf("activation args = %v, want the ones that were sent", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the worker was never activated")
	}
	if !w.Active() {
		t.Error("the worker is not active after being activated")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerEndMessageEndsThePipeline(t *testing.T) {
	t.Parallel()
	w, msgBus := newSessionWorker(t, pipeline.WorkerConfig{})

	finished := make(chan frames.Frame, 1)
	events.On(w.Events(), pipeline.EventPipelineFinished,
		func(_ context.Context, f frames.Frame) { finished <- f })

	done := runWorker(t, w)

	m := &bus.EndWorkerMessage{Reason: "shutdown"}
	m.From = runner
	m.To = w.Name()
	msgBus.Send(t.Context(), m)

	awaitClosed(t, done, "the run returning")

	// Ended by a frame traveling the pipeline, so every processor was told.
	select {
	case f := <-finished:
		if _, ok := f.(*frames.EndFrame); !ok {
			t.Errorf("the pipeline finished on %T, want an EndFrame", f)
		}
	default:
		t.Error("the pipeline never reported finishing")
	}
}

func TestWorkerCancelMessageCancelsThePipeline(t *testing.T) {
	t.Parallel()
	w, msgBus := newSessionWorker(t, pipeline.WorkerConfig{})

	done := runWorker(t, w)

	m := &bus.CancelWorkerMessage{Reason: "abort"}
	m.From = runner
	m.To = w.Name()
	msgBus.Send(t.Context(), m)

	awaitClosed(t, done, "the run returning")
	if !w.HasFinished() {
		t.Error("the worker has not finished after being canceled")
	}
}

func TestWorkerSpeakMessageBecomesAFrame(t *testing.T) {
	t.Parallel()
	w, msgBus := newSessionWorker(t, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TTSSpeakFrame{}),
	})

	spoken := make(chan *frames.TTSSpeakFrame, 1)
	events.On(w.Events(), pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if sf, ok := f.(*frames.TTSSpeakFrame); ok {
			spoken <- sf
		}
	})

	done := runWorker(t, w)

	m := &bus.TTSSpeakMessage{Text: "hello there", AppendToContext: true}
	m.From = "other"
	m.To = w.Name()
	msgBus.Send(t.Context(), m)

	select {
	case f := <-spoken:
		if f.Text != "hello there" {
			t.Errorf("spoke %q, want hello there", f.Text)
		}
		if !f.AppendToContext {
			t.Error("the spoken text was not appended to the conversation")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("nothing was spoken")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerIgnoresASpeakMessageForAnotherWorker(t *testing.T) {
	t.Parallel()
	w, msgBus := newSessionWorker(t, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TTSSpeakFrame{}),
	})

	spoken := make(chan *frames.TTSSpeakFrame, 1)
	events.On(w.Events(), pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if sf, ok := f.(*frames.TTSSpeakFrame); ok {
			spoken <- sf
		}
	})

	done := runWorker(t, w)

	m := &bus.TTSSpeakMessage{Text: "ignored"}
	m.From = "other"
	m.To = "someone-else"
	msgBus.Send(t.Context(), m)

	time.Sleep(100 * time.Millisecond)
	select {
	case f := <-spoken:
		t.Errorf("spoke %q, want nothing: the message was for another worker", f.Text)
	default:
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerHandsOffToItself(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{})

	activations := make(chan struct{}, 4)
	w.Add(workers.EventActivated, func(context.Context, any, ...any) { activations <- struct{}{} })

	done := runWorker(t, w)
	select {
	case <-activations:
	case <-time.After(5 * time.Second):
		t.Fatal("the worker was never activated to begin with")
	}

	// Handing off to itself deactivates and activates again, which is how a
	// worker restarts its own turn.
	w.ActivateWorker(t.Context(), w.Name(), workers.ActivateOptions{DeactivateSelf: true})

	select {
	case <-activations:
	case <-time.After(5 * time.Second):
		t.Fatal("the worker never activated again")
	}
	if !w.Active() {
		t.Error("the worker is not active after handing off to itself")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerQueueFrameReachesThePipeline(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})

	got := make(chan string, 4)
	events.On(w.Events(), pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TextFrame); ok {
			got <- tf.Text
		}
	})

	done := runWorker(t, w)
	w.QueueFrame(frames.NewTextFrame("injected"))

	select {
	case text := <-got:
		if text != "injected" {
			t.Errorf("reached the end of the pipeline: %q, want injected", text)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the frame never reached the end of the pipeline")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerQueueFramesReachThePipelineInOrder(t *testing.T) {
	t.Parallel()
	w, _ := newSessionWorker(t, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})

	got := make(chan string, 4)
	events.On(w.Events(), pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TextFrame); ok {
			got <- tf.Text
		}
	})

	done := runWorker(t, w)
	w.QueueFrames([]frames.Frame{frames.NewTextFrame("a"), frames.NewTextFrame("b")})

	for _, want := range []string{"a", "b"} {
		select {
		case text := <-got:
			if text != want {
				t.Errorf("reached the end of the pipeline: %q, want %q", text, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("the frame %q never reached the end of the pipeline", want)
		}
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestWorkerRunsUnderARunner(t *testing.T) {
	t.Parallel()
	noSignals := false
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &noSignals})

	w := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Name:        "test",
		IdleTimeout: -1,
	})
	r.AddWorkers(t.Context(), w)

	finished := make(chan struct{})
	events.On(w.Events(), pipeline.EventPipelineFinished,
		func(_ context.Context, _ frames.Frame) { close(finished) })

	r.Add(workers.EventRunnerReady, func(context.Context, any, ...any) { w.StopWhenDone() })

	// The runner ends when its only root worker does, so this returns on its
	// own once the pipeline has ended.
	if err := r.Run(t.Context(), workers.RunOptions{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	awaitClosed(t, finished, "the pipeline finishing")
}

// eventuallyTrue waits for cond to hold.
func eventuallyTrue(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for range 500 {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("%s did not happen", what)
}

var _ processor.Processor = (*echo)(nil)
