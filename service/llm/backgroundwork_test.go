package llm_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// An async call is work the conversation waits on; a synchronous one holds it.

// callFrames records the function-call and background-work frames a service
// sends, once per broadcast rather than once per direction, in the order it
// sends them.
type callFrames struct {
	mu     sync.Mutex
	ids    map[uint64]bool
	frames []frames.Frame
}

func watchCallFrames(svc *llm.Base) *callFrames {
	w := &callFrames{ids: make(map[uint64]bool)}
	events.On(svc.Events(), processor.EventBeforePushFrame, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.FunctionCallsStartedFrame, *frames.FunctionCallInProgressFrame,
			*frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame,
			*frames.BackgroundWorkStartedFrame, *frames.BackgroundWorkFinishedFrame:
		default:
			return
		}
		w.mu.Lock()
		defer w.mu.Unlock()
		// The second half of a broadcast names the first as its sibling.
		if sibling, ok := f.Base().BroadcastSiblingID(); ok && w.ids[sibling] {
			return
		}
		w.ids[f.ID()] = true
		w.frames = append(w.frames, f)
	})
	return w
}

func (w *callFrames) snapshot() []frames.Frame {
	w.mu.Lock()
	defer w.mu.Unlock()
	return slices.Clone(w.frames)
}

// kinds names each recorded frame by its type.
func kinds(fs []frames.Frame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		switch f.(type) {
		case *frames.FunctionCallsStartedFrame:
			out = append(out, "started")
		case *frames.FunctionCallInProgressFrame:
			out = append(out, "in_progress")
		case *frames.FunctionCallResultFrame:
			out = append(out, "result")
		case *frames.FunctionCallCancelFrame:
			out = append(out, "cancel")
		case *frames.BackgroundWorkStartedFrame:
			out = append(out, "work_started")
		case *frames.BackgroundWorkFinishedFrame:
			out = append(out, "work_finished")
		}
	}
	return out
}

// runCall runs one call of get_weather through svc in a pipeline, and returns
// the record of what the service sends, with the function that stops the
// pipeline.
func runCall(t *testing.T, svc *llm.Base) (*callFrames, func()) {
	t.Helper()
	watch := watchCallFrames(svc)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))
	return watch, func() {
		task.StopWhenDone()
		<-runDone
	}
}

// waitForKinds waits until the service has sent want, and reports what it sent.
func waitForKinds(t *testing.T, watch *callFrames, want []string) []frames.Frame {
	t.Helper()
	waitFor(3*time.Second, func() bool { return len(watch.snapshot()) >= len(want) })
	// Anything that would follow has had the chance to.
	time.Sleep(50 * time.Millisecond)
	got := watch.snapshot()
	if !slices.Equal(kinds(got), want) {
		t.Fatalf("frames = %v, want %v", kinds(got), want)
	}
	return got
}

func workID(t *testing.T, f frames.Frame) string {
	t.Helper()
	switch fr := f.(type) {
	case *frames.BackgroundWorkStartedFrame:
		return fr.WorkID
	case *frames.BackgroundWorkFinishedFrame:
		return fr.WorkID
	}
	t.Fatalf("%v is no background-work frame", f)
	return ""
}

func TestAnAsyncCallIsBracketedByTheWorkFrames(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		notFinal := false
		if err := p.Result(ctx, "working", &frames.FunctionCallResultProperties{IsFinal: &notFinal}); err != nil {
			return err
		}
		return p.Result(ctx, "done", nil)
	}, llm.WithCancelOnInterruption(false))

	watch, stop := runCall(t, svc)
	defer stop()

	got := waitForKinds(t, watch, []string{
		"started", "in_progress", "work_started", "result", "result", "work_finished",
	})
	if workID(t, got[2]) != "call_1" || workID(t, got[len(got)-1]) != "call_1" {
		t.Errorf("work ids = %q, %q, want the call's", workID(t, got[2]), workID(t, got[len(got)-1]))
	}
}

func TestASynchronousCallBroadcastsNoWorkFrames(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "done", nil)
	})

	watch, stop := runCall(t, svc)
	defer stop()

	waitForKinds(t, watch, []string{"started", "in_progress", "result"})
}

// The canceling tool's own result runs the model, so the cancellation does not
// ask for inference; the call's work is over with it.
func TestAModelRequestedCancellationFinishesTheWork(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	running := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, _ llm.FunctionCallParams) error {
		close(running)
		<-ctx.Done()
		return ctx.Err()
	}, llm.WithCancelOnInterruption(false))

	watch, stop := runCall(t, svc)
	defer stop()
	select {
	case <-running:
	case <-time.After(3 * time.Second):
		t.Fatal("the handler never ran")
	}
	if err := svc.CancelAsyncToolCall(context.Background(), "call_1"); err != nil {
		t.Fatal(err)
	}

	got := waitForKinds(t, watch, []string{"started", "in_progress", "work_started", "cancel", "work_finished"})
	if cancel, _ := got[3].(*frames.FunctionCallCancelFrame); cancel.RunLLM {
		t.Error("the cancellation asked for inference")
	}
}

// The deadline covers the handler; a call it deferred still settles, and its
// work finishes with the result.
func TestADeferredResultOfAnAsyncCallFinishesTheWork(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{}, llm.WithFunctionCallTimeout(50*time.Millisecond))
	deferred := make(chan llm.FunctionCallResultCallback, 1)
	svc.RegisterFunction("get_weather", func(_ context.Context, p llm.FunctionCallParams) error {
		deferred <- p.Result
		return nil
	}, llm.WithCancelOnInterruption(false))

	watch, stop := runCall(t, svc)
	defer stop()
	var report llm.FunctionCallResultCallback
	select {
	case report = <-deferred:
	case <-time.After(3 * time.Second):
		t.Fatal("the handler never ran")
	}
	time.Sleep(150 * time.Millisecond)
	if err := report(context.Background(), "ok", nil); err != nil {
		t.Fatal(err)
	}

	got := waitForKinds(t, watch, []string{"started", "in_progress", "work_started", "result", "work_finished"})
	if result, _ := got[3].(*frames.FunctionCallResultFrame); result.Result != "ok" {
		t.Errorf("result = %q, want the deferred one", result.Result)
	}
}

func TestAnAsyncToolTimingOutAfterAnUpdateFinishesItsWork(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		notFinal := false
		if err := p.Result(ctx, "still looking", &frames.FunctionCallResultProperties{
			IsFinal: &notFinal,
		}); err != nil {
			return err
		}
		<-ctx.Done()
		return ctx.Err()
	}, llm.WithCancelOnInterruption(false), llm.WithTimeout(50*time.Millisecond))

	watch, stop := runCall(t, svc)
	defer stop()

	got := waitForKinds(t, watch, []string{
		"started", "in_progress", "work_started", "result", "cancel", "work_finished",
	})
	if result, _ := got[3].(*frames.FunctionCallResultFrame); result.Properties.Final() {
		t.Error("the update was reported as a final result")
	}
	if cancel, _ := got[4].(*frames.FunctionCallCancelFrame); !cancel.RunLLM {
		t.Error("the deadline did not ask for inference")
	}
}

func TestAHangingAsyncToolTimesOutAndFinishesItsWork(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, _ llm.FunctionCallParams) error {
		<-ctx.Done()
		return ctx.Err()
	}, llm.WithCancelOnInterruption(false), llm.WithTimeout(50*time.Millisecond))

	watch, stop := runCall(t, svc)
	defer stop()

	got := waitForKinds(t, watch, []string{"started", "in_progress", "work_started", "cancel", "work_finished"})
	if cancel, _ := got[3].(*frames.FunctionCallCancelFrame); !cancel.RunLLM {
		t.Error("the deadline did not ask for inference")
	}
}
