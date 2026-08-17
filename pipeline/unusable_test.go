package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// erroring reports an error every time it sees a text frame, and forwards
// everything so the lifecycle frames still reach the sink.
type erroring struct {
	*processor.Base
	category errs.Category
}

func newErroring(name string, category errs.Category) *erroring {
	e := &erroring{category: category}
	e.Base = processor.New(name, e)
	return e
}

func (e *erroring) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		ef := frames.NewErrorFrame("service failed")
		ef.Category = e.category
		e.PushErrorFrame(ctx, ef, false)
	}
	return e.PushFrame(ctx, f, dir)
}

// unusableRun runs a one-processor pipeline under a policy and reports the
// errors that cost their processor its usefulness alongside the frames that
// ended the run.
func unusableRun(
	t *testing.T, p processor.Processor, policy pipeline.ProcessorUnusablePolicy, feed []frames.Frame,
) (unusable []*frames.ErrorFrame, finished []frames.Frame) {
	t.Helper()
	worker := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ProcessorUnusablePolicy: policy,
	})

	var mu sync.Mutex
	events.On(&worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		defer mu.Unlock()
		// This is how application code tells the two kinds of error apart, with
		// no dedicated event of its own.
		if src, ok := ef.Source.(processor.Processor); ok && !src.Usable() {
			unusable = append(unusable, ef)
		}
	})
	events.On(&worker.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		finished = append(finished, f)
	})

	for _, f := range feed {
		worker.QueueFrame(f)
	}

	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the worker never finished")
	}

	mu.Lock()
	defer mu.Unlock()
	return unusable, finished
}

func hasFrame[T frames.Frame](got []frames.Frame) bool {
	for _, f := range got {
		if _, ok := f.(T); ok {
			return true
		}
	}
	return false
}

func TestContinueKeepsThePipelineRunning(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errs.Authentication),
		pipeline.UnusableContinue,
		[]frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if unusable[0].Category != errs.Authentication {
		t.Errorf("category: got %q, want %q", unusable[0].Category, errs.Authentication)
	}
	// The pipeline ran until the EndFrame that was queued, not because of the error.
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestEndStopsThePipeline(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errs.Authentication),
		pipeline.UnusableEnd, []frames.Frame{frames.NewTextFrame("hello")})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestCancelStopsThePipeline(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errs.Authentication),
		pipeline.UnusableCancel, []frames.Frame{frames.NewTextFrame("hello")})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if !hasFrame[*frames.CancelFrame](finished) {
		t.Errorf("finished with %v, want a CancelFrame", finished)
	}
}

func TestDefaultPolicyIsContinue(t *testing.T) {
	var cfg pipeline.WorkerConfig
	if cfg.ProcessorUnusablePolicy != pipeline.UnusableContinue {
		t.Errorf("default policy is %q, want %q", cfg.ProcessorUnusablePolicy, pipeline.UnusableContinue)
	}
}

func TestTransientErrorsLeaveThePipelineRunning(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errs.Server),
		pipeline.UnusableEnd,
		[]frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()})

	if len(unusable) != 0 {
		t.Errorf("reported %d unusable errors, want none", len(unusable))
	}
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestUnclassifiedErrorsLeaveThePipelineRunning(t *testing.T) {
	unusable, _ := unusableRun(t, newErroring("Erroring", errs.Unknown),
		pipeline.UnusableEnd,
		[]frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()})

	if len(unusable) != 0 {
		t.Errorf("reported %d unusable errors, want none", len(unusable))
	}
}

func TestAProcessorIsActedOnOnce(t *testing.T) {
	// Every error is reported, but the policy answers the first. Under the
	// cancel policy a second answer would cancel a run already ending.
	p := newErroring("Erroring", errs.Authentication)
	unusable, finished := unusableRun(t, p, pipeline.UnusableContinue, []frames.Frame{
		frames.NewTextFrame("one"), frames.NewTextFrame("two"),
		frames.NewTextFrame("three"), frames.NewEndFrame(),
	})

	if len(unusable) != 3 {
		t.Errorf("reported %d unusable errors, want 3", len(unusable))
	}
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestEachProcessorIsActedOnSeparately(t *testing.T) {
	first := newErroring("First", errs.Authentication)
	second := newErroring("Second", errs.Authentication)
	worker := pipeline.NewWorker(pipeline.New(first, second), pipeline.WorkerConfig{
		ProcessorUnusablePolicy: pipeline.UnusableContinue,
	})

	worker.QueueFrame(frames.NewTextFrame("hello"))
	worker.QueueFrame(frames.NewEndFrame())
	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the worker never finished")
	}

	// The text frame reaches both processors, so both report an error.
	if first.Usable() || second.Usable() {
		t.Errorf("usability: first %t, second %t, want both unusable", first.Usable(), second.Usable())
	}
}
