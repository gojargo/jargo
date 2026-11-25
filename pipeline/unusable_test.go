package pipeline_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// erroring reports an error every time it sees a text frame, and forwards
// everything so the lifecycle frames still reach the sink.
type erroring struct {
	*processor.Base
	category errclass.Category
}

func newErroring(name string, category errclass.Category) *erroring {
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

// permanentlyFailing reports an error every time it sees a text frame, as one
// that keeps recurring whatever its category says.
type permanentlyFailing struct {
	*processor.Base
}

func newPermanentlyFailing() *permanentlyFailing {
	p := &permanentlyFailing{}
	p.Base = processor.New("PermanentlyFailing", p)
	return p
}

func (p *permanentlyFailing) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		p.PushError(ctx, "service failed for good", nil, false,
			processor.WithErrorCategory(errclass.Server), processor.ForceTreatAsPermanent())
	}
	return p.PushFrame(ctx, f, dir)
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
	event.On(&worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		defer mu.Unlock()
		// This is how application code tells the two kinds of error apart, with
		// no dedicated event of its own.
		if src, ok := ef.Source.(processor.Processor); ok && !src.Usable() {
			unusable = append(unusable, ef)
		}
	})
	event.On(&worker.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		finished = append(finished, f)
	})

	runPastErrors(t, worker, feed, countText(feed))

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
	unusable, finished := unusableRun(t, newErroring("Erroring", errclass.Authentication),
		pipeline.UnusableContinue,
		[]frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if unusable[0].Category != errclass.Authentication {
		t.Errorf("category: got %q, want %q", unusable[0].Category, errclass.Authentication)
	}
	// The pipeline ran until the EndFrame that was queued, not because of the error.
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestEndStopsThePipeline(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errclass.Authentication),
		pipeline.UnusableEnd, []frames.Frame{frames.NewTextFrame("hello")})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if !hasFrame[*frames.EndFrame](finished) {
		t.Errorf("finished with %v, want an EndFrame", finished)
	}
}

func TestCancelStopsThePipeline(t *testing.T) {
	unusable, finished := unusableRun(t, newErroring("Erroring", errclass.Authentication),
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
	unusable, finished := unusableRun(t, newErroring("Erroring", errclass.Server),
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
	unusable, _ := unusableRun(t, newErroring("Erroring", errclass.Unknown),
		pipeline.UnusableEnd,
		[]frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()})

	if len(unusable) != 0 {
		t.Errorf("reported %d unusable errors, want none", len(unusable))
	}
}

func TestPermanentErrorsAreActedOnWhateverTheCategory(t *testing.T) {
	unusable, finished := unusableRun(t, newPermanentlyFailing(),
		pipeline.UnusableCancel, []frames.Frame{frames.NewTextFrame("hello")})

	if len(unusable) != 1 {
		t.Fatalf("reported %d unusable errors, want 1", len(unusable))
	}
	if !hasFrame[*frames.CancelFrame](finished) {
		t.Errorf("finished with %v, want a CancelFrame", finished)
	}
}

func TestAProcessorIsActedOnOnce(t *testing.T) {
	p := newErroring("Erroring", errclass.Authentication)
	worker := runActedOn(t, []frames.Frame{
		frames.NewTextFrame("one"), frames.NewTextFrame("two"),
		frames.NewTextFrame("three"), frames.NewEndFrame(),
	}, p)

	// Every error is reported, but the policy is applied to the first.
	if got, want := worker.UnusableProcessors(), []uint64{p.ID()}; !slices.Equal(got, want) {
		t.Errorf("acted on %v, want %v", got, want)
	}
}

func TestEachProcessorIsActedOnSeparately(t *testing.T) {
	first := newErroring("First", errclass.Authentication)
	second := newErroring("Second", errclass.Authentication)
	worker := runActedOn(t, []frames.Frame{frames.NewTextFrame("hello"), frames.NewEndFrame()}, first, second)

	// The text frame reaches both processors, so both report an error.
	want := []uint64{first.ID(), second.ID()}
	slices.Sort(want)
	if got := worker.UnusableProcessors(); !slices.Equal(got, want) {
		t.Errorf("acted on %v, want %v", got, want)
	}
}

// runActedOn runs the processors under the continue policy over feed, in which
// every text frame makes every processor report an error, and returns the worker
// once it has finished.
func runActedOn(t *testing.T, feed []frames.Frame, ps ...processor.Processor) *pipeline.Worker {
	t.Helper()
	worker := pipeline.NewWorker(pipeline.New(ps...), pipeline.WorkerConfig{
		ProcessorUnusablePolicy: pipeline.UnusableContinue,
	})
	runPastErrors(t, worker, feed, countText(feed)*len(ps))
	return worker
}

// runPastErrors runs worker over feed and waits for it to finish.
//
// A trailing EndFrame is held back until the worker has heard of errors errors.
// An error travels back up the pipeline while the frame that caused it carries on
// down, and the EndFrame behind it can reach the end and finish the run before
// the error reaches the start. Holding it back keeps a test about the errors
// from depending on which of the two gets there first.
func runPastErrors(t *testing.T, worker *pipeline.Worker, feed []frames.Frame, errors int) {
	t.Helper()
	reported := make(chan struct{}, errors)
	event.On(&worker.Registry, pipeline.EventPipelineError, func(context.Context, *frames.ErrorFrame) {
		select {
		case reported <- struct{}{}:
		default:
		}
	})

	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()

	var end frames.Frame
	if n := len(feed); n > 0 {
		if _, ok := feed[n-1].(*frames.EndFrame); ok {
			feed, end = feed[:n-1], feed[n-1]
		}
	}
	for _, f := range feed {
		worker.QueueFrame(f)
	}
	if end != nil {
		for range errors {
			select {
			case <-reported:
			case <-time.After(10 * time.Second):
				t.Fatal("the errors never reached the worker")
			}
		}
		worker.QueueFrame(end)
	}

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the worker never finished")
	}
}

// countText is the number of text frames in feed.
func countText(feed []frames.Frame) int {
	n := 0
	for _, f := range feed {
		if _, ok := f.(*frames.TextFrame); ok {
			n++
		}
	}
	return n
}
