package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// The failures a tool handler returns in these tests.
var (
	errKaboom = errors.New("kaboom")
	errSecret = errors.New("connection refused: token=sk-secret")
)

// callWatch collects what a tool call produced on its way through the pipeline.
type callWatch struct {
	mu      sync.Mutex
	results []*frames.FunctionCallResultFrame
	cancels []*frames.FunctionCallCancelFrame
	errors  []*frames.ErrorFrame
	settled chan struct{}
}

func newCallWatch() *callWatch {
	return &callWatch{settled: make(chan struct{}, 8)}
}

func (w *callWatch) see(f frames.Frame) {
	w.mu.Lock()
	defer w.mu.Unlock()
	switch fr := f.(type) {
	case *frames.FunctionCallResultFrame:
		w.results = append(w.results, fr)
		select {
		case w.settled <- struct{}{}:
		default:
		}
	case *frames.FunctionCallCancelFrame:
		w.cancels = append(w.cancels, fr)
		select {
		case w.settled <- struct{}{}:
		default:
		}
	}
}

func (w *callWatch) wait(t *testing.T) {
	t.Helper()
	select {
	case <-w.settled:
	case <-time.After(3 * time.Second):
		t.Fatal("the call never settled")
	}
}

// runFailingCall runs one tool call whose handler does what handler says, and
// returns what the pipeline saw.
func runFailingCall(t *testing.T, handler llm.FunctionCallHandler) (*callWatch, *frames.LLMContext) {
	t.Helper()
	gen := &onceToolGen{}
	svc := llm.New("FakeToolLLM", gen)
	svc.RegisterFunction("get_weather", handler)

	watch := newCallWatch()
	probe := newProbe(watch.see)
	convo := toolConvo("get_weather")
	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(svc, probe, pair.Assistant()), pipeline.WorkerConfig{})
	// An error travels upstream, so it is collected where application code sees
	// it rather than at a probe below the service.
	events.On(&task.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		watch.mu.Lock()
		watch.errors = append(watch.errors, ef)
		watch.mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	watch.wait(t)
	// Let the error frame, which travels upstream, catch up with the result.
	time.Sleep(100 * time.Millisecond)

	task.StopWhenDone()
	<-runDone
	return watch, convo
}

func TestARaisingHandlerSettlesItsCall(t *testing.T) {
	watch, _ := runFailingCall(t, func(context.Context, llm.FunctionCallParams) error {
		return errKaboom
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	if len(watch.results) != 1 {
		t.Fatalf("the call produced %d results, want 1: a failed handler still settles it", len(watch.results))
	}
	want := "The function `get_weather` failed and returned no result."
	if watch.results[0].Result != want {
		t.Errorf("result = %q, want %q", watch.results[0].Result, want)
	}
}

func TestARaisingHandlerStillReportsTheErrorUpstream(t *testing.T) {
	watch, _ := runFailingCall(t, func(context.Context, llm.FunctionCallParams) error {
		return errKaboom
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	if len(watch.errors) != 1 {
		t.Fatalf("reported %d errors, want 1", len(watch.errors))
	}
	ef := watch.errors[0]
	if !strings.Contains(ef.Error, "kaboom") {
		t.Errorf("error message %q does not carry what the handler said", ef.Error)
	}
	if !errors.Is(ef.Err, errKaboom) {
		t.Errorf("error carries %v, want the handler's own failure", ef.Err)
	}
	if ef.Fatal {
		t.Error("a failing tool handler is not fatal to the pipeline")
	}
}

func TestAToolFailureDoesNotWriteOffTheService(t *testing.T) {
	// The failure came from the application's handler, so it says nothing about
	// whether the service can go on doing its job.
	watch, _ := runFailingCall(t, func(context.Context, llm.FunctionCallParams) error {
		return errKaboom
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	ef := watch.errors[0]
	if ef.Category != errs.Application {
		t.Errorf("category: got %q, want %q", ef.Category, errs.Application)
	}
	src, ok := ef.Source.(interface{ Usable() bool })
	if !ok {
		t.Fatalf("error names %T, want the service", ef.Source)
	}
	if !src.Usable() {
		t.Error("a failing tool handler cost the service its usability")
	}
}

func TestTheFailureIsKeptOutOfTheConversation(t *testing.T) {
	// A failure reaches the user through the model, so what the model is told
	// names the function and nothing else.
	watch, _ := runFailingCall(t, func(context.Context, llm.FunctionCallParams) error {
		return errSecret
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	result := watch.results[0].Result
	if strings.Contains(result, "sk-secret") || strings.Contains(result, "connection refused") {
		t.Errorf("result %q carries the failure the model should not repeat", result)
	}
}

func TestAHandlerThatReportsThenFailsKeepsItsResult(t *testing.T) {
	watch, _ := runFailingCall(t, func(ctx context.Context, p llm.FunctionCallParams) error {
		if err := p.Result(ctx, "sunny, 20C", nil); err != nil {
			return err
		}
		return errKaboom
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	if len(watch.results) != 1 {
		t.Fatalf("the call produced %d results, want 1", len(watch.results))
	}
	if watch.results[0].Result != "sunny, 20C" {
		t.Errorf("result = %q, want the one the handler reported", watch.results[0].Result)
	}
}

func TestARaisingHandlerDoesNotAlsoTimeOut(t *testing.T) {
	// The deadline is disarmed on the way out, so the call settles once.
	gen := &onceToolGen{}
	svc := llm.New("FakeToolLLM", gen, llm.WithFunctionCallTimeout(100*time.Millisecond))
	svc.RegisterFunction("get_weather", func(context.Context, llm.FunctionCallParams) error {
		return errKaboom
	})

	watch := newCallWatch()
	probe := newProbe(watch.see)
	convo := toolConvo("get_weather")
	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(svc, probe, pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	watch.wait(t)
	// Well past the deadline the failed call would otherwise have hit.
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	watch.mu.Lock()
	defer watch.mu.Unlock()
	if len(watch.cancels) != 0 {
		t.Errorf("the call was canceled %d times as well as failing", len(watch.cancels))
	}
}

func TestASecondResultIsRejected(t *testing.T) {
	// A handler can outlive its call and still hold the callback, so a late
	// result is rejected rather than broadcast for the aggregator to drop.
	watch, _ := runFailingCall(t, func(ctx context.Context, p llm.FunctionCallParams) error {
		if err := p.Result(ctx, "first", nil); err != nil {
			return err
		}
		return p.Result(ctx, "second", nil)
	})

	watch.mu.Lock()
	defer watch.mu.Unlock()
	if len(watch.results) != 1 {
		t.Fatalf("the call produced %d results, want 1", len(watch.results))
	}
	if watch.results[0].Result != "first" {
		t.Errorf("result = %q, want the first one reported", watch.results[0].Result)
	}
}

func TestAnAsyncCallCanceledByItsDeadlineIsSettledInTheConversation(t *testing.T) {
	// A call the model does not wait on is settled the same channel its results
	// would have arrived on, so the conversation is not left showing it running.
	gen := &onceToolGen{}
	svc := llm.New("FakeToolLLM", gen, llm.WithFunctionCallTimeout(50*time.Millisecond))
	release := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return nil
	}, llm.WithCancelOnInterruption(false))

	watch := newCallWatch()
	probe := newProbe(watch.see)
	convo := toolConvo("get_weather")
	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(svc, probe, pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	watch.wait(t)

	settled := waitForContext(convo, func(msgs []frames.Message) bool {
		for _, m := range msgs {
			parsed, ok := frames.ParseAsyncToolMessage(m)
			if ok && parsed.Kind == frames.AsyncToolFinal &&
				strings.Contains(parsed.Result, "CANCELLED") { //nolint:misspell // the protocol's own spelling
				return true
			}
		}
		return false
	})
	close(release)
	task.StopWhenDone()
	<-runDone

	if !settled {
		t.Errorf("messages = %+v, want the canceled call settled", convo.Messages())
	}
}
