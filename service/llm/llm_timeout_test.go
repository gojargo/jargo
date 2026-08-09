package llm_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// errConnection is what a provider reports when the exchange itself failed.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errConnection = errors.New("connection timed out")

// timeoutGen stands in for a provider that never answered in time.
type timeoutGen struct{}

func (timeoutGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return errConnection
}

// stalledGen is a provider whose wait for the response ran out.
type stalledGen struct{}

func (stalledGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return fmt.Errorf("%w: %w", llm.ErrCompletionTimeout, errConnection)
}

// TestACompletionTimeoutIsReportedAndAnnounced checks a generation that gave up
// waiting reports itself as a timeout rather than as a failure like any other,
// tells anything watching for one, and still brackets the response: a switcher
// reads the error frame to fail over, and the pipeline must not be left waiting
// for an end that never comes.
func TestACompletionTimeoutIsReportedAndAnnounced(t *testing.T) {
	svc := llm.New("StalledLLM", stalledGen{})

	var mu sync.Mutex
	announced := 0
	svc.OnCompletionTimeout(func(context.Context) {
		mu.Lock()
		announced++
		mu.Unlock()
	})

	var errs []string
	starts, ends := 0, 0
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedUpstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.ErrorFrame); ok {
				mu.Lock()
				errs = append(errs, fr.Error)
				mu.Unlock()
			}
		},
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch f.(type) {
			case *frames.LLMFullResponseStartFrame:
				starts++
			case *frames.LLMFullResponseEndFrame:
				ends++
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response did not complete")
	}
	// The announcement runs off the frame path, so give it a moment to land.
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if announced != 1 {
		t.Errorf("timeout announcements = %d, want one", announced)
	}
	if len(errs) != 1 || !strings.Contains(errs[0], "LLM completion timeout") {
		t.Errorf("errors = %v, want the failure reported as a timeout", errs)
	}
	if starts != 1 || ends != 1 {
		t.Errorf("response frames = %d start / %d end, want the response bracketed once", starts, ends)
	}
}

// TestAnOrdinaryFailureIsNotAnnouncedAsATimeout checks the announcement is kept
// for what it names: a provider failing for any other reason is reported, and
// nothing is told it timed out.
func TestAnOrdinaryFailureIsNotAnnouncedAsATimeout(t *testing.T) {
	svc := llm.New("FailingLLM", timeoutGen{})

	var mu sync.Mutex
	announced := 0
	svc.OnCompletionTimeout(func(context.Context) {
		mu.Lock()
		announced++
		mu.Unlock()
	})

	var errs []string
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedUpstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.ErrorFrame); ok {
				mu.Lock()
				errs = append(errs, fr.Error)
				mu.Unlock()
			}
		},
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response did not complete")
	}
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if announced != 0 {
		t.Errorf("timeout announcements = %d, want none", announced)
	}
	if len(errs) != 1 || strings.Contains(errs[0], "timeout") {
		t.Errorf("errors = %v, want the failure reported on its own terms", errs)
	}
}
