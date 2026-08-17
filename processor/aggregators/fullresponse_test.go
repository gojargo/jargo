package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// gatherCompletions runs a full-response aggregator over the frames feed queues
// and returns what it reported.
func gatherCompletions(t *testing.T, feed func(queue func(frames.Frame))) []aggregators.Completion {
	t.Helper()
	agg := aggregators.NewFullResponse("FullResponse")

	var mu sync.Mutex
	var got []aggregators.Completion
	reported := make(chan struct{}, 8)
	events.On(agg.Events(), aggregators.EventCompletion,
		func(_ context.Context, c aggregators.Completion) {
			mu.Lock()
			got = append(got, c)
			mu.Unlock()
			select {
			case reported <- struct{}{}:
			default:
			}
		})

	task, _, runDone := runProcessor(t, agg)
	feed(func(f frames.Frame) { task.QueueFrame(f) })

	select {
	case <-reported:
	case <-time.After(3 * time.Second):
		t.Fatal("no completion was ever reported")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return append([]aggregators.Completion(nil), got...)
}

func TestFullResponseReportsAWholeResponse(t *testing.T) {
	got := gatherCompletions(t, func(queue func(frames.Frame)) {
		queue(frames.NewLLMFullResponseStartFrame())
		queue(frames.NewLLMTextFrame("Hello, "))
		queue(frames.NewLLMTextFrame("world."))
		queue(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) != 1 {
		t.Fatalf("reported %d completions, want 1", len(got))
	}
	if got[0].Text != "Hello, world." {
		t.Errorf("text = %q, want the whole response", got[0].Text)
	}
	if !got[0].Completed {
		t.Error("the response finished, so it should be reported as completed")
	}
}

func TestFullResponseReportsWhatAnInterruptionCutShort(t *testing.T) {
	// The interruption is a system frame, so it overtakes anything still
	// queued. The text has to have been gathered before it arrives, which is
	// what happens in a real turn: the model streams, and the user cuts in.
	agg := aggregators.NewFullResponse("FullResponse")

	var mu sync.Mutex
	var got []aggregators.Completion
	reported := make(chan struct{}, 4)
	events.On(agg.Events(), aggregators.EventCompletion,
		func(_ context.Context, c aggregators.Completion) {
			mu.Lock()
			got = append(got, c)
			mu.Unlock()
			select {
			case reported <- struct{}{}:
			default:
			}
		})

	task, seen, runDone := runProcessor(t, agg)
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Half a "))
	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMTextFrame)
		return ok
	}, "the streamed text")
	task.QueueFrame(frames.NewInterruptionFrame())

	select {
	case <-reported:
	case <-time.After(3 * time.Second):
		t.Fatal("the interruption never reported what had been said")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 {
		t.Fatalf("reported %d completions, want 1", len(got))
	}
	if got[0].Text != "Half a " {
		t.Errorf("text = %q, want what had been said", got[0].Text)
	}
	if got[0].Completed {
		t.Error("a response cut short should not be reported as completed")
	}
}

func TestFullResponseIgnoresTextOutsideAResponse(t *testing.T) {
	// Text that belongs to no response belongs to nothing.
	got := gatherCompletions(t, func(queue func(frames.Frame)) {
		queue(frames.NewLLMTextFrame("stray"))
		queue(frames.NewLLMFullResponseStartFrame())
		queue(frames.NewLLMTextFrame("kept"))
		queue(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) != 1 || got[0].Text != "kept" {
		t.Errorf("completions = %+v, want only the text inside the response", got)
	}
}

func TestFullResponseStartsEachResponseAfresh(t *testing.T) {
	got := gatherCompletions(t, func(queue func(frames.Frame)) {
		queue(frames.NewLLMFullResponseStartFrame())
		queue(frames.NewLLMTextFrame("first"))
		queue(frames.NewLLMFullResponseEndFrame())
		queue(frames.NewLLMFullResponseStartFrame())
		queue(frames.NewLLMTextFrame("second"))
		queue(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) != 2 {
		t.Fatalf("reported %d completions, want 2", len(got))
	}
	if got[0].Text != "first" || got[1].Text != "second" {
		t.Errorf("completions = %+v, want each response on its own", got)
	}
}

func TestFullResponseForwardsEverything(t *testing.T) {
	// It watches from beside the stream; nothing it sees is consumed.
	agg := aggregators.NewFullResponse("FullResponse")
	task, seen, runDone := runProcessor(t, agg)

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("kept"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	var starts, texts, ends int
	deadline := time.After(2 * time.Second)
	for starts == 0 || texts == 0 || ends == 0 {
		select {
		case f := <-seen:
			switch f.(type) {
			case *frames.LLMFullResponseStartFrame:
				starts++
			case *frames.LLMTextFrame:
				texts++
			case *frames.LLMFullResponseEndFrame:
				ends++
			}
		case <-deadline:
			task.StopWhenDone()
			<-runDone
			t.Fatalf("frames were consumed: starts=%d texts=%d ends=%d", starts, texts, ends)
		}
	}

	task.StopWhenDone()
	<-runDone
}

// TestFullResponseReportsAnEmptyResponse is upstream's own case: a response
// that said nothing still completed, and is reported as such.
func TestFullResponseReportsAnEmptyResponse(t *testing.T) {
	got := gatherCompletions(t, func(queue func(frames.Frame)) {
		queue(frames.NewLLMFullResponseStartFrame())
		queue(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) != 1 {
		t.Fatalf("reported %d completions, want 1", len(got))
	}
	if got[0].Text != "" {
		t.Errorf("text = %q, want nothing said", got[0].Text)
	}
	if !got[0].Completed {
		t.Error("an empty response still completed")
	}
}

// TestFullResponseReportsBothSidesOfAnInterruption is upstream's own case: the
// response the interruption cut short, then the one that replaced it.
func TestFullResponseReportsBothSidesOfAnInterruption(t *testing.T) {
	agg := aggregators.NewFullResponse("FullResponse")

	var mu sync.Mutex
	var got []aggregators.Completion
	reported := make(chan struct{}, 8)
	events.On(agg.Events(), aggregators.EventCompletion,
		func(_ context.Context, c aggregators.Completion) {
			mu.Lock()
			got = append(got, c)
			mu.Unlock()
			reported <- struct{}{}
		})

	task, seen, runDone := runProcessor(t, agg)
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	// The interruption is a system frame and would otherwise overtake the text.
	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMTextFrame)
		return ok
	}, "the streamed text")
	task.QueueFrame(frames.NewInterruptionFrame())
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("there!"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	for range 2 {
		select {
		case <-reported:
		case <-time.After(3 * time.Second):
			t.Fatal("both responses were not reported")
		}
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	want := []aggregators.Completion{
		{Text: "Hello ", Completed: false},
		{Text: "Hello there!", Completed: true},
	}
	if len(got) != len(want) {
		t.Fatalf("completions = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("completion %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}
