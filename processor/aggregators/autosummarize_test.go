package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// Summarization wired through the aggregator pair. The summarizer itself is
// covered in summarizer_test.go; what is under test here is that WithSummarization
// attaches one to the assistant half and that the request it raises reaches the
// LLM, which sits upstream of that half.

// summarizingPair runs an assistant aggregator configured to summarize, and
// collects the summary requests it puts to the model.
func summarizingPair(
	t *testing.T, convo *frames.LLMContext, cfg frames.AutoSummarizationConfig,
) (*aggregators.Pair, *pipeline.Worker, func() []*frames.LLMContextSummaryRequestFrame, func()) {
	t.Helper()

	var (
		mu  sync.Mutex
		got []*frames.LLMContextSummaryRequestFrame
	)
	pair := aggregators.New(convo, aggregators.WithSummarization(cfg))
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.LLMContextSummaryRequestFrame); ok {
			mu.Lock()
			got = append(got, fr)
			mu.Unlock()
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	requests := func() []*frames.LLMContextSummaryRequestFrame {
		mu.Lock()
		defer mu.Unlock()
		return append([]*frames.LLMContextSummaryRequestFrame(nil), got...)
	}
	stop := func() {
		t.Helper()
		task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("task did not finish")
		}
	}
	return pair, task, requests, stop
}

// TestSummarizationRequestReachesTheModel checks the wiring end to end: a
// conversation that has outgrown its budget raises a request, and the request
// travels upstream, which is where the LLM service sits relative to the
// assistant half. Without that, summarization is configured and never happens.
func TestSummarizationRequestReachesTheModel(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "You are helpful."}})
	cfg := frames.AutoSummarizationConfig{MaxContextTokens: new(50)}

	_, task, requests, stop := summarizingPair(t, convo, cfg)
	defer stop()

	addUserMessages(convo, 10, "This is a test message that adds tokens to the context.")
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())

	if !waitFor(3*time.Second, func() bool { return len(requests()) == 1 }) {
		t.Fatalf("the model was asked to summarize %d times, want 1", len(requests()))
	}
	if got := requests()[0].Context; got != convo {
		t.Error("the request does not carry the conversation it is for")
	}
}

// TestSummarizerIsReachableForHandlers checks the summarizer the option built is
// exposed and is the live one, which is the reason the accessor exists: a caller
// attaches to the events it raises.
func TestSummarizerIsReachableForHandlers(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "You are helpful."}})
	cfg := frames.AutoSummarizationConfig{MaxContextTokens: new(50)}

	pair, task, _, stop := summarizingPair(t, convo, cfg)
	defer stop()

	s := pair.Assistant().Summarizer()
	if s == nil {
		t.Fatal("Summarizer() is nil, want the one WithSummarization built")
	}

	seen := make(chan struct{}, 1)
	s.Add(aggregators.EventRequestSummarization, func(context.Context, any, ...any) {
		select {
		case seen <- struct{}{}:
		default:
		}
	})

	addUserMessages(convo, 10, "This is a test message that adds tokens to the context.")
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())

	select {
	case <-seen:
	case <-time.After(3 * time.Second):
		t.Error("a handler attached through Summarizer() never fired, so it is " +
			"not the summarizer the aggregator is using")
	}
}

// TestWithoutTheOptionSummarizationOnlyHappensOnDemand checks what the option
// actually decides. A summarizer is always there, so a pushed
// LLMSummarizeContextFrame compresses the conversation whatever the
// configuration; the option decides only whether the thresholds trigger one on
// their own.
func TestWithoutTheOptionSummarizationOnlyHappensOnDemand(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "You are helpful."}})

	var (
		mu  sync.Mutex
		got int
	)
	pair := aggregators.New(convo) // no WithSummarization
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMContextSummaryRequestFrame); ok {
			mu.Lock()
			got++
			mu.Unlock()
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("task did not finish")
		}
	}()

	count := func() int {
		mu.Lock()
		defer mu.Unlock()
		return got
	}

	// Far past any threshold: without the option, nothing is triggered by it.
	addUserMessages(convo, 50, "This is a test message that adds tokens to the context.")
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	time.Sleep(300 * time.Millisecond)
	if n := count(); n != 0 {
		t.Errorf("the model was asked to summarize %d times, want 0 without the option", n)
	}

	// Asked outright, it summarizes anyway.
	task.QueueFrame(frames.NewLLMSummarizeContextFrame())
	if !waitFor(3*time.Second, func() bool { return count() == 1 }) {
		t.Errorf("the model was asked to summarize %d times, want 1: an outright "+
			"request is acted on whatever the configuration", count())
	}
}
