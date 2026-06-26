package aggregators_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// fakeSummarizer signals when it is called and returns a fixed summary.
type fakeSummarizer struct {
	called  chan []frames.Message
	summary string
}

func (f *fakeSummarizer) Summarize(_ context.Context, _ string, dropped []frames.Message) (string, error) {
	select {
	case f.called <- dropped:
	default:
	}
	return f.summary, nil
}

func TestAssistantAggregatorSummarizesWhenContextGrows(t *testing.T) {
	convo := frames.NewLLMContext("system prompt")
	// Prior history so there is something to compact once the turn commits.
	convo.AddUserMessage("hello there, how are you doing today?")
	convo.AddAssistantMessage("I am doing well, thank you for asking.")
	convo.AddUserMessage("can you tell me about the weather, please?")
	convo.AddAssistantMessage("It is sunny and quite warm outside right now.")

	fs := &fakeSummarizer{called: make(chan []frames.Message, 1), summary: "user greeted and asked about weather"}
	pair := aggregators.New(convo, aggregators.WithSummarization(aggregators.SummarizeConfig{
		Summarizer:         fs,
		TriggerTokens:      5, // tiny, so the existing context is already over it
		KeepRecentMessages: 2,
	}))

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Drive one assistant turn; committing it triggers background summarization.
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Sure, here is the latest."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case dropped := <-fs.called:
		if len(dropped) == 0 {
			t.Fatal("summarizer called with no dropped messages")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("summarizer was not invoked after the context grew past the trigger")
	}

	// The summary is applied right after the summarizer returns; poll for it.
	if !waitFor(2*time.Second, func() bool { return convo.Summary() != "" }) {
		t.Fatal("summary was not applied to the context")
	}
	if convo.Summary() != fs.summary {
		t.Fatalf("Summary() = %q, want %q", convo.Summary(), fs.summary)
	}
	if msgs := convo.Messages(); len(msgs) != 3 {
		t.Fatalf("remaining messages = %d, want 3 (KeepRecent=2 plus the new turn)", len(msgs))
	}

	task.StopWhenDone()
	<-runDone
}

func TestAssistantAggregatorNoSummarizationWhenDisabled(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("a fairly long user message that carries plenty of content")
	convo.AddAssistantMessage("a fairly long assistant reply that also carries content")
	pair := aggregators.New(convo) // no WithSummarization

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("reply"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone

	if convo.Summary() != "" {
		t.Fatalf("Summary() = %q, want empty when summarization is disabled", convo.Summary())
	}
}

// waitFor polls cond until it returns true or the timeout elapses.
func waitFor(timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond()
}
