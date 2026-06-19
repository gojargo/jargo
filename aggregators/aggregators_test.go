package aggregators_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

func TestUserAggregatorTriggersLLMOnFinal(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	triggered := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// An interim transcription must not trigger the LLM.
	task.QueueFrame(frames.NewInterimTranscriptionFrame("hel", "u", "ts"))
	// A finalized transcription ends the turn and triggers the LLM.
	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)

	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("user aggregator did not emit an LLMContextFrame")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || msgs[0].Text != "hello there" {
		t.Fatalf("context messages = %+v, want one user 'hello there'", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

func TestAssistantAggregatorCollectsResponse(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("world"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello world" {
		t.Fatalf("context messages = %+v, want one assistant 'Hello world'", msgs)
	}
}
