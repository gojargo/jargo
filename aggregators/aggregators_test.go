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

func TestUserAggregatorTurnTakingGatesOnEndOfTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurnTaking())

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

	// A finalized transcript without an end-of-turn must NOT trigger the LLM.
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)

	select {
	case <-triggered:
		t.Fatal("LLM triggered before end-of-turn")
	case <-time.After(300 * time.Millisecond):
	}

	// The end-of-turn frame, with a finalized transcript already in hand, now
	// triggers the LLM.
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("LLM not triggered after end-of-turn")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "hello there" {
		t.Fatalf("context messages = %+v, want one user 'hello there'", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

func TestAssistantAggregatorCommitsPartialOnInterruption(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("wor"))
	// Let the text be aggregated before the interruption arrives.
	time.Sleep(300 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello wor" {
		t.Fatalf("context messages = %+v, want one assistant 'Hello wor'", msgs)
	}
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
