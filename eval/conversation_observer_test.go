package eval_test

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
)

// reportedConversations collects what an observer reports when its pipeline is
// done.
type reportedConversations struct {
	mu   sync.Mutex
	seen [][]map[string]any
}

func (r *reportedConversations) watch(o *eval.ConversationObserver) {
	event.On(&o.Registry, eval.EventConversationEnded, func(_ context.Context, m eval.ConversationEndedMessage) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.seen = append(r.seen, m.Conversation)
	})
}

func (r *reportedConversations) all() [][]map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.seen
}

func TestConversationObserverReportsTheTurnsAndCallsOnceThePipelineIsDone(t *testing.T) {
	observer := eval.NewConversationObserver()
	var reported reportedConversations
	reported.watch(observer)
	push := func(f frames.Frame, dir processor.Direction) {
		observer.OnPushFrame(processor.FramePushed{Frame: f, Direction: dir, FirstPush: true})
	}

	// A turn answered early is reported once, as written when it ends.
	speculation := frames.NewLLMContextUserTurnFrame("a table at six", "t")
	speculation.Speculation = true
	push(speculation, processor.Downstream)
	push(frames.NewLLMContextUserTurnFrame("A table at six.", "t"), processor.Downstream)
	// A broadcast call arrives as two frames: the upstream one is the same call.
	downstream := frames.NewFunctionCallInProgressFrame("check", "check", json.RawMessage(`{"time":"6"}`), true, "")
	upstream := frames.NewFunctionCallInProgressFrame("check", "check", json.RawMessage(`{}`), true, "")
	downstream.SetBroadcastSiblingID(upstream.ID())
	upstream.SetBroadcastSiblingID(downstream.ID())
	push(downstream, processor.Downstream)
	push(upstream, processor.Upstream)
	push(frames.NewLLMContextAssistantTurnFrame("Let me check.", "t"), processor.Downstream)
	push(frames.NewExternalFunctionCallInProgressFrame("book", "book", json.RawMessage(`{"time":"6"}`)),
		processor.Downstream)
	push(frames.NewFunctionCallCancelFrame("book", "book"), processor.Downstream)
	push(frames.NewLLMContextAssistantTurnFrame("Six is free.", "t"), processor.Downstream)
	// Text a client sent is a user turn as well.
	push(rtvi.NewSendTextFrame("1", "Thanks.", ""), processor.Downstream)
	push(frames.NewLLMContextAssistantTurnFrame("Bye!", "t"), processor.Downstream)
	if got := reported.all(); len(got) != 0 {
		t.Fatalf("reported before the pipeline was done: %v", got)
	}

	if err := observer.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}

	want := [][]map[string]any{{
		{"role": "user", "content": "A table at six."},
		{"role": "tool_call", "name": "check", "arguments": map[string]any{"time": "6"}, cancelledKey: false},
		{"role": "tool_call", "name": "book", "arguments": map[string]any{"time": "6"}, cancelledKey: true},
		{"role": "assistant", "content": "Let me check. Six is free."},
		{"role": "user", "content": "Thanks."},
		{"role": "assistant", "content": "Bye!"},
	}}
	if got := reported.all(); !reflect.DeepEqual(got, want) {
		t.Fatalf("reported %v, want %v", got, want)
	}
}

func TestConversationObserverTheAggregatorsTurnsReachTheConversationThroughAPipeline(t *testing.T) {
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewExternalStart(turns.ExternalStartConfig{})},
			Stop:  []turns.StopStrategy{turns.NewExternalStop(turns.ExternalStopConfig{Timeout: 60 * time.Second})},
		},
	}))
	observer := eval.NewConversationObserver()
	var reported reportedConversations
	reported.watch(observer)
	turnStarted := make(chan struct{}, 1)
	event.On(pair.User().Events(), aggregators.EventUserTurnStarted,
		func(context.Context, turns.StartStrategy) { signal(turnStarted) })
	userAdded := make(chan struct{}, 1)
	event.On(pair.User().Events(), aggregators.EventUserTurnMessageAdded,
		func(context.Context, aggregators.UserTurnMessageAdded) { signal(userAdded) })
	assistantDone := make(chan struct{}, 1)
	event.On(pair.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(context.Context, aggregators.AssistantTurnStopped) { signal(assistantDone) })
	worker := pipeline.NewWorker(pipeline.New(pair.User(), pair.Assistant()), pipeline.WorkerConfig{
		Observers: []processor.Observer{observer},
	})
	started := make(chan struct{}, 1)
	event.On(worker.Events(), pipeline.EventPipelineStarted,
		func(context.Context, *frames.StartFrame) { signal(started) })
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	waitFor(t, started, "the pipeline to start")

	// The proposed start is a system frame and the transcription a data frame,
	// so they reach the aggregator on separate paths: the transcription waits
	// for the turn to open, as a service's transcript always follows its start.
	// The proposed stop is a control frame and stays behind the transcription.
	worker.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	waitFor(t, turnStarted, "the user's turn to start")
	worker.QueueFrame(frames.NewTranscriptionFrame("A table at six, please.", "", "now"))
	worker.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame())
	waitFor(t, userAdded, "the user's turn to be written to the conversation")
	worker.QueueFrame(frames.NewLLMFullResponseStartFrame())
	worker.QueueFrame(frames.NewLLMTextFrame("Six is free."))
	worker.QueueFrame(frames.NewLLMFullResponseEndFrame())
	waitFor(t, assistantDone, "the assistant's turn to end")
	worker.StopWhenDone()
	if err := <-runDone; err != nil {
		t.Fatal(err)
	}

	want := [][]map[string]any{{
		{"role": "user", "content": "A table at six, please."},
		{"role": "assistant", "content": "Six is free."},
	}}
	if got := reported.all(); !reflect.DeepEqual(got, want) {
		t.Fatalf("reported %v, want %v", got, want)
	}
}

// signal sends on ch without blocking when a signal is already pending.
func signal(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

// waitFor fails the test if ch does not fire within a generous bound.
func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}
