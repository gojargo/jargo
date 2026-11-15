package eval_test

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

// reportedConversations collects what an observer reports when its pipeline is
// done.
type reportedConversations struct {
	mu   sync.Mutex
	seen [][]map[string]any
}

func (r *reportedConversations) watch(o *eval.ConversationObserver) {
	events.On(&o.Registry, eval.EventConversationEnded, func(_ context.Context, m eval.ConversationEndedMessage) {
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
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
		},
	}))
	observer := eval.NewConversationObserver()
	var reported reportedConversations
	reported.watch(observer)
	worker := pipeline.NewWorker(pipeline.New(pair.User(), pair.Assistant()), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{observer},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()

	worker.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	transcription := frames.NewTranscriptionFrame("A table at six, please.", "", "now")
	transcription.Finalized = true
	worker.QueueFrame(transcription)
	// The stop is a system frame and would overtake the transcription still
	// queued, so the pause lets the transcription reach the turn first.
	time.Sleep(50 * time.Millisecond)
	worker.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	time.Sleep(time.Second)
	worker.QueueFrame(frames.NewLLMFullResponseStartFrame())
	worker.QueueFrame(frames.NewLLMTextFrame("Six is free."))
	worker.QueueFrame(frames.NewLLMFullResponseEndFrame())
	time.Sleep(100 * time.Millisecond)
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
