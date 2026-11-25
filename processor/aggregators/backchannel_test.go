package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// runThrough runs feed through p, pausing after each frame so system frames do
// not overtake the data frames around them, and returns what reached the end.
func runThrough(t *testing.T, p processor.Processor, feed ...frames.Frame) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var down []frames.Frame
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range feed {
		task.QueueFrame(f)
		time.Sleep(20 * time.Millisecond)
	}
	task.StopWhenDone()
	<-done
	settle()
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), down...)
}

func hasBackchannel(down []frames.Frame) bool {
	for _, f := range down {
		if _, ok := f.(*frames.UserBackchannelFrame); ok {
			return true
		}
	}
	return false
}

// TestUserBackchannel checks a user's backchannel is reported and passed on,
// and stays out of the conversation.
func TestUserBackchannel(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)
	var mu sync.Mutex
	var got []aggregators.UserBackchannel
	event.On(pair.User().Events(), aggregators.EventUserBackchannel,
		func(_ context.Context, b aggregators.UserBackchannel) {
			mu.Lock()
			got = append(got, b)
			mu.Unlock()
		})

	down := runThrough(t, pair.User(), frames.NewUserBackchannelFrame("mhm", "user", "now"))

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || got[0] != (aggregators.UserBackchannel{Content: "mhm", Timestamp: "now", UserID: "user"}) {
		t.Errorf("backchannels = %+v, want the one the user said", got)
	}
	if !hasBackchannel(down) {
		t.Error("the backchannel was not passed on")
	}
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("messages = %+v, want none", msgs)
	}
}

// TestUserBackchannelIsSuppressedWhileTheUserIsMuted checks a backchannel said
// while the user is muted is dropped, and not reported.
func TestUserBackchannelIsSuppressedWhileTheUserIsMuted(t *testing.T) {
	pair := aggregators.New(frames.NewLLMContext(""),
		aggregators.WithMuteStrategies(turns.NewFirstSpeechUserMute()))
	var mu sync.Mutex
	reported := 0
	event.On(pair.User().Events(), aggregators.EventUserBackchannel,
		func(context.Context, aggregators.UserBackchannel) {
			mu.Lock()
			reported++
			mu.Unlock()
		})

	// The bot is speaking, so the user is muted.
	down := runThrough(t, pair.User(), frames.NewBotStartedSpeakingFrame(),
		frames.NewUserBackchannelFrame("mhm", "user", "now"))

	if hasBackchannel(down) {
		t.Error("the backchannel was passed on while the user was muted")
	}
	mu.Lock()
	defer mu.Unlock()
	if reported != 0 {
		t.Errorf("reported %d backchannels, want none", reported)
	}
}

// TestAssistantBackchannel checks the assistant's backchannel is reported once,
// by the segment the TTS service will speak, and stays out of the conversation.
func TestAssistantBackchannel(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)
	var mu sync.Mutex
	var got []string
	event.On(pair.Assistant().Events(), aggregators.EventAssistantBackchannel,
		func(_ context.Context, b aggregators.AssistantBackchannel) {
			mu.Lock()
			got = append(got, b.Content)
			mu.Unlock()
		})

	// The frames a TTS service pushes for a BotBackchannelFrame, and then for
	// a sentence of a reply.
	backchannel := frames.NewAggregatedTextFrame("Mm-hmm.", frames.TextTypeBackchannel)
	backchannel.WillBeSpoken = true
	backchannel.AppendToContext = false
	spoken := frames.NewTTSTextFrame("Mm-hmm.", frames.TextTypeBackchannel)
	spoken.SegmentID = backchannel.ID()
	spoken.AppendToContext = false
	sentence := frames.NewAggregatedTextFrame("Sounds fun.", frames.TextTypeSentence)
	sentence.WillBeSpoken = true
	sentence.AppendToContext = false
	runThrough(t, pair.Assistant(), backchannel, spoken, sentence)

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || got[0] != "Mm-hmm." {
		t.Errorf("backchannels = %q, want the one segment", got)
	}
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("messages = %+v, want none", msgs)
	}
}
