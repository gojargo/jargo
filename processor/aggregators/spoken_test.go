package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// runAssistant drives frames through an assistant aggregator over a real
// pipeline and returns the conversation once the task has finished with them.
func runAssistant(t *testing.T, convo *frames.LLMContext, queue func(task *pipeline.Task)) {
	t.Helper()
	pair := aggregators.New(convo)
	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	queue(task)
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// TestAssistantAggregatorRecordsWordsAsTheyAreSpoken covers the context a
// word-timing TTS builds. Its words reach the aggregator in step with playback,
// so the conversation is kept current with what has actually been heard rather
// than with what the model produced. The words are one message that grows, not
// one message each.
func TestAssistantAggregatorRecordsWordsAsTheyAreSpoken(t *testing.T) {
	convo := frames.NewLLMContext("system")

	runAssistant(t, convo, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		for _, word := range []string{"Let", "me", "check", "that"} {
			task.QueueFrame(frames.NewTTSTextFrame(word))
		}
	})

	msgs := convo.Messages()
	if len(msgs) != 1 {
		t.Fatalf("context messages = %+v, want one message that grew with the words", msgs)
	}
	if msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Let me check that" {
		t.Errorf("assistant message = %q (role %v), want %q", msgs[0].Text, msgs[0].Role, "Let me check that")
	}
}

// TestAssistantAggregatorRecordsTheWrittenForm covers the difference between
// what is said and what is written. A word spoken as an expansion of its written
// form (a number read out, say) has to go into the conversation as it was
// written, or the model reads back its own speech rather than its own text.
func TestAssistantAggregatorRecordsTheWrittenForm(t *testing.T) {
	convo := frames.NewLLMContext("system")

	runAssistant(t, convo, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewTTSTextFrame("Room"))
		spoken := frames.NewTTSTextFrame("twenty-three")
		spoken.RawText = "23"
		task.QueueFrame(spoken)
	})

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "Room 23" {
		t.Fatalf("context messages = %+v, want one assistant %q", msgs, "Room 23")
	}
}

// TestAssistantAggregatorIgnoresWordsNotBoundForTheContext covers speech that is
// deliberately kept out of the conversation. The flag is how a caller says this
// utterance is not part of the conversation, and the aggregator has to respect it.
func TestAssistantAggregatorIgnoresWordsNotBoundForTheContext(t *testing.T) {
	convo := frames.NewLLMContext("system")

	runAssistant(t, convo, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		for _, word := range []string{"not", "for", "the", "record"} {
			f := frames.NewTTSTextFrame(word)
			f.AppendToContext = false
			task.QueueFrame(f)
		}
	})

	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("context messages = %+v, want none: this speech was not for the conversation", msgs)
	}
}

// TestAssistantAggregatorKeepsOnlyTheWordsActuallySpoken covers a barge-in
// part-way through the bot's reply. Only the words already played have reached
// the aggregator, so what the conversation keeps is exactly what the user heard.
// Keeping the whole reply would leave the model believing it said things the
// user was never played.
func TestAssistantAggregatorKeepsOnlyTheWordsActuallySpoken(t *testing.T) {
	convo := frames.NewLLMContext("system")

	runAssistant(t, convo, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewTTSTextFrame("Let"))
		task.QueueFrame(frames.NewTTSTextFrame("me"))
		// The words have to be recorded before the barge-in arrives, which is
		// what playback pacing would otherwise ensure.
		time.Sleep(300 * time.Millisecond)
		task.QueueFrame(frames.NewInterruptionFrame())
	})

	msgs := convo.Messages()
	if len(msgs) != 1 {
		t.Fatalf("context messages = %+v, want one", msgs)
	}
	if msgs[0].Text != "Let me" {
		t.Errorf("assistant message = %q, want %q: the conversation kept words that were never spoken",
			msgs[0].Text, "Let me")
	}
}

// TestUserAggregatorDropsInputWhileMuted covers the mute strategies. While the
// bot holds the floor the user's input is dropped outright, before the turn
// controllers ever see it, so it can neither barge in nor reach the
// conversation. The state change is announced both ways so the rest of the
// pipeline can follow it.
func TestUserAggregatorDropsInputWhileMuted(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		MuteStrategies: []turns.MuteStrategy{turns.NewAlwaysUserMute()},
	}))

	var mu sync.Mutex
	var muteStarted, muteStopped int
	var transcripts []string
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.UserMuteStartedFrame:
				muteStarted++
			case *frames.UserMuteStoppedFrame:
				muteStopped++
			case *frames.TranscriptionFrame:
				transcripts = append(transcripts, fr.Text)
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The bot takes the floor, so the user is muted from here.
	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	time.Sleep(200 * time.Millisecond)

	// Finalized, so it would open and close a turn if it were not dropped.
	muted := frames.NewTranscriptionFrame("interrupting you", "user", "")
	muted.Finalized = true
	task.QueueFrame(muted)
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	got := append([]string(nil), transcripts...)
	mu.Unlock()
	if len(got) != 0 {
		t.Errorf("speech while muted got through: %q", got)
	}
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("context messages = %+v, want none: speech while muted reached the conversation", msgs)
	}

	// The bot gives the floor up, and the user is heard again.
	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(200 * time.Millisecond)
	heard := frames.NewTranscriptionFrame("now can you hear me", "user", "")
	heard.Finalized = true
	task.QueueFrame(heard)
	time.Sleep(200 * time.Millisecond)

	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(transcripts) != 1 || transcripts[0] != "now can you hear me" {
		t.Errorf("transcripts through = %q, want only the one spoken after the mute ended", transcripts)
	}
	if muteStarted != 1 || muteStopped != 1 {
		t.Errorf("announced %d mute starts and %d stops, want 1 of each", muteStarted, muteStopped)
	}
}
