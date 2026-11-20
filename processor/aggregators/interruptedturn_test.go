package aggregators_test

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for the assistant turn an interruption cuts short. The interruption is a
// system frame, so it overtakes the words still waiting in the aggregator's
// queue; the user heard those words, so they belong to the interrupted turn,
// which closes once they are in.
//
// Ported from upstream's assistant-aggregator suite.

// announced is what the aggregator told the pipeline about the conversation, in
// the order it reached the end: the interruption, and the frames each written
// turn is announced with.
func (h *assistantHarness) announced() []string {
	var out []string
	for _, name := range h.downstream() {
		switch name {
		case "InterruptionFrame", "LLMContextFrame", "LLMContextAssistantTimestampFrame",
			"LLMContextAssistantTurnFrame", "TTSStartedFrame":
			out = append(out, name)
		}
	}
	return out
}

// awaitAnnounced waits for the aggregator to have announced n frames.
func awaitAnnounced(t *testing.T, h *assistantHarness, n int) {
	t.Helper()
	if !waitFor(3*time.Second, func() bool { return len(h.announced()) >= n }) {
		t.Fatalf("downstream = %v, want %d frames announced", h.announced(), n)
	}
}

// interruptedTurnFrames is what an interrupted turn followed by a finished one
// announces: the interruption first, then each turn as it is written.
func interruptedTurnFrames() []string {
	return []string{
		"InterruptionFrame",
		"LLMContextFrame", "LLMContextAssistantTimestampFrame", "LLMContextAssistantTurnFrame",
		"LLMContextFrame", "LLMContextAssistantTimestampFrame", "LLMContextAssistantTurnFrame",
	}
}

func word(text string) *frames.TTSTextFrame {
	return frames.NewTTSTextFrame(text, frames.TextTypeWord)
}

// TestInterruption is upstream's test_interruption: the interrupted turn is
// written with what was said, and announced just after the interruption.
func TestInterruption(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)
	log := watchAssistant(h.assistant)

	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	settle()
	h.task.QueueFrame(frames.NewInterruptionFrame())
	// The next reply follows the user's turn, well behind the interruption.
	settle()
	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	h.task.QueueFrame(frames.NewLLMTextFrame("there!"))
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	want := interruptedTurnFrames()
	awaitAnnounced(t, h, len(want))
	settle()
	if got := h.announced(); !slices.Equal(got, want) {
		t.Errorf("downstream = %v, want %v", got, want)
	}
	if got := log.assistantStarts(); got != 2 {
		t.Errorf("assistant turn started %d times, want 2", got)
	}
	turns := log.assistantTurns()
	if len(turns) != 2 {
		t.Fatalf("assistant turns reported = %d, want 2", len(turns))
	}
	if !turns[0].Interrupted || turns[0].Content != "Hello" {
		t.Errorf("first turn = %+v, want %q interrupted", turns[0], "Hello")
	}
	if turns[1].Interrupted || turns[1].Content != "Hello there!" {
		t.Errorf("second turn = %+v, want %q finished", turns[1], "Hello there!")
	}
}

// TestInterruptionKeepsOvertakenTextInTheInterruptedTurn is upstream's
// test_interruption_keeps_overtaken_text_in_the_interrupted_turn.
//
// The output transport releases each word downstream as it is played, and the
// InterruptionFrame is a system frame, so it is dequeued ahead of words that
// arrived before it. The user heard those words: they are written to the
// interrupted turn, reported in its turn-stopped event, and kept off the front
// of the next turn.
func TestInterruptionKeepsOvertakenTextInTheInterruptedTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)
	log := watchAssistant(h.assistant)

	h.task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(), word("The"), word("pool"), word("opens"),
	})
	settle()

	// Sent back-to-back with the interruption, so both words sit in the
	// aggregator's queue when the interruption is taken past them. The frame
	// the paused aggregator holds is one that is handled the moment it is
	// released, ahead of the rest.
	h.assistant.PauseProcessingSystemFrames()
	for _, f := range []frames.Frame{
		frames.NewBotSpeakingFrame(), word("at"), word("six"), frames.NewInterruptionFrame(),
	} {
		if err := h.assistant.QueueFrame(context.Background(), f, processor.Downstream); err != nil {
			t.Fatal(err)
		}
	}
	h.assistant.ResumeProcessingSystemFrames()
	settle()

	h.task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		word("We"), word("close"), word("at"), word("nine."),
		frames.NewLLMFullResponseEndFrame(),
	})

	want := interruptedTurnFrames()
	awaitAnnounced(t, h, len(want))
	settle()
	if got := h.announced(); !slices.Equal(got, want) {
		t.Errorf("downstream = %v, want %v", got, want)
	}
	msgs := convo.Messages()
	if len(msgs) != 2 || msgs[0].Text != "The pool opens at six" || msgs[1].Text != "We close at nine." {
		t.Fatalf("conversation = %+v, want the overtaken words in the interrupted turn", msgs)
	}
	turns := log.assistantTurns()
	if len(turns) != 2 {
		t.Fatalf("assistant turns reported = %d, want 2", len(turns))
	}
	if !turns[0].Interrupted || turns[0].Content != "The pool opens at six" {
		t.Errorf("first turn = %+v, want %q interrupted", turns[0], "The pool opens at six")
	}
	if turns[1].Interrupted || turns[1].Content != "We close at nine." {
		t.Errorf("second turn = %+v, want %q finished", turns[1], "We close at nine.")
	}
}

// TestInterruptionDropsThePushWaitingForTheBotToStop is upstream's
// test_interruption_drops_the_push_waiting_for_the_bot_to_stop: the
// BotStoppedSpeakingFrame that follows an interruption runs no LLM.
func TestInterruptionDropsThePushWaitingForTheBotToStop(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(word("Checking"))
	// Asks to run while the bot speaks, so the push waits for it to stop.
	h.task.QueueFrame(appendFrame("Backend: Done.", true))
	awaitMessageCount(t, convo, 1)
	h.task.QueueFrame(frames.NewInterruptionFrame())
	h.task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	settle()

	if got := h.generations(); got != 0 {
		t.Errorf("generation ran %d times for the canceled response, want 0", got)
	}
}

// TestInterruptionWithNoOpenTurnWritesNothing is upstream's
// test_interruption_with_no_open_turn_writes_nothing: an interruption between
// turns leaves the conversation alone.
func TestInterruptionWithNoOpenTurnWritesNothing(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)
	log := watchAssistant(h.assistant)

	h.task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hello there!"),
		frames.NewLLMFullResponseEndFrame(),
	})
	awaitMessageCount(t, convo, 1)
	h.task.QueueFrame(frames.NewInterruptionFrame())

	want := []string{
		"LLMContextFrame", "LLMContextAssistantTimestampFrame", "LLMContextAssistantTurnFrame",
		"InterruptionFrame",
	}
	awaitAnnounced(t, h, len(want))
	settle()
	if got := h.announced(); !slices.Equal(got, want) {
		t.Errorf("downstream = %v, want %v", got, want)
	}
	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "Hello there!" {
		t.Errorf("conversation = %+v, want the finished turn alone", msgs)
	}
	turns := log.assistantTurns()
	if len(turns) != 1 || turns[0].Interrupted {
		t.Errorf("assistant turns = %+v, want one, not interrupted", turns)
	}
}

// TestTTSSpeakInterruptionRecordsPartialText is upstream's
// test_tts_speak_interruption_records_partial_text: interrupting an utterance
// the TTS service speaks with no response around it records the words spoken so
// far.
//
// The TTSStartedFrame opens the turn, so a mid-utterance interruption finds an
// open turn: the words spoken so far are written to the conversation and the
// turn is reported as interrupted. (No LLMAssistantPushAggregationFrame arrives:
// the interruption cancels the utterance before the TTS service commits it.)
func TestTTSSpeakInterruptionRecordsPartialText(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)
	log := watchAssistant(h.assistant)

	started := frames.NewTTSStartedFrame()
	started.AppendToContext = true
	h.task.QueueFrames([]frames.Frame{started, word("Let me"), word("check on")})
	settle()
	h.task.QueueFrame(frames.NewInterruptionFrame())

	want := []string{
		"TTSStartedFrame", "InterruptionFrame",
		"LLMContextFrame", "LLMContextAssistantTimestampFrame", "LLMContextAssistantTurnFrame",
	}
	awaitAnnounced(t, h, len(want))
	settle()
	if got := h.announced(); !slices.Equal(got, want) {
		t.Errorf("downstream = %v, want %v", got, want)
	}
	if got := log.assistantStarts(); got != 1 {
		t.Errorf("assistant turn started %d times, want once", got)
	}
	turns := log.assistantTurns()
	if len(turns) != 1 || !turns[0].Interrupted || turns[0].Content != "Let me check on" {
		t.Errorf("assistant turns = %+v, want %q interrupted", turns, "Let me check on")
	}
	msgs := convo.Messages()
	if last := msgs[len(msgs)-1]; last.Role != frames.RoleAssistant || last.Text != "Let me check on" {
		t.Errorf("last message = %+v, want the assistant's %q", last, "Let me check on")
	}
}
