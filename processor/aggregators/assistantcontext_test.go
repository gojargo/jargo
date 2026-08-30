package aggregators_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// The conversation-changing frames arriving from the assistant's side of the
// pipeline. Both halves of the pair handle these and both consume them, so each
// half needs its own coverage: the user half's is in context_update_test.go.
//
// Ported from upstream's assistant-aggregator suite.

// TestAssistantMessagesUpdateReplacesConversation is upstream's
// test_llm_messages_update on the assistant half: the conversation is replaced,
// and with no request to run the model nothing is asked of it.
func TestAssistantMessagesUpdateReplacesConversation(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "the old conversation"})
	task, generations, stop := assistantUpstream(t, convo)
	defer stop()

	update := frames.NewLLMMessagesUpdateFrame([]frames.Message{
		{Role: frames.RoleUser, Text: "Hi there!"},
	})
	task.QueueFrame(update)

	if !waitFor(3*time.Second, func() bool {
		msgs := convo.Messages()
		return len(msgs) == 1 && msgs[0].Text == "Hi there!"
	}) {
		t.Fatalf("conversation = %+v, want it replaced by the update", convo.Messages())
	}
	// Give a generation every chance to be asked for before concluding none was.
	time.Sleep(200 * time.Millisecond)
	if got := generations(); got != 0 {
		t.Errorf("the model was asked to run %d times, want 0: the update did not ask", got)
	}
}

// TestAssistantMessagesUpdateRunsTheModel is upstream's
// test_llm_messages_update_run: an update that asks for it runs the model on the
// replaced conversation.
func TestAssistantMessagesUpdateRunsTheModel(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, generations, stop := assistantUpstream(t, convo)
	defer stop()

	update := frames.NewLLMMessagesUpdateFrame([]frames.Message{
		{Role: frames.RoleUser, Text: "Hi there!"},
	})
	update.RunLLM = true
	task.QueueFrame(update)

	awaitCount(t, generations, 1, "an update asking to run the model")
	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "Hi there!" {
		t.Errorf("conversation = %+v, want it replaced by the update", msgs)
	}
}

// A marker frame is how a service tells the conversation that a turn was gated,
// carrying a symbol rather than words. There are two shapes: one written as its
// own assistant message the moment it arrives, and one held back to be flushed
// with the reply that follows, so a gated answer reads as one entry rather than
// two.

// TestMarkerIsWrittenAsItsOwnMessage checks the stand-alone shape: the marker is
// the whole assistant turn, so it is recorded at once and generation is asked
// for on the conversation it just changed.
func TestMarkerIsWrittenAsItsOwnMessage(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, generations, stop := assistantUpstream(t, convo)
	defer stop()

	task.QueueFrame(frames.NewLLMMarkerFrame("◐"))

	if !waitFor(3*time.Second, func() bool {
		msgs := convo.Messages()
		return len(msgs) == 1 && msgs[0].Role == frames.RoleAssistant && msgs[0].Text == "◐"
	}) {
		t.Fatalf("conversation = %+v, want the marker recorded as an assistant message",
			convo.Messages())
	}
	awaitCount(t, generations, 1, "a marker written on its own")
}

// TestHeldMarkerJoinsTheReplyButNotTheTranscript checks the held shape: the
// marker is flushed with the text that follows, as one message, and comes out of
// what the turn is reported as. The conversation keeps it, because the model
// reads its own earlier verdicts back; a transcript does not, because a marker
// is a signal to the machinery rather than something the bot said.
//
// This is upstream's test_turn_completion_markers_stripped_from_transcript,
// driven through the marker frame rather than the marker's text.
func TestHeldMarkerJoinsTheReplyButNotTheTranscript(t *testing.T) {
	convo := frames.NewLLMContext("system")

	var (
		mu       sync.Mutex
		reported []aggregators.AssistantTurnStopped
	)
	pair := aggregators.New(convo)
	assistant := pair.Assistant()
	events.On(assistant.Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			mu.Lock()
			reported = append(reported, m)
			mu.Unlock()
		})

	task := pipeline.NewWorker(pipeline.New(assistant), pipeline.WorkerConfig{})
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

	marker := frames.NewLLMMarkerFrame("\u2731")
	marker.AppendToContextImmediately = false

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(marker)
	task.QueueFrame(frames.NewLLMTextFrame("Hello from the bot."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	if !waitFor(3*time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) == 1
	}) {
		t.Fatal("the turn was never reported as stopped")
	}

	mu.Lock()
	said := reported[0].Content
	mu.Unlock()
	if strings.Contains(said, "\u2731") {
		t.Errorf("the turn was reported as %q, want the marker stripped from it", said)
	}
	if said != "Hello from the bot." {
		t.Errorf("the turn was reported as %q, want the reply that followed the marker", said)
	}

	msgs := convo.Messages()
	if len(msgs) != 1 {
		t.Fatalf("conversation = %+v, want the marker and the reply as one message", msgs)
	}
	if !strings.Contains(msgs[0].Text, "\u2731") {
		t.Errorf("the conversation recorded %q, want it to keep the marker the "+
			"model reads back", msgs[0].Text)
	}
}
