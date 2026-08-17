package aggregators_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the frames that mutate the shared LLM context. Each must do two
// things: apply the change to the context, and reach downstream. The second half
// is what a realtime (speech-to-speech) service depends on — it generates
// continuously and never re-reads the context, so a change made only on the
// context object would never reach the model.

// runAggregator starts a task around the user aggregator and collects the frames
// that reach the end of the pipeline.
func runAggregator(t *testing.T, convo *frames.LLMContext) (*pipeline.Worker, chan frames.Frame, chan error) {
	t.Helper()
	pair := aggregators.New(convo)
	seen := make(chan frames.Frame, 32)
	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, seen, runDone
}

// awaitFrame waits for a frame the predicate accepts.
// sawFrame reports whether a matching frame reached the end of the pipeline
// within a short grace period. It is the negative counterpart of awaitFrame: a
// frame that should have been consumed must not arrive.
func sawFrame(seen chan frames.Frame, match func(frames.Frame) bool) bool {
	deadline := time.After(300 * time.Millisecond)
	for {
		select {
		case f := <-seen:
			if match(f) {
				return true
			}
		case <-deadline:
			return false
		}
	}
}

func awaitFrame(t *testing.T, seen chan frames.Frame, match func(frames.Frame) bool, what string) frames.Frame {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-seen:
			if match(f) {
				return f
			}
		case <-deadline:
			t.Fatalf("%s never reached the end of the pipeline", what)
			return nil
		}
	}
}

// TestSetToolsFrameAppliesAndForwards is the regression test for the defect that
// motivated the frame: changing tools must both update the context and travel
// downstream, or a continuously running realtime service keeps the old toolset.
func TestSetToolsFrameAppliesAndForwards(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	tools := []frames.Tool{{
		Name:        "get_weather",
		Description: "look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}}
	task.QueueFrame(frames.NewLLMSetToolsFrame(tools))

	got := awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMSetToolsFrame)
		return ok
	}, "LLMSetToolsFrame")

	if fr, ok := got.(*frames.LLMSetToolsFrame); !ok || len(fr.Tools) != 1 {
		t.Errorf("forwarded frame = %v, want it to carry the new toolset", got)
	}
	if ctxTools := convo.Tools(); len(ctxTools) != 1 || ctxTools[0].Name != "get_weather" {
		t.Errorf("context tools = %+v, want the frame applied to the shared context", ctxTools)
	}

	task.StopWhenDone()
	<-runDone
}

// TestSetToolChoiceFrameIsAppliedAndConsumed covers the tool-choice counterpart.
// Unlike the toolset, it is consumed: a speech-to-speech service is told which
// tools exist because it cannot pick that up on a next run, but the choice of
// how to use them is read from the shared conversation like any other setting.
func TestSetToolChoiceFrameIsAppliedAndConsumed(t *testing.T) {
	convo := frames.NewLLMContext("system")
	if got := convo.ToolChoice(); got != frames.ToolChoiceAuto {
		t.Errorf("ToolChoice() = %q, want auto by default", got)
	}

	task, seen, runDone := runAggregator(t, convo)
	task.QueueFrame(frames.NewLLMSetToolChoiceFrame(frames.ToolChoiceRequired))

	if !waitFor(2*time.Second, func() bool { return convo.ToolChoice() == frames.ToolChoiceRequired }) {
		t.Errorf("ToolChoice() = %q, want required", convo.ToolChoice())
	}
	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMSetToolChoiceFrame)
		return ok
	}) {
		t.Error("the tool-choice frame was forwarded, want it consumed once applied")
	}

	task.StopWhenDone()
	<-runDone
}

// TestMessagesUpdateFrameReplacesConversation checks the update frame replaces
// the messages rather than appending, leaves the system prompt alone, and is
// consumed: both halves of the pair share the conversation, so a frame that
// reached one of them has been applied and must not travel on to the other.
func TestMessagesUpdateFrameReplacesConversation(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("old one")
	convo.AddAssistantMessage("old two")

	task, seen, runDone := runAggregator(t, convo)
	task.QueueFrame(frames.NewLLMMessagesUpdateFrame([]frames.Message{
		{Role: frames.RoleUser, Text: "restored"},
	}))

	if !waitFor(2*time.Second, func() bool { return len(convo.Messages()) == 1 }) {
		t.Fatalf("the update was never applied; messages = %+v", convo.Messages())
	}
	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMMessagesUpdateFrame)
		return ok
	}) {
		t.Error("the update frame was forwarded, want it consumed once applied")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "restored" {
		t.Errorf("messages = %+v, want the conversation replaced, not appended", msgs)
	}
	if convo.System() != "system" {
		t.Errorf("System() = %q, want the system prompt untouched", convo.System())
	}

	task.StopWhenDone()
	<-runDone
}

// TestMessagesUpdateFrameRunLLM checks the update can trigger a generation on the
// replaced conversation.
func TestMessagesUpdateFrameRunLLM(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	update := frames.NewLLMMessagesUpdateFrame([]frames.Message{{Role: frames.RoleUser, Text: "go"}})
	update.RunLLM = true
	task.QueueFrame(update)

	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextFrame)
		return ok
	}, "LLMContextFrame")

	task.StopWhenDone()
	<-runDone
}

// keepUserMessages is the transform upstream's tests use: it drops everything
// the user did not say.
func keepUserMessages(msgs []frames.Message) []frames.Message {
	var kept []frames.Message
	for _, m := range msgs {
		if m.Role == frames.RoleUser {
			kept = append(kept, m)
		}
	}
	return kept
}

// TestMessagesTransformFrameRewritesTheConversation checks the transform is
// applied to what the conversation currently holds, which is what a wholesale
// replacement cannot express.
func TestMessagesTransformFrameRewritesTheConversation(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleUser, Text: "Hello"},
		{Role: frames.RoleAssistant, Text: "Hi there!"},
		{Role: frames.RoleUser, Text: "How are you?"},
	})
	task, seen, runDone := runAggregator(t, convo)

	task.QueueFrame(frames.NewLLMMessagesTransformFrame(keepUserMessages))

	// Nothing asked for a run, so the rewrite is all that should happen.
	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextFrame)
		return ok
	}) {
		t.Error("the transform ran the model without being asked to")
	}
	msgs := convo.Messages()
	if len(msgs) != 2 || msgs[0].Text != "Hello" || msgs[1].Text != "How are you?" {
		t.Errorf("messages = %+v, want the user's two", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

// TestMessagesTransformFrameRunLLM checks the transform can trigger a generation
// on the rewritten conversation.
func TestMessagesTransformFrameRunLLM(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetMessages([]frames.Message{{Role: frames.RoleUser, Text: "Hello"}})
	task, seen, runDone := runAggregator(t, convo)

	transform := frames.NewLLMMessagesTransformFrame(func(msgs []frames.Message) []frames.Message {
		out := make([]frames.Message, len(msgs))
		for i, m := range msgs {
			m.Text = strings.ToUpper(m.Text)
			out[i] = m
		}
		return out
	})
	transform.RunLLM = true
	task.QueueFrame(transform)

	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextFrame)
		return ok
	}, "LLMContextFrame")

	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "HELLO" {
		t.Errorf("messages = %+v, want the rewritten one", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

// TestMessagesTransformFrameIsConsumed checks the frame does not travel on. Both
// halves share one conversation, so a frame that reached either has been applied
// and must not be applied again by the other.
func TestMessagesTransformFrameIsConsumed(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetMessages([]frames.Message{{Role: frames.RoleUser, Text: "Hello"}})
	task, seen, runDone := runAggregator(t, convo)

	task.QueueFrame(frames.NewLLMMessagesTransformFrame(keepUserMessages))

	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMMessagesTransformFrame)
		return ok
	}) {
		t.Error("the transform frame traveled on after being applied")
	}

	task.StopWhenDone()
	<-runDone
}

// TestAssistantMessagesTransformFrame checks the assistant half applies the
// rewrite too, and runs the model upstream when asked, which is the direction
// the LLM service lies in from there.
func TestAssistantMessagesTransformFrame(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleUser, Text: "Hello"},
		{Role: frames.RoleAssistant, Text: "Hi there!"},
		{Role: frames.RoleUser, Text: "How are you?"},
	})

	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	seen := make(chan frames.Frame, 32)
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	transform := frames.NewLLMMessagesTransformFrame(keepUserMessages)
	transform.RunLLM = true
	task.QueueFrame(transform)

	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextFrame)
		return ok
	}, "LLMContextFrame")

	if msgs := convo.Messages(); len(msgs) != 2 {
		t.Errorf("messages = %+v, want the user's two", msgs)
	}

	task.StopWhenDone()
	<-runDone
}
