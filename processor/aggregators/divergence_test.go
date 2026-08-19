package aggregators_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for what does and does not become part of the user's turn, and for the
// spacing the joined transcript ends up with.

func isContextFrame(f frames.Frame) bool { _, ok := f.(*frames.LLMContextFrame); return ok }

// finalTranscript builds a transcription that ends the user's turn.
func finalTranscript(text string) *frames.TranscriptionFrame {
	tf := frames.NewTranscriptionFrame(text, "u", "ts")
	tf.Finalized = true
	return tf
}

// TestWhitespaceTranscriptIsNotATurn checks that a final transcript of nothing
// but whitespace neither becomes a user message nor runs the model. A service
// reporting one is reporting silence, not speech.
func TestWhitespaceTranscriptIsNotATurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	task.QueueFrame(finalTranscript("   "))

	if sawFrame(seen, isContextFrame) {
		t.Error("a whitespace-only transcript ran the model")
	}
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("context messages = %+v, want none", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

// TestTranslationIsNotAggregated checks that a translation is consumed and never
// folded into the user's message. A provider that transcribes and translates
// reports both, and only the transcription is what the user actually said.
func TestTranslationIsNotAggregated(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	task.QueueFrame(frames.NewTranslationFrame("bonjour", "u", "ts"))
	task.QueueFrame(finalTranscript("hello"))

	awaitFrame(t, seen, isContextFrame, "the context frame")
	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "hello" {
		t.Fatalf("context messages = %+v, want one user 'hello'", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

// TestTranslationIsNotForwarded checks that the translation is consumed rather
// than traveling on. What is downstream is the model, which is given the
// conversation rather than the transcripts it was built from.
func TestTranslationIsNotForwarded(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	task.QueueFrame(frames.NewTranslationFrame("bonjour", "u", "ts"))

	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.TranslationFrame)
		return ok
	}) {
		t.Error("the translation traveled on downstream")
	}

	task.StopWhenDone()
	<-runDone
}

// TestTranscriptKeepsItsOwnSpacing checks that segments carrying their own
// spacing are joined as they arrive, rather than having another space inserted
// between them.
func TestTranscriptKeepsItsOwnSpacing(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	for _, part := range []string{"Hello,", " world."} {
		tf := frames.NewTranscriptionFrame(part, "u", "ts")
		tf.IncludesInterFrameSpaces = true
		task.QueueFrame(tf)
	}
	final := finalTranscript(" Goodbye.")
	final.IncludesInterFrameSpaces = true
	task.QueueFrame(final)

	awaitFrame(t, seen, isContextFrame, "the context frame")
	msgs := convo.Messages()
	if want := "Hello, world. Goodbye."; len(msgs) != 1 || msgs[0].Text != want {
		t.Fatalf("context messages = %+v, want one saying %q", msgs, want)
	}

	task.StopWhenDone()
	<-runDone
}

// TestTranscriptWithoutSpacingIsSpaced checks the other kind: segments that do
// not carry their own spacing are separated when they are joined.
func TestTranscriptWithoutSpacingIsSpaced(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	for _, part := range []string{"Hello", "world"} {
		task.QueueFrame(frames.NewTranscriptionFrame(part, "u", "ts"))
	}
	task.QueueFrame(finalTranscript("again"))

	awaitFrame(t, seen, isContextFrame, "the context frame")
	msgs := convo.Messages()
	if want := "Hello world again"; len(msgs) != 1 || msgs[0].Text != want {
		t.Fatalf("context messages = %+v, want one saying %q", msgs, want)
	}

	task.StopWhenDone()
	<-runDone
}

// TestAssistantRunsTheModelOnRequest checks that an explicit request to run the
// model, arriving at the assistant half from upstream, is turned into a run.
//
// It is the path the LLM service's own re-prompt takes after an incomplete user
// turn: the service pushes the request downstream, so it lands here, and the run
// it asks for has to travel back the other way to reach the service.
func TestAssistantRunsTheModelOnRequest(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	seen := make(chan frames.Frame, 32)
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMMessagesAppendFrame(
		[]frames.Message{{Role: frames.RoleDeveloper, Text: "Go on."}}))
	task.QueueFrame(frames.NewLLMRunFrame())

	awaitFrame(t, seen, isContextFrame, "the run the re-prompt asked for")

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "Go on." {
		t.Errorf("context messages = %+v, want the re-prompt alone", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

// TestAssistantConsumesTheRunRequest checks that the request itself is not
// forwarded. Running the model is what it asked for, and nothing beyond the
// assistant half acts on it.
func TestAssistantConsumesTheRunRequest(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	seen := make(chan frames.Frame, 32)
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
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

	task.QueueFrame(frames.NewLLMRunFrame())

	if sawFrame(seen, func(f frames.Frame) bool { _, ok := f.(*frames.LLMRunFrame); return ok }) {
		t.Error("the run request traveled on downstream")
	}

	task.StopWhenDone()
	<-runDone
}

// TestToolResultBeforeItsCallStartsIsIgnored checks that a result arriving
// before the frame that starts its call is not applied. The call was announced
// but has not begun, so there is no tool-use block in the conversation for the
// result to answer, and writing one anyway would leave the turn unbalanced.
func TestToolResultBeforeItsCallStartsIsIgnored(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{}`)}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame(calls),
		// No FunctionCallInProgressFrame: the call never started.
		frames.NewFunctionCallResultFrame("c1", "get_weather", nil, "sunny"),
	)

	for _, m := range convo.Messages() {
		if len(m.ToolResults) > 0 {
			t.Fatalf("a tool result was written with no call to answer: %+v", convo.Messages())
		}
	}
}
