package aggregators_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
)

// replyFrameTypes runs a reply through an assistant aggregator and returns the
// types of what reached the end, lifecycle frames and the messages the worker
// sends its client left out.
func replyFrameTypes(t *testing.T, in ...frames.Frame) []string {
	t.Helper()
	convo := frames.NewLLMContext("")
	task, seen, runDone := runProcessor(t, aggregators.New(convo).Assistant())
	task.QueueFrames(in)
	got := drainAfterGrace(seen)
	task.StopWhenDone()
	<-runDone
	var types []string
	for _, f := range got {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame, *frames.OutputTransportMessageUrgentFrame:
			continue
		}
		types = append(types, fmt.Sprintf("%T", f))
	}
	return types
}

// Ported from upstream (TestLLMAssistantAggregator::test_simple). The reply's
// text and the response around it become conversation at the assistant
// aggregator and go no further: what is past it is the context and the turn.
func TestAssistantAggregatorConsumesTheReply(t *testing.T) {
	got := replyFrameTypes(t,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hello from jargo!"),
		frames.NewLLMFullResponseEndFrame(),
	)
	want := []string{
		"*frames.LLMContextFrame",
		"*frames.LLMContextAssistantTimestampFrame",
		"*frames.LLMContextAssistantTurnFrame",
	}
	if !slices.Equal(got, want) {
		t.Errorf("reached the end %v, want %v", got, want)
	}
}

// Ported from upstream (test_tts_speak_fires_turn_started_and_stopped). Spoken
// text and the push that records it stop at the aggregator too.
func TestAssistantAggregatorConsumesSpokenTextAndThePush(t *testing.T) {
	started := frames.NewTTSStartedFrame()
	started.AppendToContext = true
	got := replyFrameTypes(t,
		started,
		frames.NewTTSTextFrame("Hello", frames.TextTypeSentence),
		frames.NewLLMAssistantPushAggregationFrame(),
	)
	for _, ty := range got {
		if ty == "*frames.TTSTextFrame" || ty == "*frames.LLMAssistantPushAggregationFrame" {
			t.Errorf("reached the end %v, want the spoken text and the push consumed", got)
		}
	}
}
