package live

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
)

// Port of test_partial_assistant_message_survives_interruption: text streamed
// before a barge-in reaches the conversation even though the frames closing the
// turn are still held waiting for an idle status.
func TestPartialAssistantMessageSurvivesInterruption(t *testing.T) {
	// The frames the service pushes while streaming a reply that is cut short:
	// transcript chunks arrive, the turn's completion is held because the
	// status is IN_PROGRESS, then the user barges in.
	s, sink := newTestService(t)
	ctx := context.Background()
	for _, chunk := range []string{"The answer ", "is 42."} {
		s.handle(ctx, decode(t, `{"serverContent":{"outputTranscription":{"text":"`+chunk+`"}}}`))
	}
	s.handle(ctx, decode(t, `{"serverContent":{"turnComplete":true,"interactionStatus":"IN_PROGRESS"}}`))
	s.handleInterruption(ctx)

	var streamed []frames.Frame
	for _, f := range sink.frames() {
		switch f.(type) {
		case *frames.StartFrame, *frames.LLMServiceMetadataFrame:
			continue
		}
		streamed = append(streamed, f)
	}

	// Replay that sequence into an assistant aggregator, with the interruption
	// the pipeline broadcasts at the barge-in, one frame after another as
	// upstream's harness delivers them.
	convo := frames.NewLLMContext("")
	assistant := aggregators.New(convo).Assistant()
	sinkAfter := newCapture()
	assistant.Link(sinkAfter)
	if err := assistant.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = assistant.Cleanup(ctx) })
	replay := append([]frames.Frame{frames.NewStartFrame()}, streamed...)
	replay = append(replay, frames.NewInterruptionFrame())
	for _, f := range replay {
		if err := assistant.ProcessFrame(ctx, f, processor.Downstream); err != nil {
			t.Fatalf("%s: %v", f.Name(), err)
		}
	}

	var last string
	for _, m := range convo.Messages() {
		if m.Role == frames.RoleAssistant {
			last = m.Text
		}
	}
	if last != "The answer is 42." {
		t.Errorf("assistant message = %q, want the partial reply recorded", last)
	}
}
