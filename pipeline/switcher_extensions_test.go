package pipeline_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// quietLLM stands in for a model that is never asked to generate.
type quietLLM struct{}

func (quietLLM) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

// Installing the turn-completion extension reaches inactive members too. The
// stop strategy installs it with one frame on start, as a function that builds
// one per member. A member that missed it would emit none of the markers the
// strategy waits on, so user turns would stop being finalized as soon as it
// became the active LLM.
func TestTurnCompletionExtensionAppliedToAllMemberLLMs(t *testing.T) {
	members := []*llm.Base{llm.New("LLM1", quietLLM{}), llm.New("LLM2", quietLLM{})}
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{members[0], members[1]}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}
	task, _, stop := runCollector(t, sw)
	defer stop()

	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions{
		func() llm.Extension { return llm.NewTurnCompletionExtension(llm.UserTurnCompletionConfig{}) },
	})
	update := frames.NewLLMUpdateSettingsFrame(&delta)
	update.ReachInactiveServices = true
	task.QueueFrame(update)

	var built []*llm.TurnCompletionExtension
	deadline := time.Now().Add(3 * time.Second)
	for _, member := range members {
		for {
			extension, ok := llm.GetExtension[*llm.TurnCompletionExtension](member)
			if ok && member.SystemInstruction() == llm.UserTurnCompletionInstructions {
				built = append(built, extension)
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("%s never took the turn-completion extension and its instructions", member.Name())
			}
			time.Sleep(time.Millisecond)
		}
	}
	if built[0] == built[1] {
		t.Error("the members share one extension; each needs one of its own")
	}
}
