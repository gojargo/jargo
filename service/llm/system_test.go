package llm_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// plainGenerator is a provider with no settings of its own, which is what makes
// the base hold the general LLM settings for it.
type plainGenerator struct{}

func (plainGenerator) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func TestSystemInstructionStartsEmpty(t *testing.T) {
	svc := llm.New("FakeLLM", plainGenerator{})
	if got := svc.SystemInstruction(); got != "" {
		t.Errorf("SystemInstruction = %q, want empty", got)
	}
}

func TestSetSystemInstructionReplacesTheBase(t *testing.T) {
	svc := llm.New("FakeLLM", plainGenerator{})
	svc.SetSystemInstruction("be brief")
	if got := svc.SystemInstruction(); got != "be brief" {
		t.Errorf("SystemInstruction = %q, want %q", got, "be brief")
	}
	// Replacing the base rebuilds from it rather than adding to it.
	svc.SetSystemInstruction("be warm")
	if got := svc.SystemInstruction(); got != "be warm" {
		t.Errorf("SystemInstruction = %q, want %q", got, "be warm")
	}
}

func TestAppendedInstructionsSurviveANewBase(t *testing.T) {
	svc := llm.New("FakeLLM", plainGenerator{})
	svc.SetSystemInstruction("be brief")
	svc.AppendSystemInstruction("answer in JSON")

	got := svc.SystemInstruction()
	if !strings.HasPrefix(got, "be brief") {
		t.Errorf("SystemInstruction = %q, want the base prompt first", got)
	}
	if !strings.Contains(got, "answer in JSON") {
		t.Errorf("SystemInstruction = %q, want the appended text", got)
	}

	// The application replaced its prompt; what a framework component appended is
	// still needed, so it survives.
	svc.SetSystemInstruction("be warm")
	got = svc.SystemInstruction()
	if !strings.HasPrefix(got, "be warm") {
		t.Errorf("SystemInstruction = %q, want the new base prompt first", got)
	}
	if !strings.Contains(got, "answer in JSON") {
		t.Errorf("SystemInstruction = %q, want the appended text kept", got)
	}
	if strings.Contains(got, "be brief") {
		t.Errorf("SystemInstruction = %q, want the old base prompt gone", got)
	}
}

func TestComposingIsRepeatable(t *testing.T) {
	svc := llm.New("FakeLLM", plainGenerator{})
	svc.SetSystemInstruction("base")
	svc.AppendSystemInstruction("addon")
	first := svc.SystemInstruction()

	// Rebuilding always starts from the base, so nothing compounds.
	svc.SetSystemInstruction("base")
	if got := svc.SystemInstruction(); got != first {
		t.Errorf("SystemInstruction = %q, want it unchanged at %q", got, first)
	}
}

// A settings update carrying a system instruction is how a conversation flow
// changes the assistant's persona mid-call.
func TestSystemInstructionSettingsUpdate(t *testing.T) {
	svc := llm.New("FakeLLM", plainGenerator{})
	svc.AppendSystemInstruction("answer in JSON")
	task, stop := runLLM(t, svc)
	defer stop()

	task.QueueFrame(frames.NewLLMUpdateSettingsFrame(&settings.LLM{
		SystemInstruction: settings.Set("you are a barista"),
	}))

	if !waitFor(3*time.Second, func() bool {
		return strings.Contains(svc.SystemInstruction(), "you are a barista")
	}) {
		t.Fatal("timed out waiting for the system instruction to be applied")
	}

	got := svc.SystemInstruction()
	if !strings.HasPrefix(got, "you are a barista") {
		t.Errorf("SystemInstruction = %q, want the new persona first", got)
	}
	if !strings.Contains(got, "answer in JSON") {
		t.Errorf("SystemInstruction = %q, want the appended text kept", got)
	}
}
