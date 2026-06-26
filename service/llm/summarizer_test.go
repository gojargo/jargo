package llm_test

import (
	"context"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// recordGen records the conversation it is asked to generate from and emits a
// fixed list of deltas.
type recordGen struct {
	deltas    []string
	gotSystem string
	gotUser   string
}

func (g *recordGen) Generate(_ context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	g.gotSystem = convo.System()
	if msgs := convo.Messages(); len(msgs) > 0 {
		g.gotUser = msgs[0].Text
	}
	for _, d := range g.deltas {
		if err := emit(d); err != nil {
			return err
		}
	}
	return nil
}

func TestSummarizerBuildsPromptAndCollectsText(t *testing.T) {
	gen := &recordGen{deltas: []string{"user ", "asked about the weather"}}
	s := llm.NewSummarizer(gen)

	dropped := []frames.Message{
		{Role: frames.RoleUser, Text: "what's the weather?"},
		{Role: frames.RoleAssistant, Text: "It is sunny."},
	}
	out, err := s.Summarize(context.Background(), "earlier: greeted", dropped)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if out != "user asked about the weather" {
		t.Fatalf("summary = %q, want the trimmed concatenation of deltas", out)
	}

	for _, want := range []string{"earlier: greeted", "User: what's the weather?", "Assistant: It is sunny."} {
		if !strings.Contains(gen.gotUser, want) {
			t.Fatalf("prompt %q missing %q", gen.gotUser, want)
		}
	}
	if !strings.Contains(strings.ToLower(gen.gotSystem), "summary") {
		t.Fatalf("system instruction = %q, want it to mention the summary task", gen.gotSystem)
	}
}

func TestSummarizerRendersToolTurns(t *testing.T) {
	gen := &recordGen{deltas: []string{"ok"}}
	s := llm.NewSummarizer(gen)

	dropped := []frames.Message{
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{{ID: "t1", Name: "lookup"}}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "t1", Name: "lookup", Content: "42"}}},
	}
	if _, err := s.Summarize(context.Background(), "", dropped); err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	for _, want := range []string{"Assistant called tool lookup", "Tool result (lookup): 42"} {
		if !strings.Contains(gen.gotUser, want) {
			t.Fatalf("prompt %q missing %q", gen.gotUser, want)
		}
	}
}

func TestSummarizerWithInstruction(t *testing.T) {
	gen := &recordGen{deltas: []string{"x"}}
	s := llm.NewSummarizer(gen, llm.WithInstruction("be very terse"))
	if _, err := s.Summarize(context.Background(), "", []frames.Message{{Role: frames.RoleUser, Text: "hi"}}); err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if gen.gotSystem != "be very terse" {
		t.Fatalf("system = %q, want the overridden instruction", gen.gotSystem)
	}
}
