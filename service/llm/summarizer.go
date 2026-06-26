package llm

import (
	"context"
	"strings"

	"github.com/gojargo/jargo/frames"
)

// defaultSummaryInstruction steers the model toward a compact, faithful running
// summary suited to a continuing voice conversation.
const defaultSummaryInstruction = "You maintain a running summary of a conversation. " +
	"Given the summary so far and the next stretch of conversation, return a single updated summary " +
	"that preserves the user's goals, decisions, stated facts, names, and any open questions. " +
	"Write terse third-person notes, not a transcript, and add no greetings, commentary, or speculation."

// Summarizer condenses older conversation turns into a compact summary using a
// Generator. It satisfies aggregators.Summarizer, so the same kind of provider
// that answers can also compress history. Give it its own service instance —
// ideally a small, fast model — rather than the one wired into the live
// pipeline: it is invoked off to the side, and a standalone instance emits no
// pipeline frames.
//
//	sum := llm.NewSummarizer(anthropic.NewLLM(anthropic.Config{}))
//	pair := aggregators.New(ctx, aggregators.WithSummarization(
//	    aggregators.SummarizeConfig{Summarizer: sum}))
type Summarizer struct {
	gen         Generator
	instruction string
}

// SummarizerOption configures a Summarizer.
type SummarizerOption func(*Summarizer)

// WithInstruction overrides the system instruction that steers the summary. An
// empty string is ignored.
func WithInstruction(instruction string) SummarizerOption {
	return func(s *Summarizer) {
		if instruction != "" {
			s.instruction = instruction
		}
	}
}

// NewSummarizer builds a Summarizer backed by gen.
func NewSummarizer(gen Generator, opts ...SummarizerOption) *Summarizer {
	s := &Summarizer{gen: gen, instruction: defaultSummaryInstruction}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Summarize renders the prior summary and the dropped messages into a prompt,
// runs the generator, and returns the collected summary text. It pushes no
// frames: Generate only streams text deltas to the supplied callback.
func (s *Summarizer) Summarize(ctx context.Context, prior string, dropped []frames.Message) (string, error) {
	convo := frames.NewLLMContext(s.instruction)
	convo.AddUserMessage(buildSummaryPrompt(prior, dropped))

	var b strings.Builder
	if err := s.gen.Generate(ctx, convo, func(text string) error {
		b.WriteString(text)
		return nil
	}); err != nil {
		return "", err
	}
	return strings.TrimSpace(b.String()), nil
}

// buildSummaryPrompt renders the prior summary (if any) and the dropped turns as
// a plain transcript for the model to fold together.
func buildSummaryPrompt(prior string, dropped []frames.Message) string {
	var b strings.Builder
	if prior != "" {
		b.WriteString("Summary so far:\n")
		b.WriteString(prior)
		b.WriteString("\n\n")
	}
	b.WriteString("New conversation to fold in:\n")
	for _, m := range dropped {
		writeTurn(&b, m)
	}
	b.WriteString("\nReturn the updated summary.")
	return b.String()
}

// writeTurn renders one message as a transcript line, flattening tool turns into
// a readable note.
func writeTurn(b *strings.Builder, m frames.Message) {
	switch {
	case len(m.ToolResults) > 0:
		for _, r := range m.ToolResults {
			b.WriteString("Tool result")
			if r.Name != "" {
				b.WriteString(" (")
				b.WriteString(r.Name)
				b.WriteString(")")
			}
			b.WriteString(": ")
			b.WriteString(r.Content)
			b.WriteString("\n")
		}
	case len(m.ToolCalls) > 0:
		if m.Text != "" {
			writeLine(b, "Assistant: ", m.Text)
		}
		for _, c := range m.ToolCalls {
			writeLine(b, "Assistant called tool ", c.Name)
		}
	case m.Text == "":
		// Nothing to render.
	case m.Role == frames.RoleUser:
		writeLine(b, "User: ", m.Text)
	case m.Role == frames.RoleAssistant:
		writeLine(b, "Assistant: ", m.Text)
	default:
		writeLine(b, "System: ", m.Text)
	}
}

// writeLine writes "<prefix><text>\n" to b without building a temporary string.
func writeLine(b *strings.Builder, prefix, text string) {
	b.WriteString(prefix)
	b.WriteString(text)
	b.WriteString("\n")
}
