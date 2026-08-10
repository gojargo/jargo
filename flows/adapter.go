package flows

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// summaryHeader introduces the summary in the message that carries it into the
// conversation.
const summaryHeader = "Here's a summary of the conversation:"

// Inferencer answers a conversation once, off to the side of the pipeline. It is
// what the reset-with-summary strategy summarizes through; the concrete LLM
// services satisfy it.
type Inferencer interface {
	RunInference(ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions) (string, error)
}

// summaryAdapter generates and formats the conversation summary the
// reset-with-summary strategy puts at the head of the new context.
type summaryAdapter struct{}

// formatSummaryMessage renders a summary as the message that carries it into the
// conversation. It is a developer message: an out-of-band note to the model
// about what was said, not something anyone in the conversation said.
func (summaryAdapter) formatSummaryMessage(summary string) frames.Message {
	return frames.Message{
		Role: frames.RoleDeveloper,
		Text: fmt.Sprintf("%s\n%s", summaryHeader, summary),
	}
}

// generateSummary summarizes the conversation with a one-shot inference, off to
// the side of the pipeline so nothing is spoken.
//
// It reports an empty summary rather than an error when the inference fails:
// the caller falls back to appending, which keeps the conversation whole, and a
// failed summary is not worth failing a transition over.
func (summaryAdapter) generateSummary(
	ctx context.Context, inf Inferencer, prompt string, convo *frames.LLMContext,
) string {
	history := &frames.LLMContext{}
	history.SetMessages([]frames.Message{{
		Role: frames.RoleDeveloper,
		Text: fmt.Sprintf("Conversation history: %s", renderMessages(convo.Messages())),
	}})

	summary, err := inf.RunInference(ctx, history, llm.InferenceOptions{SystemInstruction: prompt})
	if err != nil {
		slog.ErrorContext(ctx, "flows: summary generation failed", "err", err)
		return ""
	}
	return summary
}

// renderMessages writes the conversation out for the summarizer to read.
func renderMessages(msgs []frames.Message) string {
	out := make([]byte, 0, 256)
	for i, m := range msgs {
		if i > 0 {
			out = append(out, '\n')
		}
		out = append(out, fmt.Sprintf("{'role': '%s', 'content': '%s'}", m.Role, m.Text)...)
	}
	return string(out)
}
