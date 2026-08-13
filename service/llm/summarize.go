package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/gojargo/jargo/frames"
	uctx "github.com/gojargo/jargo/utils/context"
)

// The ways a summary generation can fail before it reaches the model.
var (
	// ErrNothingToSummarize marks a request whose conversation has nothing the
	// summary could cover, once the messages kept back and any unresolved tool
	// call are taken out.
	ErrNothingToSummarize = errors.New("no messages to summarize")
	// ErrEmptySummary marks a model that answered the summarization with nothing.
	ErrEmptySummary = errors.New("the model returned an empty summary")
	// ErrNoInference marks a service that cannot answer a conversation off the
	// pipeline, and so cannot generate a summary.
	ErrNoInference = errors.New("the LLM service does not run inference")
)

// inferencer is the concrete service behind this base, when it can answer a
// conversation on its own. Generating a summary is the one thing the base does
// that needs the provider directly rather than through the frame path.
func (b *Base) inferencer() (Inferencer, error) {
	inf, ok := b.Self().(Inferencer)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNoInference, b.Name())
	}
	return inf, nil
}

// GenerateSummary compresses the conversation a request names, on inf, and
// reports the summary together with the index of the last message it covers.
//
// The summary is generated from a transcript of the selected messages put to
// the model as a conversation of its own, so nothing about the summarization
// enters the conversation being summarized.
func GenerateSummary(
	ctx context.Context,
	inf Inferencer,
	request *frames.LLMContextSummaryRequestFrame,
) (string, int, error) {
	selected := uctx.GetMessagesToSummarize(request.Context, request.MinMessagesToKeep)
	if len(selected.Messages) == 0 {
		return "", -1, ErrNothingToSummarize
	}

	slog.Debug("generating a conversation summary",
		"messages", len(selected.Messages),
		"last_index", selected.LastSummarizedIndex,
		"target_tokens", request.TargetContextTokens)

	transcript := uctx.FormatMessagesForSummary(selected.Messages)
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{
		Role: frames.RoleUser,
		Text: "Conversation history:\n" + transcript,
	}})

	summary, err := inf.RunInference(ctx, convo, InferenceOptions{
		MaxTokens:         request.TargetContextTokens,
		SystemInstruction: request.SummarizationPrompt,
	})
	if err != nil {
		return "", -1, fmt.Errorf("generating the conversation summary: %w", err)
	}
	if summary == "" {
		return "", -1, ErrEmptySummary
	}

	summary = strings.TrimSpace(summary)
	slog.Info("generated a conversation summary",
		"chars", len(summary), "messages", len(selected.Messages))

	return summary, selected.LastSummarizedIndex, nil
}

// handleSummaryRequest answers a summarization request from the aggregator. The
// generation runs on a goroutine of its own, so a conversation is never held up
// waiting for its own history to be compressed, and the outcome is broadcast so
// it reaches the summarizer whichever side of this service it sits on.
func (b *Base) handleSummaryRequest(request *frames.LLMContextSummaryRequestFrame) {
	slog.Debug("processing a summarization request", "request_id", request.RequestID)

	b.summaryWG.Go(func() { b.generateSummaryTask(request) })
}

// generateSummaryTask runs one summary generation and broadcasts its outcome.
//
// A generation that ran out of time is told apart from one that failed. Both
// are reported to the pipeline, but only a failure is carried on the result:
// running out of time closes the request out with a summary that covers
// nothing, where a failure is handed on so whatever asked for the summary
// learns why there is none.
func (b *Base) generateSummaryTask(request *frames.LLMContextSummaryRequestFrame) {
	timeout := request.SummarizationTimeout
	if timeout == 0 {
		timeout = frames.DefaultSummarizationTimeout
	}
	ctx, cancel := context.WithTimeout(b.summaryCtx, timeout)
	defer cancel()

	summary, lastIndex := "", -1
	inf, err := b.inferencer()
	if err == nil {
		summary, lastIndex, err = GenerateSummary(ctx, inf, request)
	}

	var result *frames.LLMContextSummaryResultFrame
	switch {
	case err == nil:
		result = frames.NewLLMContextSummaryResultFrame(request.RequestID, summary, lastIndex)
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		b.PushError(b.summaryCtx,
			fmt.Sprintf("context summarization timed out after %s", timeout), err, false)
		result = frames.NewLLMContextSummaryResultFrame(request.RequestID, "", -1)
	default:
		b.PushError(b.summaryCtx, "context summarization failed", err, false)
		result = frames.NewLLMContextSummaryErrorFrame(request.RequestID, err.Error())
	}

	if err := b.Broadcast(b.summaryCtx, func() frames.Frame {
		f := *result
		return &f
	}); err != nil {
		slog.Error("delivering the summary result failed", "error", err)
	}
}
