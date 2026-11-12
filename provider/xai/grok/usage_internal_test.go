package grok

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Ported from upstream. xAI reports reasoning apart from the completion tokens,
// with the total covering both, so they are added to the completion count.
func TestAddsReasoningReportedApartFromCompletionTokens(t *testing.T) {
	got := tokenUsage(frames.LLMTokenUsage{
		PromptTokens: 32, CompletionTokens: 9, TotalTokens: 135, ReasoningTokens: new(int64(94)),
	})
	reasoning, _ := frames.TokenCount(got.ReasoningTokens)
	if got.CompletionTokens != 103 || reasoning != 94 || got.TotalTokens != 135 {
		t.Errorf("usage = completion %d, reasoning %d, total %d; want 103, 94, 135",
			got.CompletionTokens, reasoning, got.TotalTokens)
	}
}

// Ported from upstream. A total that already matches the prompt and completion
// counts means reasoning was included, and the counts are kept as reported.
func TestKeepsReasoningAlreadyInCompletionTokens(t *testing.T) {
	got := tokenUsage(frames.LLMTokenUsage{
		PromptTokens: 32, CompletionTokens: 103, TotalTokens: 135, ReasoningTokens: new(int64(94)),
	})
	if got.CompletionTokens != 103 {
		t.Errorf("completion tokens = %d, want 103", got.CompletionTokens)
	}
}
