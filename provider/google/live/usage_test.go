package live

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestUsageMetadataParsing feeds a Live API usage message and asserts the token
// usage, including the per-modality audio/text breakdown folded from the detail
// lists.
func TestUsageMetadataParsing(t *testing.T) {
	raw := `{
		"usageMetadata": {
			"promptTokenCount": 100,
			"responseTokenCount": 40,
			"totalTokenCount": 140,
			"cachedContentTokenCount": 12,
			"promptTokensDetails": [
				{"modality": "TEXT", "tokenCount": 30},
				{"modality": "AUDIO", "tokenCount": 70}
			],
			"responseTokensDetails": [
				{"modality": "TEXT", "tokenCount": 5},
				{"modality": "AUDIO", "tokenCount": 35}
			]
		}
	}`

	var msg serverMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if msg.UsageMetadata == nil {
		t.Fatal("usageMetadata not parsed")
	}

	got := msg.UsageMetadata.tokenUsage()
	want := frames.LLMTokenUsage{
		PromptTokens:      100,
		CompletionTokens:  40,
		TotalTokens:       140,
		CacheReadTokens:   new(int64(12)),
		InputTextTokens:   new(int64(30)),
		InputAudioTokens:  new(int64(70)),
		OutputTextTokens:  new(int64(5)),
		OutputAudioTokens: new(int64(35)),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokenUsage = %+v, want %+v", got, want)
	}
}

// TestUsageMetadataTotalFallback verifies the total is derived from prompt +
// completion when the model omits totalTokenCount.
func TestUsageMetadataTotalFallback(t *testing.T) {
	raw := `{"usageMetadata": {"promptTokenCount": 8, "responseTokenCount": 3}}`
	var msg serverMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := msg.UsageMetadata.tokenUsage().TotalTokens; got != 11 {
		t.Fatalf("TotalTokens = %d, want 11", got)
	}
}

// TestUsageCountsToolResultsAndReasoning covers what Gemini counts apart from
// the prompt and the response: the tool results fed back to the model are input
// and the reasoning is output, and the total is computed rather than taken,
// since the reported one sometimes leaves the reasoning out. Nothing read from
// the cache leaves the cache count unreported rather than zero. (Not a port of
// an upstream test: upstream's covering test runs against the live API.)
func TestUsageCountsToolResultsAndReasoning(t *testing.T) {
	raw := `{"usageMetadata": {
		"promptTokenCount": 100, "toolUsePromptTokenCount": 20,
		"responseTokenCount": 40, "thoughtsTokenCount": 15,
		"totalTokenCount": 160}}`
	var msg serverMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got := msg.UsageMetadata.tokenUsage()
	if got.PromptTokens != 120 || got.CompletionTokens != 55 || got.TotalTokens != 175 {
		t.Errorf("usage = %d prompt, %d completion, %d total, want 120, 55, 175",
			got.PromptTokens, got.CompletionTokens, got.TotalTokens)
	}
	if got.ReasoningTokens == nil || *got.ReasoningTokens != 15 {
		t.Errorf("ReasoningTokens = %v, want 15", got.ReasoningTokens)
	}
	if got.CacheReadTokens != nil {
		t.Errorf("CacheReadTokens = %v, want unreported", *got.CacheReadTokens)
	}
}
