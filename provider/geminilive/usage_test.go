package geminilive

import (
	"encoding/json"
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
		CacheReadTokens:   12,
		InputTextTokens:   30,
		InputAudioTokens:  70,
		OutputTextTokens:  5,
		OutputAudioTokens: 35,
	}
	if got != want {
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
