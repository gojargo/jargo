package openairealtime

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestResponseDoneUsageParsing feeds a response.done event and asserts the token
// usage, including the audio/text breakdown from the input/output token details.
func TestResponseDoneUsageParsing(t *testing.T) {
	raw := `{
		"type": "response.done",
		"response": {
			"usage": {
				"total_tokens": 200,
				"input_tokens": 150,
				"output_tokens": 50,
				"input_token_details": {
					"text_tokens": 100,
					"audio_tokens": 50,
					"cached_tokens": 20
				},
				"output_token_details": {
					"text_tokens": 10,
					"audio_tokens": 40
				}
			}
		}
	}`

	var ev serverEvent
	if err := json.Unmarshal([]byte(raw), &ev); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if ev.Response == nil || ev.Response.Usage == nil {
		t.Fatal("response usage not parsed")
	}

	got := ev.Response.Usage.tokenUsage()
	want := frames.LLMTokenUsage{
		PromptTokens:      150,
		CompletionTokens:  50,
		TotalTokens:       200,
		CacheReadTokens:   20,
		InputTextTokens:   100,
		InputAudioTokens:  50,
		OutputTextTokens:  10,
		OutputAudioTokens: 40,
	}
	if got != want {
		t.Fatalf("tokenUsage = %+v, want %+v", got, want)
	}
}

// TestNonResponseEventHasNoUsage confirms events without a response object leave
// the usage nil, so the service does not report spurious token usage.
func TestNonResponseEventHasNoUsage(t *testing.T) {
	raw := `{"type": "response.audio.delta", "delta": "AAAA"}`
	var ev serverEvent
	if err := json.Unmarshal([]byte(raw), &ev); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if ev.Response != nil {
		t.Fatalf("response = %+v, want nil", ev.Response)
	}
}
