package tracing_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/telemetry/tracing"
)

// TestProviderName checks that a service name reduces to the provider name the
// GenAI conventions give it.
func TestProviderName(t *testing.T) {
	cases := []struct {
		service, kind, want string
	}{
		{"OpenAILLM", "LLM", "openai"},
		{"AnthropicLLM", "LLM", "anthropic"},
		{"CartesiaTTS", "TTS", "cartesia"},
		{"DeepgramSTT", "STT", "deepgram"},
		// The suffix is dropped only from the end, so a provider whose own name
		// ends in the kind it implements keeps it.
		{"XTTSTTS", "TTS", "xtts"},
		{"LMNTTTS", "TTS", "lmnt"},
		// Services the conventions spell differently are looked up.
		{"GoogleLLM", "LLM", "gcp.gemini"},
		{"GoogleVertexLLM", "LLM", "gcp.vertex_ai"},
		{"AzureOpenAILLM", "LLM", "azure.ai.openai"},
		{"BedrockLLM", "LLM", "aws.bedrock"},
		{"XAILLM", "LLM", "x_ai"},
		{"MistralLLM", "LLM", "mistral_ai"},
	}
	for _, c := range cases {
		if got := tracing.ProviderName(c.service, c.kind); got != c.want {
			t.Errorf("ProviderName(%q, %q) = %q, want %q", c.service, c.kind, got, c.want)
		}
	}
}

// TestSetLLMAttributesSplitsParameters checks that the generation parameters the
// conventions name are recorded under gen_ai.request.*, and the rest under
// param.*, so a backend reading the standard keys finds them where it expects.
func TestSetLLMAttributesSplitsParameters(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetLLMAttributes(spanOf(ctx), tracing.LLMAttributes{
			Service: "OpenAILLM",
			Model:   "gpt-4.1",
			Stream:  true,
			Parameters: map[string]any{
				"temperature": 0.7,
				"max_tokens":  int64(512),
				"model":       "gpt-4.1",
				"reasoning":   "low",
			},
			ExtraParameters: map[string]any{"service_tier": "flex"},
		})
	})

	want := map[string]string{
		"gen_ai.provider.name":       "openai",
		"gen_ai.request.model":       "gpt-4.1",
		"gen_ai.operation.name":      "chat",
		"gen_ai.output.type":         "text",
		"stream":                     "true",
		"gen_ai.request.temperature": "0.7",
		"gen_ai.request.max_tokens":  "512",
		"param.model":                "gpt-4.1",
		"param.reasoning":            "low",
		"extra.service_tier":         "flex",
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetAttributesSkipsUnrenderableSettings checks that a setting a span cannot
// carry is left off rather than rendered as text. Settings are plain data and
// may hold a value with no attribute representation.
func TestSetAttributesSkipsUnrenderableSettings(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetTTSAttributes(spanOf(ctx), tracing.TTSAttributes{
			Service: "CartesiaTTS",
			Model:   "sonic-3",
			VoiceID: "a-voice",
			Settings: map[string]any{
				"speed":    1.2,
				"language": "en",
				"emotion":  []string{"calm"},
				"cleared":  nil,
			},
		})
	})

	if attrs["settings.speed"] != "1.2" || attrs["settings.language"] != "en" {
		t.Errorf("scalar settings missing (all: %v)", attrs)
	}
	for _, k := range []string{"settings.emotion", "settings.cleared"} {
		if v, ok := attrs[k]; ok {
			t.Errorf("attr %q = %q, want it left off the span", k, v)
		}
	}
}

// TestSetSTTAttributesOptionalCounts checks that a reported false lands on the
// span while an unreported value stays off it. A transcript the service marked
// as not final is a statement about it; one it said nothing about is not.
func TestSetSTTAttributesOptionalCounts(t *testing.T) {
	interim := false
	withFinal := stringAttrs(t, func(ctx context.Context) {
		tracing.SetSTTAttributes(spanOf(ctx), tracing.STTAttributes{
			Service: "DeepgramSTT", Model: "nova-3", Final: &interim,
		})
	})
	if withFinal["is_final"] != "false" {
		t.Errorf("is_final = %q, want the reported false", withFinal["is_final"])
	}

	withoutFinal := stringAttrs(t, func(ctx context.Context) {
		tracing.SetSTTAttributes(spanOf(ctx), tracing.STTAttributes{
			Service: "DeepgramSTT", Model: "nova-3",
		})
	})
	if v, ok := withoutFinal["is_final"]; ok {
		t.Errorf("is_final = %q, want it left off when the service said nothing", v)
	}
	// vad_enabled is always recorded, as a plain flag rather than an optional.
	if withoutFinal["vad_enabled"] != "false" {
		t.Errorf("vad_enabled = %q, want it always recorded", withoutFinal["vad_enabled"])
	}
}

// TestSetGeminiLiveAttributes checks the shape of a Gemini Live operation span:
// the provider is fixed, the operation is named twice (once under the standard
// key and once under the service one), and the toolset is summarized.
func TestSetGeminiLiveAttributes(t *testing.T) {
	isInput := true
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetGeminiLiveAttributes(spanOf(ctx), tracing.GeminiLiveAttributes{
			Model:             "gemini-live-2.5",
			Operation:         "llm_response",
			VoiceID:           "Charon",
			Modalities:        "AUDIO",
			Tools:             []string{"book_table", "", "cancel"},
			ToolsSerialized:   `[{"name":"book_table"}]`,
			Transcript:        "a coffee please",
			TranscriptIsInput: &isInput,
			Settings: map[string]any{
				"voice": "Charon",
				// A grouped setting is expanded one level rather than dropped.
				"vad": map[string]any{"disabled": false, "start_sensitivity": "HIGH"},
			},
		})
	})

	want := map[string]string{
		"gen_ai.provider.name":           "gcp.gemini",
		"gen_ai.request.model":           "gemini-live-2.5",
		"gen_ai.operation.name":          "llm_response",
		"service.operation":              "llm_response",
		"voice_id":                       "Charon",
		"modalities":                     "AUDIO",
		"transcript":                     "a coffee please",
		"transcript.is_input":            "true",
		"tools.count":                    "3",
		"tools.available":                "true",
		"tools.names":                    "book_table,cancel",
		"tools.definitions":              `[{"name":"book_table"}]`,
		"settings.voice":                 "Charon",
		"settings.vad.disabled":          "false",
		"settings.vad.start_sensitivity": "HIGH",
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

// TestSetOpenAIRealtimeAttributes checks the shape of a Realtime operation span,
// including the function calls a turn requested and the session's grouped
// turn-detection configuration.
func TestSetOpenAIRealtimeAttributes(t *testing.T) {
	attrs := stringAttrs(t, func(ctx context.Context) {
		tracing.SetOpenAIRealtimeAttributes(spanOf(ctx), tracing.OpenAIRealtimeAttributes{
			Model:         "gpt-realtime",
			Operation:     "llm_response",
			FunctionCalls: []string{"book_table", "send_confirmation"},
			SessionProperties: map[string]any{
				"voice":          "alloy",
				"turn_detection": map[string]any{"type": "server_vad", "threshold": 0.5},
			},
			Extra: map[string]any{"response.status": "completed", "response.output_items": 3},
		})
	})

	want := map[string]string{
		"gen_ai.provider.name":             "openai",
		"gen_ai.request.model":             "gpt-realtime",
		"gen_ai.operation.name":            "llm_response",
		"service.operation":                "llm_response",
		"function_calls.count":             "2",
		"function_calls.first_name":        "book_table",
		"function_calls.all_names":         "book_table,send_confirmation",
		"session.voice":                    "alloy",
		"session.turn_detection.type":      "server_vad",
		"session.turn_detection.threshold": "0.5",
		"response.status":                  "completed",
		"response.output_items":            "3",
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}
