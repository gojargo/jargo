package sarvam

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter/openai"
)

// TestTheDefaultBaseURLIsResolvedPerModel checks each model is addressed under
// the API version that serves it: /v1 for sarvam-105b-conversations, /v2 for
// every other.
func TestTheDefaultBaseURLIsResolvedPerModel(t *testing.T) {
	cases := map[string]string{
		"sarvam-105b":               "https://api.sarvam.ai/v2",
		"deepseekv4.1-flash":        "https://api.sarvam.ai/v2",
		"gemma4":                    "https://api.sarvam.ai/v2",
		"glm5.3":                    "https://api.sarvam.ai/v2",
		"sarvam-105b-conversations": "https://api.sarvam.ai/v1",
	}
	for model, want := range cases {
		t.Run(model, func(t *testing.T) {
			if got := llmBaseURL(model); got != want {
				t.Errorf("base URL = %q, want %q", got, want)
			}
		})
	}
}

// TestAToolChoiceRequiresNonEmptyTools checks a tool_choice with no tools to
// choose from is refused before it is sent. The OpenAI adapter only sends a
// tool_choice alongside tools, so this guards what another adapter could build.
func TestAToolChoiceRequiresNonEmptyTools(t *testing.T) {
	s := &LLMSettings{}
	err := s.checkRequest(openai.Params{
		Messages:   []openai.Message{{Role: openai.RoleUser, Content: "Hello"}},
		ToolChoice: "required",
	})
	if err == nil || !strings.Contains(err.Error(), "requires non-empty `tools`") {
		t.Errorf("check = %v, want the tool_choice refused", err)
	}
}
