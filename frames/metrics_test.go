package frames_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestMetricsFrameIsSystemFrame(t *testing.T) {
	var f frames.Frame = frames.NewMetricsFrame("AnthropicLLM#1")
	if _, ok := f.(frames.SystemFrame); !ok {
		t.Error("MetricsFrame should be a SystemFrame")
	}
	if _, ok := f.(frames.DataFrame); ok {
		t.Error("MetricsFrame should not be a DataFrame")
	}
}

func TestMetricsFrameStringReportsTokens(t *testing.T) {
	f := frames.NewMetricsFrame("AnthropicLLM#1")
	f.Tokens = &frames.LLMTokenUsage{PromptTokens: 12, CompletionTokens: 3}
	got := f.String()
	for _, want := range []string{"AnthropicLLM#1", "12 in", "3 out"} {
		if !strings.Contains(got, want) {
			t.Errorf("String() = %q, want substring %q", got, want)
		}
	}
}
