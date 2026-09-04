package frames_test

import (
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

func TestMetricsFrameIsSystemFrame(t *testing.T) {
	var f frames.Frame = frames.NewMetricsFrame()
	if _, ok := f.(frames.SystemFrame); !ok {
		t.Error("MetricsFrame should be a SystemFrame")
	}
	if _, ok := f.(frames.DataFrame); ok {
		t.Error("MetricsFrame should not be a DataFrame")
	}
}

// TestMetricsFrameCarriesSeveralKinds covers the shape the frame exists for: one
// frame reporting more than one kind of measurement, from more than one
// processor.
func TestMetricsFrameCarriesSeveralKinds(t *testing.T) {
	llm := frames.BaseMetricsData{Processor: "AnthropicLLM#1", Model: "claude"}
	tts := frames.BaseMetricsData{Processor: "CartesiaTTS#2", Model: "sonic"}

	f := frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: llm, Value: 250 * time.Millisecond},
		frames.LLMUsageMetricsData{
			BaseMetricsData: llm,
			Value:           frames.LLMTokenUsage{PromptTokens: 12, CompletionTokens: 3},
		},
		frames.TTSUsageMetricsData{BaseMetricsData: tts, Value: 40},
	)

	if len(f.Data) != 3 {
		t.Fatalf("Data has %d measurements, want 3", len(f.Data))
	}
	if got := f.Data[0].MetricsProcessor(); got != "AnthropicLLM#1" {
		t.Errorf("Data[0] processor = %q, want AnthropicLLM#1", got)
	}
	if got := f.Data[2].MetricsProcessor(); got != "CartesiaTTS#2" {
		t.Errorf("Data[2] processor = %q, want CartesiaTTS#2", got)
	}
	if got := f.Data[2].MetricsModel(); got != "sonic" {
		t.Errorf("Data[2] model = %q, want sonic", got)
	}

	// A consumer reads a measurement by switching on its type.
	usage, ok := f.Data[1].(frames.LLMUsageMetricsData)
	if !ok {
		t.Fatalf("Data[1] is %T, want LLMUsageMetricsData", f.Data[1])
	}
	if usage.Value.PromptTokens != 12 || usage.Value.CompletionTokens != 3 {
		t.Errorf("token usage = %+v, want 12 in / 3 out", usage.Value)
	}
}

func TestMetricsFrameStringNamesASingleProcessor(t *testing.T) {
	f := frames.NewMetricsFrame(frames.ProcessingMetricsData{
		Processor: "AnthropicLLM#1",
		Value:     time.Second,
	})
	if got := f.String(); !strings.Contains(got, "AnthropicLLM#1") {
		t.Errorf("String() = %q, want the processor named", got)
	}
}

// TestTTFAReportsItsBreakdown pins the relationship the type documents: the time
// to the first audible sample is the time to first byte plus the silence padded
// on in front of it.
func TestTTFAReportsItsBreakdown(t *testing.T) {
	d := frames.TTFAMetricsData{
		Processor:      "tts-1",
		TTFA:           300 * time.Millisecond,
		TTFB:           200 * time.Millisecond,
		LeadingSilence: 100 * time.Millisecond,
	}
	if d.TTFB+d.LeadingSilence != d.TTFA {
		t.Errorf("TTFB %v + leading silence %v should equal TTFA %v", d.TTFB, d.LeadingSilence, d.TTFA)
	}
}
