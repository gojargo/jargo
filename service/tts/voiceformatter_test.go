package tts_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// TestVoiceFormatterRegistersAsATransform checks the end the divergence was at:
// a VoiceFormatter goes straight into SetTextTransformers, reshapes what the
// provider is given, and leaves the conversation's record of the turn alone.
func TestVoiceFormatterRegistersAsATransform(t *testing.T) {
	f, err := ttstext.NewVoiceFormatter(ttstext.DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}

	h := newTransformHarness(t, nil)
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform:    f.Transform,
	})

	h.runTurn(t, "The API costs $5.")

	spoken := h.syn.texts()
	if len(spoken) != 1 || !strings.Contains(spoken[0], "A P I") ||
		!strings.Contains(spoken[0], "five dollars") {
		t.Errorf("spoken = %q, want the acronym spelled out and the amount in words", spoken)
	}
	for _, recorded := range h.recorded() {
		if strings.Contains(recorded, "A P I") {
			t.Errorf("the conversation recorded %q, want what the model wrote", recorded)
		}
	}
}
