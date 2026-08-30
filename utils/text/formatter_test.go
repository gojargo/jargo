package text

import (
	"context"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// textTransform is the shape the TTS base's text transformers take, spelled out
// here rather than imported: service/tts already depends on this package, so it
// cannot be depended on back.
type textTransform func(ctx context.Context, text string, aggregatedBy frames.AggregationType) (string, error)

// TestVoiceFormatterIsATextTransform checks a VoiceFormatter can be handed
// straight to the TTS base's text transformers. It is the shape the base takes,
// so registering one is not meant to need a wrapper written by hand.
func TestVoiceFormatterIsATextTransform(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}

	var transform textTransform = f.Transform

	got, err := transform(t.Context(), "The API costs $5", frames.AnyAggregation)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if !strings.Contains(got, "A P I") || !strings.Contains(got, "five dollars") {
		t.Errorf("Transform = %q, want the acronym spelled out and the amount in words", got)
	}
}
