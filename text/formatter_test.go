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
type textTransform func(ctx context.Context, text string, aggregatedBy frames.TextType) (string, error)

// TestVoiceFormatterIsATextTransform checks a VoiceFormatter can be handed
// straight to the TTS base's text transformers. It is the shape the base takes,
// so registering one is not meant to need a wrapper written by hand.
func TestVoiceFormatterIsATextTransform(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}

	var transform textTransform = f.Transform

	got, err := transform(t.Context(), "The API costs $5", frames.AnyTextType)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if !strings.Contains(got, "A P I") || !strings.Contains(got, "five dollars") {
		t.Errorf("Transform = %q, want the acronym spelled out and the amount in words", got)
	}
}

// StripURLs is handed to the Markdown stripping: on, a link is reduced to its
// label; off, the URL stays.
func TestVoiceFormatterStripURLs(t *testing.T) {
	text := "See [the docs](https://example.com/docs) here."

	stripped, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	opts := DefaultFormatterOptions()
	opts.StripURLs = false
	kept, err := NewVoiceFormatter(opts)
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}

	if got, _ := stripped.Transform(t.Context(), text, frames.AnyTextType); got != "See the docs here." {
		t.Errorf("with StripURLs, Transform = %q, want %q", got, "See the docs here.")
	}
	got, _ := kept.Transform(t.Context(), text, frames.AnyTextType)
	if !strings.Contains(got, "https://example.com/docs") {
		t.Errorf("without StripURLs, Transform = %q, want the URL kept", got)
	}
}
