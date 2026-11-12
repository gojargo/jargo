package frames_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestAsTextFrameMatchesEveryTextKind checks every frame that is a text frame,
// of whatever kind, gives up its text, and a frame that is not gives none.
func TestAsTextFrameMatchesEveryTextKind(t *testing.T) {
	for _, f := range []frames.Frame{
		frames.NewTextFrame("a"),
		frames.NewLLMTextFrame("a"),
		frames.NewAggregatedTextFrame("a", frames.TextTypeSentence),
		frames.NewTTSTextFrame("a", frames.TextTypeWord),
		frames.NewTranscriptionFrame("a", "u", "now"),
		frames.NewInterimTranscriptionFrame("a", "u", "now"),
	} {
		if tf, ok := frames.AsTextFrame(f); !ok || tf.Text != "a" {
			t.Errorf("%T: AsTextFrame = %v, %v, want its text", f, tf, ok)
		}
	}
	if _, ok := frames.AsTextFrame(frames.NewEndFrame()); ok {
		t.Error("AsTextFrame matched an EndFrame")
	}
}

// TestWithTextCopiesTheFrame checks the copy keeps the frame's kind, id and
// fields with the new text, and leaves the original as it was.
func TestWithTextCopiesTheFrame(t *testing.T) {
	skip := true
	orig := frames.NewLLMTextFrame("hello")
	orig.SkipTTS = &skip

	got, ok := frames.WithText(orig, "HELLO")
	if !ok {
		t.Fatal("WithText refused an LLM token")
	}
	cp, isToken := got.(*frames.LLMTextFrame)
	if !isToken {
		t.Fatalf("copy is %T, want an LLM token", got)
	}
	if cp.Text != "HELLO" || cp.SkipTTS == nil || !*cp.SkipTTS || cp.ID() != orig.ID() {
		t.Errorf("copy = %+v, want the new text with the original's id and fields", cp)
	}
	if orig.Text != "hello" {
		t.Errorf("original text = %q, want it untouched", orig.Text)
	}
}
