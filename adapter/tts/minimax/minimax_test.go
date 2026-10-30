package minimax_test

import (
	"testing"

	"github.com/gojargo/jargo/adapter/tts/minimax"
)

func TestSoundsAndPauses(t *testing.T) {
	a := minimax.MiniMaxExpressionAdapter{}
	if got := a.FormatSound("laughs"); got != "(laughs)" {
		t.Errorf("sound = %q", got)
	}
	if got := a.FormatPause(600); got != "<#0.6#>" {
		t.Errorf("pause = %q", got)
	}
	if got := a.FormatText("Hi.", []string{"happy"}, nil); got != "Hi." {
		t.Errorf("text = %q", got)
	}
}

func TestVocabularyIsFixed(t *testing.T) {
	// MiniMax reads an interjection outside its list aloud.
	c := minimax.MiniMaxExpressionAdapter{}.Capabilities()
	if c.AllowsFreeTextSounds || c.AllowsFreeTextDelivery {
		t.Fatalf("capabilities = %+v", c)
	}
}
