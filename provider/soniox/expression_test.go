package soniox

import (
	"reflect"
	"slices"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttssoniox "github.com/gojargo/jargo/adapter/tts/soniox"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"tts-rt-v2", ttssoniox.SonioxExpressionAdapter{}},
		{"tts-rt-v1", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := newTTSSynthesizer(TTSConfig{APIKey: "k", Model: tt.model})
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

// wordsOf reports text as Soniox times it, a character at a time.
func wordsOf(t *testing.T, text string) []string {
	t.Helper()
	s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
	var words []string
	for _, w := range s.toWordTimes("s1", timings(text, 0)) {
		words = append(words, w.Word)
	}
	return words
}

func TestAnAudioTagIsOneWordWithItsSpaces(t *testing.T) {
	got := wordsOf(t, "7. [neutral][long pause] Hi ")
	if !slices.Equal(got, []string{"7.", "[neutral][long pause]", "Hi"}) {
		t.Fatalf("words = %q", got)
	}
}

func TestAStressedWordIsReportedWithoutItsAsterisks(t *testing.T) {
	if got := wordsOf(t, "*Really* good "); !slices.Equal(got, []string{"Really", "good"}) {
		t.Fatalf("words = %q", got)
	}
}
