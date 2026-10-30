package soniox

import (
	"reflect"
	"slices"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttssoniox "github.com/gojargo/jargo/adapter/tts/soniox"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"tts-rt-v2", ttssoniox.SonioxExpressionAdapter{}},
		{"tts-rt-v1", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &ttsSynthesizer{cfg: TTSConfig{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

// wordsOf reports text as Soniox times it, a character at a time.
func wordsOf(t *testing.T, text string) []string {
	t.Helper()
	s := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k"}.withTTSDefaults()}
	timings := &ttsTimings{}
	for i, r := range text {
		timings.Characters = append(timings.Characters, string(r))
		timings.Starts = append(timings.Starts, float64(i)/10)
	}
	var words []string
	if err := s.reportWords(&uctx.CharAccumulator{}, timings,
		func(ws []uctx.WordTiming, _ tts.WordTimingOptions) error {
			for _, w := range ws {
				words = append(words, w.Word)
			}
			return nil
		}); err != nil {
		t.Fatal(err)
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
