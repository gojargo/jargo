package tts_test

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// Sentence language selection from the TTS settings, in streaming text.

// languageSynth records each sentence it is given and the language the service
// found its boundaries in.
type languageSynth struct {
	base  *tts.Base
	store settings.TTS

	mu        sync.Mutex
	sentences []string
}

func (s *languageSynth) SampleRate() int { return 24000 }

func (s *languageSynth) Settings() any { return &s.store }

// ServiceLanguage names a language the provider's way: its base code.
func (s *languageSynth) ServiceLanguage(l language.Language) string { return l.BaseCode() }

func (s *languageSynth) RunTTS(_ context.Context, text, _ string, yield func(frames.Frame) error) error {
	s.mu.Lock()
	s.sentences = append(s.sentences, strings.TrimSpace(text)+" | "+s.base.TextAggregationLanguage())
	s.mu.Unlock()
	return tts.PCMYielder(yield, s.SampleRate())([]byte{1, 2, 3, 4})
}

func (s *languageSynth) spoken() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.sentences)
}

func newLanguageTTS(lang string) (*tts.Base, *languageSynth) {
	syn := &languageSynth{}
	if lang != "" {
		syn.store.Language = settings.Set(lang)
	}
	syn.base = tts.New("LanguageTTS", syn)
	return syn.base, syn
}

// playLanguage sends each frame in turn and stops the pipeline once they have
// played out.
func playLanguage(t *testing.T, svc *tts.Base, fs ...frames.Frame) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range fs {
		task.QueueFrame(f)
		time.Sleep(50 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not end")
	}
}

func languageUpdate(lang string) frames.Frame {
	if lang == "" {
		return frames.NewTTSUpdateSettingsFrame(&settings.TTS{Language: settings.Cleared[string]()})
	}
	return frames.NewTTSUpdateSettingsFrame(&settings.TTS{Language: settings.Set(lang)})
}

func TestTTSFindsSentencesInItsSettingsLanguage(t *testing.T) {
	svc, _ := newLanguageTTS("de")
	if got := svc.TextAggregationLanguage(); got != "de" {
		t.Fatalf("language = %q, want de", got)
	}
	svc, _ = newLanguageTTS("")
	if got := svc.TextAggregationLanguage(); got != "en" {
		t.Fatalf("language = %q, want English when the settings name none", got)
	}
}

// An update applies to the text that follows, and keeps what is buffered.
func TestTTSLanguageUpdateAppliesToFollowingTextAndPreservesBuffer(t *testing.T) {
	svc, syn := newLanguageTTS("de")
	playLanguage(t, svc,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw."),
		languageUpdate("en"),
		frames.NewLLMTextFrame(" wichtig."),
		frames.NewLLMFullResponseEndFrame(),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
	)
	want := []string{"Das ist bzw. | en", "wichtig. | en", "Das ist bzw. | en", "wichtig. | en"}
	if got := syn.spoken(); !slices.Equal(got, want) {
		t.Fatalf("spoken = %q, want %q", got, want)
	}
}

// With no language the service finds English sentences, and an update applies
// to the generation after it; a language given no value goes back to English.
func TestTTSUnspecifiedLanguageDefaultsToEnglishAndUpdates(t *testing.T) {
	svc, syn := newLanguageTTS("")
	playLanguage(t, svc,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
		languageUpdate("de"),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
		languageUpdate(""),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
	)
	want := []string{
		"Das ist bzw. | en", "wichtig. | en",
		"Das ist bzw. wichtig. | de",
		"Das ist bzw. | en", "wichtig. | en",
	}
	if got := syn.spoken(); !slices.Equal(got, want) {
		t.Fatalf("spoken = %q, want %q", got, want)
	}
}

// An update addressed to another service leaves this one's language alone.
func TestTTSTargetedLanguageUpdateDoesNotChangeOtherService(t *testing.T) {
	svc, syn := newLanguageTTS("de")
	other, _ := newLanguageTTS("en")
	update := frames.NewTTSUpdateSettingsFrame(&settings.TTS{Language: settings.Set("en")})
	update.Service = other
	playLanguage(t, svc,
		update,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
	)
	if got, want := syn.spoken(), []string{"Das ist bzw. wichtig. | de"}; !slices.Equal(got, want) {
		t.Fatalf("spoken = %q, want %q", got, want)
	}
}

// An interruption discards the buffer, and the next generation uses the new
// language.
func TestTTSInterruptionDiscardsBufferAndAllowsNewLanguage(t *testing.T) {
	svc, syn := newLanguageTTS("de")
	playLanguage(t, svc,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw."),
		frames.NewInterruptionFrame(),
		languageUpdate("en"),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Das ist bzw. wichtig."),
		frames.NewLLMFullResponseEndFrame(),
	)
	if got, want := syn.spoken(), []string{"Das ist bzw. | en", "wichtig. | en"}; !slices.Equal(got, want) {
		t.Fatalf("spoken = %q, want %q", got, want)
	}
}
