package soniox

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/text"
)

// TestTTSConfigValidate pins which TTSConfig fields the provider requires and
// the ranges Soniox accepts.
func TestTTSConfigValidate(t *testing.T) {
	slow, tooSlow := 0.7, 0.6
	fast, tooFast := 1.3, 1.4
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "supported sample rate", Cfg: TTSConfig{APIKey: "k", SampleRate: 44100}, Valid: true},
		{Name: "unsupported sample rate", Cfg: TTSConfig{APIKey: "k", SampleRate: 22050}, Valid: false},
		{Name: "slowest speed", Cfg: TTSConfig{APIKey: "k", Speed: &slow}, Valid: true},
		{Name: "below the slowest speed", Cfg: TTSConfig{APIKey: "k", Speed: &tooSlow}, Valid: false},
		{Name: "fastest speed", Cfg: TTSConfig{APIKey: "k", Speed: &fast}, Valid: true},
		{Name: "above the fastest speed", Cfg: TTSConfig{APIKey: "k", Speed: &tooFast}, Valid: false},
	})
}

// TestNewTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTTS(t *testing.T) {
	providertest.Service(t, "SonioxTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestTTSConfigMessage checks the configuration that opens a stream.
func TestTTSConfigMessage(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
		cfg := s.buildConfigMsg("ctx-1")

		want := map[string]any{
			keyStreamID:         "ctx-1",
			"model":             defaultTTSModel,
			"voice":             defaultVoice,
			"language":          "en",
			"audio_format":      defaultTTSAudioFormat,
			"sample_rate":       defaultTTSSampleRate,
			"return_timestamps": true,
		}
		if len(cfg) != len(want) {
			t.Errorf("config = %v, want exactly %v", cfg, want)
		}
		for key, val := range want {
			if cfg[key] != val {
				t.Errorf("%s = %v, want %v", key, cfg[key], val)
			}
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		speed, reduce, off := 1.1, true, false
		// Built through the service, which names the language the Soniox way.
		_, s := newTTS(TTSConfig{
			APIKey:            "k",
			Voice:             "9f2b0c1e",
			Language:          language.FrenchCA,
			Speed:             &speed,
			ReduceSilence:     &reduce,
			ClientReferenceID: "call-7",
			WordTimestamps:    &off,
		})
		cfg := s.buildConfigMsg("ctx-1")

		for key, val := range map[string]any{
			"language":            "fr",
			"speed":               speed,
			"reduce_silence":      true,
			"voice":               "9f2b0c1e",
			"return_timestamps":   false,
			"client_reference_id": "call-7",
		} {
			if cfg[key] != val {
				t.Errorf("%s = %v, want %v", key, cfg[key], val)
			}
		}
	})

	t.Run("a format that is not PCM", func(t *testing.T) {
		s := newTTSSynthesizer(TTSConfig{APIKey: "k", AudioFormat: "mp3"})
		cfg := s.buildConfigMsg("ctx-1")
		if cfg["audio_format"] != "mp3" {
			t.Errorf("audio_format = %v, want mp3", cfg["audio_format"])
		}
		if _, ok := cfg["sample_rate"]; ok {
			t.Error("sample_rate was sent with a format that is not raw PCM")
		}
	})
}

// The language the service is built with is stored as the caller gave it and
// named the Soniox way once, as the service is built, the way an update's is.
func TestTTSStartingLanguageIsNamedOnce(t *testing.T) {
	raw := newTTSSynthesizer(TTSConfig{APIKey: "k", Language: language.NorwegianBokmalNO})
	if got := raw.live.Language.Or(""); got != "nb-NO" {
		t.Errorf("stored language before the service names it = %q, want nb-NO as given", got)
	}
	_, s := newTTS(TTSConfig{APIKey: "k", Language: language.NorwegianBokmalNO})
	if got := s.live.Language.Or(""); got != "no" {
		t.Errorf("language = %q, want no", got)
	}
}

// TestSonioxTTSLanguage checks the codes Soniox takes: the base code, with
// Norwegian Bokmal sent as Norwegian.
func TestSonioxTTSLanguage(t *testing.T) {
	for in, want := range map[language.Language]string{
		language.NorwegianBokmal:   "no",
		language.NorwegianBokmalNO: "no",
		language.FrenchCA:          "fr",
		language.ChineseCN:         "zh",
		language.English:           "en",
	} {
		if got := sonioxTTSLanguage(in); got != want {
			t.Errorf("sonioxTTSLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestSpacelessLanguage checks which languages get per-character word timings.
func TestSpacelessLanguage(t *testing.T) {
	cases := map[language.Language]bool{
		language.Chinese:   true,
		language.ChineseTW: true,
		language.Japanese:  true,
		language.English:   false,
		language.Korean:    false, // Korean does use spaces between words
		"":                 false,
	}
	for lang, want := range cases {
		s := newTTSSynthesizer(TTSConfig{Language: lang})
		if got := s.spacelessLanguage(); got != want {
			t.Errorf("spacelessLanguage(%q) = %v, want %v", lang, got, want)
		}
	}
}

// TestTTSSynthesizerDescription checks the rate it emits and the model and
// voice the synthesis is billed against.
func TestTTSSynthesizerDescription(t *testing.T) {
	s := newTTSSynthesizer(TTSConfig{APIKey: "k", SampleRate: 16000})
	if got := s.SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want 16000", got)
	}
	meta := s.Metadata()
	if meta.Model != defaultTTSModel || meta.VoiceID != defaultVoice {
		t.Errorf("Metadata() = %+v, want the default model and voice", meta)
	}
}

// TestNewTTSWordTimestampPath checks the word-aligned path is on by default and
// can be turned off.
func TestNewTTSWordTimestampPath(t *testing.T) {
	off := false
	if !(TTSConfig{APIKey: "k"}).wordTimestamps() {
		t.Error("wordTimestamps() = false, want it on by default")
	}
	if (TTSConfig{APIKey: "k", WordTimestamps: &off}).wordTimestamps() {
		t.Error("wordTimestamps() = true, want it off when explicitly disabled")
	}
}

// timings is text as Soniox times it, a character every tenth of a second from
// start.
func timings(s string, start float64) *ttsTimings {
	t := &ttsTimings{}
	for i, r := range []rune(s) {
		t.Characters = append(t.Characters, string(r))
		t.Starts = append(t.Starts, float64(int((start+float64(i)*0.1)*10+0.5))/10)
	}
	return t
}

func synthIn(lang language.Language) *ttsSynthesizer {
	return newTTSSynthesizer(TTSConfig{APIKey: "k", Language: lang})
}

func TestToWordTimesAssemblesWords(t *testing.T) {
	s := synthIn(language.English)
	want := []uctx.WordTiming{{Word: "Hi", Offset: 0}, {Word: "world.", Offset: 0.3}}
	if got := s.toWordTimes("s1", timings("Hi world. ", 0)); !slices.Equal(got, want) {
		t.Errorf("words = %v, want %v", got, want)
	}
}

func TestToWordTimesCarriesAPartialWordAcrossMessages(t *testing.T) {
	s := synthIn(language.English)
	if got := s.toWordTimes("s1", timings("Hel", 0)); len(got) != 0 {
		t.Errorf("words = %v, want none before the word ends", got)
	}
	want := []uctx.WordTiming{{Word: "Hello", Offset: 0}, {Word: "you", Offset: 0.6}}
	if got := s.toWordTimes("s1", timings("lo you ", 0.3)); !slices.Equal(got, want) {
		t.Errorf("words = %v, want %v", got, want)
	}
}

// The last word has no space to end it, so it waits for the stream to
// terminate.
func TestToWordTimesBuffersTheFinalWord(t *testing.T) {
	s := synthIn(language.English)
	want := []uctx.WordTiming{{Word: "Hi", Offset: 0}}
	if got := s.toWordTimes("s1", timings("Hi you", 0)); !slices.Equal(got, want) {
		t.Errorf("words = %v, want %v", got, want)
	}
	if p := s.partials["s1"]; p != (partialWord{word: "you", start: 0.3}) {
		t.Errorf("partial = %+v, want you at 0.3", p)
	}
}

func TestToWordTimesKeepsStreamsApart(t *testing.T) {
	s := synthIn(language.English)
	s.toWordTimes("s1", timings("Hel", 0))
	s.toWordTimes("s2", timings("wor", 1.0))
	if p := s.partials["s1"]; p != (partialWord{word: "Hel", start: 0}) {
		t.Errorf("s1 partial = %+v", p)
	}
	if p := s.partials["s2"]; p != (partialWord{word: "wor", start: 1.0}) {
		t.Errorf("s2 partial = %+v", p)
	}
	if got := s.toWordTimes("s1", timings("lo ", 0.3)); !slices.Equal(got, []uctx.WordTiming{{Word: "Hello", Offset: 0}}) {
		t.Errorf("s1 words = %v", got)
	}
	got := s.toWordTimes("s2", timings("ld ", 1.3))
	if !slices.Equal(got, []uctx.WordTiming{{Word: "world", Offset: 1.0}}) {
		t.Errorf("s2 words = %v", got)
	}
}

func TestToWordTimesMismatchYieldsNothing(t *testing.T) {
	s := synthIn(language.English)
	got := s.toWordTimes("s1", &ttsTimings{Characters: []string{"H", "i"}, Starts: []float64{0}})
	if len(got) != 0 {
		t.Errorf("words = %v, want none", got)
	}
	if _, ok := s.partials["s1"]; ok {
		t.Error("a mismatched message left a partial word")
	}
}

func TestToWordTimesSpacelessLanguagesPerCharacter(t *testing.T) {
	for _, tc := range []struct {
		lang language.Language
		text string
		want []uctx.WordTiming
	}{
		{language.Japanese, "こんにちは、私", []uctx.WordTiming{
			{Word: "こ", Offset: 0},
			{Word: "ん", Offset: 0.1},
			{Word: "に", Offset: 0.2},
			{Word: "ち", Offset: 0.3},
			{Word: "は", Offset: 0.4},
			{Word: "私", Offset: 0.6},
		}},
		{language.Chinese, "你好，世界。", []uctx.WordTiming{
			{Word: "你", Offset: 0}, {Word: "好", Offset: 0.1}, {Word: "世", Offset: 0.3}, {Word: "界", Offset: 0.4},
		}},
	} {
		s := synthIn(tc.lang)
		if got := s.toWordTimes("s1", timings(tc.text, 0)); !slices.Equal(got, tc.want) {
			t.Errorf("%s: words = %v, want %v", tc.lang, got, tc.want)
		}
		if _, ok := s.partials["s1"]; ok {
			t.Errorf("%s: a partial word was kept", tc.lang)
		}
	}
}

// Japanese tokens carry their own spacing, so joining them adds none.
func TestSpacelessTokensConcatenateWithoutSpaces(t *testing.T) {
	s := synthIn(language.Japanese)
	var parts []text.Part
	for _, w := range s.toWordTimes("s1", timings("こんにちは、私", 0)) {
		parts = append(parts, text.Part{Text: w.Word, IncludesInterPartSpaces: s.spacelessLanguage()})
	}
	if got := text.Concatenate(parts); got != "こんにちは私" {
		t.Errorf("joined = %q, want the characters with nothing between them", got)
	}
}

// The punctuation the per-character tokens leave out is recovered by the word
// tracker, which matches on the alphanumerics and commits spans of the
// original text.
func TestSpacelessPunctuationRecoveredByTheWordTracker(t *testing.T) {
	const sentence = "こんにちは、私はAIです。"
	s := synthIn(language.Japanese)
	tracker := uctx.NewWordCompletionTracker(sentence, "", "")
	complete := false
	for _, w := range s.toWordTimes("s1", timings(sentence, 0)) {
		complete = tracker.AddWord(w.Word)
	}
	if !complete {
		t.Error("the tracker did not complete on the tokens")
	}
	if got := tracker.AccumulatedUserFacingText(); got != sentence {
		t.Errorf("accumulated = %q, want %q", got, sentence)
	}
}
