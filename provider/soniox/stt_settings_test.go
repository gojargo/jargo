package soniox

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

func sttConfig() Config {
	return Config{APIKey: "k", URL: defaultURL, Model: defaultModel, Language: language.EnglishUS}
}

func handshake(t *testing.T, c *connector) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal(c.config(16000), &got); err != nil {
		t.Fatalf("handshake is not valid JSON: %v", err)
	}
	return got
}

// A change reaches the session the only way Soniox accepts one: through the
// handshake that opens the next session.
func TestSTTSettingsChangeReachesTheHandshake(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}

	before := handshake(t, c)
	if got := before["model"]; got != defaultModel {
		t.Fatalf("model = %v, want %q", got, defaultModel)
	}
	if hints, ok := before["language_hints"].([]any); !ok || len(hints) != 1 || hints[0] != "en" {
		t.Fatalf("language_hints = %v, want [en]", before["language_hints"])
	}

	if _, err := settings.Apply(c.live, &Settings{
		Model:                    settings.Set("stt-rt-preview-v2"),
		LanguageHints:            settings.Set([]string{"fr"}),
		EnableSpeakerDiarization: settings.Set(true),
		EndpointSensitivity:      settings.Set(0.6),
		ClientReferenceID:        settings.Set("call-7"),
	}); err != nil {
		t.Fatalf("apply: %v", err)
	}

	after := handshake(t, c)
	if got := after["model"]; got != "stt-rt-preview-v2" {
		t.Errorf("model = %v, want stt-rt-preview-v2", got)
	}
	if hints, ok := after["language_hints"].([]any); !ok || len(hints) != 1 || hints[0] != "fr" {
		t.Errorf("language_hints = %v, want [fr]", after["language_hints"])
	}
	if got := after["enable_speaker_diarization"]; got != true {
		t.Errorf("enable_speaker_diarization = %v, want true", got)
	}
	if got := after["endpoint_sensitivity"]; got != 0.6 {
		t.Errorf("endpoint_sensitivity = %v, want 0.6", got)
	}
	if got := after["client_reference_id"]; got != "call-7" {
		t.Errorf("client_reference_id = %v, want call-7", got)
	}
	// What the caller did not touch is untouched.
	if got := after["audio_format"]; got != "pcm_s16le" {
		t.Errorf("audio_format = %v, want pcm_s16le", got)
	}
}

// A setting nobody gave is left out of the handshake, so Soniox applies its own
// default rather than being sent a zero that means something else.
func TestUnsetSettingsAreLeftOutOfTheHandshake(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	got := handshake(t, c)

	for _, key := range []string{
		"language_hints_strict", "context", "enable_speaker_diarization",
		"enable_language_identification", "max_endpoint_delay_ms",
		"endpoint_sensitivity", "endpoint_latency_adjustment_level",
		"client_reference_id",
	} {
		if _, present := got[key]; present {
			t.Errorf("%s was sent though nobody set it: %v", key, got[key])
		}
	}
}

// What the service was built with seeds the settings, so a configured option
// reaches the first handshake.
func TestConfiguredOptionsSeedTheSettings(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	strict := true
	delay := 900
	cfg.LanguageHintsStrict = &strict
	cfg.MaxEndpointDelayMs = &delay
	cfg.ClientReferenceID = "call-1"
	cfg.Context = &Context{Text: "ordering coffee", Terms: []string{"flat white"}}

	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	got := handshake(t, c)

	if got["language_hints_strict"] != true {
		t.Errorf("language_hints_strict = %v, want true", got["language_hints_strict"])
	}
	if got["max_endpoint_delay_ms"] != float64(900) {
		t.Errorf("max_endpoint_delay_ms = %v, want 900", got["max_endpoint_delay_ms"])
	}
	if got["client_reference_id"] != "call-1" {
		t.Errorf("client_reference_id = %v, want call-1", got["client_reference_id"])
	}
	ctx, ok := got["context"].(map[string]any)
	if !ok || ctx["text"] != "ordering coffee" {
		t.Errorf("context = %v, want the configured context", got["context"])
	}
}

func TestSTTAnyChangeAsksForANewSession(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	reopen, err := c.UpdateSettings(context.Background(), settings.Changed{"model": "old"})
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	if !reopen {
		t.Error("a change did not ask for a new session")
	}
}

// Ported from upstream (test_norwegian_bokmal_maps_to_soniox_norwegian). Soniox
// takes Bokmål as Norwegian, under "no", whether it is named as a language
// update or as the language the service was built with.
func TestNorwegianBokmalMapsToSonioxNorwegian(t *testing.T) {
	t.Parallel()

	for _, l := range []language.Language{language.NorwegianBokmal, language.NorwegianBokmalNO} {
		cfg := sttConfig()
		cfg.Language = l
		c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
		if got := c.ServiceLanguage(l); got != "no" {
			t.Errorf("%s: language = %q, want no", l, got)
		}
		if hints := handshake(t, c)["language_hints"]; !slices.Equal(anyStrings(hints), []string{"no"}) {
			t.Errorf("%s: language_hints = %v, want [no]", l, hints)
		}
	}
}

// A language update replaces the hints, since the hints are what Soniox reads,
// and leaves the language itself unstored. Hints given in the same update win.
func TestLanguageUpdateReplacesTheHints(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	changed, err := settings.Apply(c.live, &settings.STT{Language: settings.Set("sv")})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if !changed.Has("language_hints") || changed.Has("language") {
		t.Errorf("changed = %v, want language_hints alone", changed)
	}
	if hints, _ := c.live.LanguageHints.Value(); !slices.Equal(hints, []string{"sv"}) {
		t.Errorf("hints = %v, want [sv]", hints)
	}
	if _, ok := c.live.Language.Value(); ok {
		t.Errorf("language = %v, want it left unstored", c.live.Language)
	}

	both := &Settings{LanguageHints: settings.Set([]string{"fi", "sv"})}
	both.Language = settings.Set("fi")
	if _, err := settings.Apply(c.live, both); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if hints, _ := c.live.LanguageHints.Value(); !slices.Equal(hints, []string{"fi", "sv"}) {
		t.Errorf("hints = %v, want the [fi sv] given alongside the language", hints)
	}
}

// Ported from upstream (test_language_hints_given_with_a_language_are_kept).
// Hints given alongside the language the service is built with are the
// hints, rather than being replaced by that language.
func TestLanguageHintsGivenWithALanguageAreKept(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	cfg.Language = language.Finnish
	cfg.LanguageHints = []language.Language{language.Finnish, language.Swedish}
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	if hints, _ := c.live.LanguageHints.Value(); !slices.Equal(hints, []string{"fi", "sv"}) {
		t.Errorf("hints = %v, want [fi sv]", hints)
	}
}

// Ported from upstream (the test_language_from_tokens cases). A transcript is
// labeled with the language most of its tokens were identified as, the first
// one on a tie. Tokens naming no language, or one the catalog does not know,
// do not count.
func TestLanguageFromTokens(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		tokens []token
		want   string
	}{
		"single recognized language": {
			tokens: []token{{Text: "Hello", Language: "en"}, {Text: " world", Language: "en"}},
			want:   "en",
		},
		"most common language": {
			tokens: []token{{Text: "Ik", Language: "nl"}, {Text: " zoek", Language: "nl"}, {Text: " computer", Language: "en"}},
			want:   "nl",
		},
		"skips unknown language": {
			tokens: []token{{Text: "Hello", Language: "en"}, {Text: "!", Language: "klingon"}},
			want:   "en",
		},
		"skips missing language": {
			tokens: []token{{Text: "Hello", Language: "en"}, {Text: " wereld"}},
			want:   "en",
		},
		"ignores unknown and missing languages": {
			tokens: []token{{Text: "Hello", Language: "klingon"}, {Text: " world"}, {Text: "!"}},
			want:   "",
		},
		"uses first language on tie": {
			tokens: []token{{Text: "Hello", Language: "en"}, {Text: " wereld", Language: "nl"}},
			want:   "en",
		},
	} {
		if got := languageFromTokens(tc.tokens); got != tc.want {
			t.Errorf("%s: language = %q, want %q", name, got, tc.want)
		}
	}
}

// Soniox wants the base code, so a regional name resolves to it.
func TestSTTServiceLanguage(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	if got := c.ServiceLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("language = %q, want fr", got)
	}
}
