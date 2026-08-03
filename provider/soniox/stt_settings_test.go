package soniox

import (
	"context"
	"encoding/json"
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

	if _, err := settings.Apply(c.live, &settings.STT{
		Base:     settings.Base{Model: settings.Set("stt-rt-preview-v2")},
		Language: settings.Set("fr"),
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
	// What the caller did not touch is untouched.
	if got := after["audio_format"]; got != "s16le" {
		t.Errorf("audio_format = %v, want s16le", got)
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

// Soniox wants the base code, so a regional name resolves to it.
func TestSTTServiceLanguage(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	if got := c.ServiceLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("language = %q, want fr", got)
	}
}
