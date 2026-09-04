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
	if got := after["audio_format"]; got != "s16le" {
		t.Errorf("audio_format = %v, want s16le", got)
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

// Soniox wants the base code, so a regional name resolves to it.
func TestSTTServiceLanguage(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	if got := c.ServiceLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("language = %q, want fr", got)
	}
}
