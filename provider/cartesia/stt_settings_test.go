package cartesia

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

func sttConfig() STTConfig {
	return STTConfig{
		APIKey:   "k",
		URL:      defaultSTTURL,
		Version:  defaultSTTVersion,
		Model:    defaultSTTModel,
		Encoding: defaultSTTEncoding,
		Language: language.EnglishUS,
	}
}

// A change reaches the session the only way Cartesia accepts one: through the
// query the next session is opened with.
func TestSTTSettingsChangeReachesTheEndpoint(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}

	before := queryOf(t, c.endpoint(16000))
	if got := before.Get("model"); got != defaultSTTModel {
		t.Fatalf("model = %q, want %q", got, defaultSTTModel)
	}
	if got := before.Get("language"); got != "en" {
		t.Fatalf("language = %q, want en: Cartesia takes the base code", got)
	}

	if _, err := settings.Apply(c.live, &settings.STT{
		Base:     settings.Base{Model: settings.Set("ink-whisper-2")},
		Language: settings.Set("fr"),
	}); err != nil {
		t.Fatalf("apply: %v", err)
	}

	after := queryOf(t, c.endpoint(16000))
	if got := after.Get("model"); got != "ink-whisper-2" {
		t.Errorf("model = %q, want ink-whisper-2", got)
	}
	if got := after.Get("language"); got != "fr" {
		t.Errorf("language = %q, want fr", got)
	}
	// What the caller did not touch is untouched.
	if got := after.Get("encoding"); got != defaultSTTEncoding {
		t.Errorf("encoding = %q, want %q", got, defaultSTTEncoding)
	}
}

func TestSTTAnyChangeAsksForANewSession(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	reopen, err := c.UpdateSettings(context.Background(), settings.Changed{"model": "old"})
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	if !reopen {
		t.Error("a change did not ask for a new session")
	}
}

func TestSTTMetadataFollowsTheSettings(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	if _, err := settings.Apply(c.live, &settings.STT{
		Base: settings.Base{Model: settings.Set("ink-whisper-2")},
	}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if got := c.Metadata().Model; got != "ink-whisper-2" {
		t.Errorf("model = %q, want ink-whisper-2", got)
	}
}

// Cartesia wants the base code, so a regional name resolves to it.
func TestSTTServiceLanguage(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	if got := c.ServiceLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("language = %q, want fr", got)
	}
}

func queryOf(t *testing.T, endpoint string) url.Values {
	t.Helper()
	_, raw, ok := strings.Cut(endpoint, "?")
	if !ok {
		t.Fatalf("endpoint %q carries no query", endpoint)
	}
	q, err := url.ParseQuery(raw)
	if err != nil {
		t.Fatalf("parse query: %v", err)
	}
	return q
}
