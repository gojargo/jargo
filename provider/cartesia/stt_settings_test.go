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
	c := newSTTConnector(cfg)

	before := queryOf(t, c.endpoint(16000))
	if got := before.Get("model"); got != defaultSTTModel {
		t.Fatalf("model = %q, want %q", got, defaultSTTModel)
	}
	if got := before.Get("language"); got != "en" {
		t.Fatalf("language = %q, want en: Cartesia takes the base code", got)
	}

	if _, err := settings.Apply(c.live, &STTSettings{
		Model:    settings.Set("ink-whisper-2"),
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
	c := newSTTConnector(cfg)
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
	c := newSTTConnector(cfg)
	if _, err := settings.Apply(c.live, &STTSettings{
		Model: settings.Set("ink-whisper-2"),
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
	c := newSTTConnector(cfg)
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

// Keyterms go out as repeated parameters, with spaces percent-encoded: Cartesia
// expects %20 inside a keyterm, and Go's own encoding writes a space as a plus.
func TestKeytermsReachTheEndpoint(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	cfg.Model = "ink-2-preview"
	cfg.Keyterm = []string{"flat white", " ", "oat milk"}
	c := newSTTConnector(cfg)

	endpoint := c.endpoint(16000)
	if strings.Contains(endpoint, "+") {
		t.Errorf("endpoint encodes a space as a plus: %s", endpoint)
	}
	q := queryOf(t, endpoint)
	got := q["keyterm"]
	if len(got) != 2 || got[0] != "flat white" || got[1] != "oat milk" {
		t.Errorf("keyterm = %v, want [flat white, oat milk]: blanks are dropped", got)
	}
}

// Only the ink-2 family honors keyterms, so they are left off any other model
// rather than being sent where they mean nothing.
func TestKeytermsAreLeftOffAnUnsupportedModel(t *testing.T) {
	t.Parallel()

	cfg := sttConfig()
	cfg.Model = defaultSTTModel
	cfg.Keyterm = []string{"flat white"}
	c := newSTTConnector(cfg)

	if got := queryOf(t, c.endpoint(16000))["keyterm"]; len(got) != 0 {
		t.Errorf("keyterm = %v, want none on model %q", got, defaultSTTModel)
	}
}

// Cartesia caps a connection at 100 keyterms totaling 1200 characters, so an
// oversized list is truncated rather than left to fail the connection.
func TestKeytermsAreCapped(t *testing.T) {
	t.Parallel()

	many := make([]string, 150)
	for i := range many {
		many[i] = "term"
	}
	if got := len(prepareKeyterms(many)); got != maxKeyterms {
		t.Errorf("kept %d keyterms, want %d", got, maxKeyterms)
	}

	long := []string{strings.Repeat("a", maxKeytermChars), "dropped"}
	if got := prepareKeyterms(long); len(got) != 1 {
		t.Errorf("kept %d keyterms, want 1: the second exceeds the character cap", len(got))
	}
}

// The turn-detecting service reopens for a keyterm change and says so for
// anything else, since Cartesia binds keyterms to a connection and the rest of
// what it is told at the session cannot change on one already running.
func TestTurnsKeytermChangeReopensAndTheRestIsReported(t *testing.T) {
	t.Parallel()

	cfg := TurnsSTTConfig{APIKey: "k", URL: defaultTurnsURL, Version: defaultVersion, Model: defaultTurnsModel}
	cfg.Keyterm = []string{"flat white"}
	c := newTurnsConnector(cfg)

	reopen, err := c.UpdateSettings(context.Background(), settings.Changed{"keyterm": nil})
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	if !reopen {
		t.Error("a keyterm change did not ask for a new session")
	}

	reopen, err = c.UpdateSettings(context.Background(), settings.Changed{"model": "old"})
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	if reopen {
		t.Error("a model change asked for a new session, but it cannot take effect")
	}
}
