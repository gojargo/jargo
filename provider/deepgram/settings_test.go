package deepgram

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

func baseConfig() Config {
	return Config{
		APIKey:   "k",
		Model:    defaultSTTModel,
		Language: language.EnglishUS,
		Encoding: defaultEncoding,
		Channels: defaultChannels,
	}
}

// A change reaches the session the only way Deepgram accepts one: through the
// query the next session is opened with.
func TestSettingsChangeReachesTheQuery(t *testing.T) {
	t.Parallel()

	cfg := baseConfig()
	live := newSettings(cfg)

	if got := live.Model.Or(""); got != defaultSTTModel {
		t.Fatalf("model = %q, want %q", got, defaultSTTModel)
	}

	changed, err := settings.Apply(live, &Settings{
		STT:     settings.STT{Base: settings.Base{Model: settings.Set("nova-2")}},
		Diarize: settings.Set(true),
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if !changed.Has("model") || !changed.Has("diarize") {
		t.Fatalf("changed = %v, want model and diarize", changed)
	}

	q := cfg.query(16000, live)
	if got := q.Get("model"); got != "nova-2" {
		t.Errorf("model = %q, want nova-2", got)
	}
	if got := q.Get("diarize"); got != "true" {
		t.Errorf("diarize = %q, want true", got)
	}
	// Everything the caller did not touch is still what it was.
	if got := q.Get("language"); got != "en-US" {
		t.Errorf("language = %q, want en-US", got)
	}
	if got := q.Get("punctuate"); got != "true" {
		t.Errorf("punctuate = %q, want true", got)
	}
}

// Deepgram takes all of this when the session opens, so any change needs a new
// one.
func TestAnyChangeAsksForANewSession(t *testing.T) {
	t.Parallel()

	c := &connector{cfg: baseConfig(), live: newSettings(baseConfig())}
	reopen, err := c.UpdateSettings(context.Background(), settings.Changed{"diarize": false})
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	if !reopen {
		t.Error("a change did not ask for a new session")
	}
}

// The model reported for billing follows the settings, not what the service was
// built with.
func TestMetadataFollowsTheSettings(t *testing.T) {
	t.Parallel()

	c := &connector{cfg: baseConfig(), live: newSettings(baseConfig())}
	if got := c.Metadata().Model; got != defaultSTTModel {
		t.Fatalf("model = %q, want %q", got, defaultSTTModel)
	}

	if _, err := settings.Apply(c.live, &Settings{
		STT: settings.STT{Base: settings.Base{Model: settings.Set("nova-2")}},
	}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if got := c.Metadata().Model; got != "nova-2" {
		t.Errorf("model = %q, want nova-2", got)
	}
}

// Deepgram uses BCP-47 codes directly, so a canonical code passes through.
func TestServiceLanguage(t *testing.T) {
	t.Parallel()

	c := &connector{cfg: baseConfig(), live: newSettings(baseConfig())}
	if got := c.ServiceLanguage(language.FrenchCA); got != "fr-CA" {
		t.Errorf("language = %q, want fr-CA", got)
	}
}

// A list setting replaces what was there rather than adding to it.
func TestListSettingsReplace(t *testing.T) {
	t.Parallel()

	cfg := baseConfig()
	cfg.Keyterm = []string{"one"}
	live := newSettings(cfg)

	if _, err := settings.Apply(live, &Settings{Keyterm: settings.Set([]string{"two", "three"})}); err != nil {
		t.Fatalf("apply: %v", err)
	}

	q := cfg.query(16000, live)
	got := q["keyterm"]
	if len(got) != 2 || got[0] != "two" || got[1] != "three" {
		t.Errorf("keyterm = %v, want [two three]", got)
	}
}
