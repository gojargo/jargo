package deepgram

import (
	"testing"

	"github.com/gojargo/jargo/language"
)

// TestQueryDefaults checks the opinionated Deepgram params are omitted when
// unset, so Deepgram's own defaults apply.
func TestQueryDefaults(t *testing.T) {
	cfg := Config{
		APIKey:   "k",
		Model:    defaultSTTModel,
		Language: language.EnglishUS,
		Encoding: defaultEncoding,
		Channels: defaultChannels,
	}
	q := cfg.query(16000, newSettings(cfg))

	// Omitted when unset → Deepgram's own defaults apply.
	for _, key := range []string{"smart_format", "vad_events", "utterance_end_ms", "endpointing"} {
		if q.Has(key) {
			t.Errorf("%s should be omitted by default, got %q", key, q.Get(key))
		}
	}
	// These remain on.
	if q.Get("interim_results") != "true" || q.Get("punctuate") != "true" {
		t.Errorf("interim_results=%q punctuate=%q, want both true",
			q.Get("interim_results"), q.Get("punctuate"))
	}
}
