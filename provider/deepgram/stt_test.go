package deepgram

import (
	"testing"

	"github.com/gojargo/jargo/language"
)

// TestQueryDefaultsMatchPipecat pins the STT query defaults to Pipecat's: the
// opinionated Deepgram params jargo used to force are omitted when unset, while
// endpointing stays set because jargo gates turn detection on speech_final.
func TestQueryDefaultsMatchPipecat(t *testing.T) {
	cfg := Config{
		APIKey:   "k",
		Model:    defaultSTTModel,
		Language: language.EnglishUS,
		Encoding: defaultEncoding,
		Channels: defaultChannels,
	}
	q := cfg.query(16000)

	// Pipecat leaves these unset → Deepgram's own defaults apply.
	for _, key := range []string{"smart_format", "vad_events", "utterance_end_ms"} {
		if q.Has(key) {
			t.Errorf("%s should be omitted by default, got %q", key, q.Get(key))
		}
	}
	// endpointing is jargo's turn-detection gate, so it stays set.
	if got := q.Get("endpointing"); got != "300" {
		t.Errorf("endpointing = %q, want 300", got)
	}
	// These remain on.
	if q.Get("interim_results") != "true" || q.Get("punctuate") != "true" {
		t.Errorf("interim_results=%q punctuate=%q, want both true",
			q.Get("interim_results"), q.Get("punctuate"))
	}
}
