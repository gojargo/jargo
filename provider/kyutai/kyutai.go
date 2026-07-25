// Package kyutai provides streaming speech-to-text and text-to-speech backed by
// a self-hosted Kyutai moshi-server (the Delayed Streams Modeling STT and TTS
// models). Both talk MessagePack over a WebSocket: STT streams 24 kHz float32
// PCM up and receives word and semantic-VAD messages back; TTS streams words up
// and receives 24 kHz float32 PCM back.
//
// The STT service emits cumulative interim transcripts as words arrive and a
// single finalized end-of-turn transcript when moshi's semantic VAD predicts a
// pause, so it works whether the pipeline runs its own turn detection or leans
// on that end-of-turn signal.
package kyutai

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// moshiSampleRate is the PCM rate the moshi-server models run at, for both the
// audio sent to STT and the audio received from TTS.
const moshiSampleRate = 24000

// defaultToken is moshi-server's default shared API token, sent as the
// kyutai-api-key header. Deployments that set a real token override it via the
// APIKey config field.
const defaultToken = "public_token"

// msgTypeKey is the moshi message discriminator field, used as the map key when
// framing outbound audio and text messages.
const msgTypeKey = "type"

// Config configures the Kyutai STT service.
type Config struct {
	// APIKey is moshi-server's shared token (sent as the kyutai-api-key header);
	// empty uses moshi's default "public_token".
	APIKey string
	// URL overrides the moshi-server ASR WebSocket endpoint; empty uses localhost.
	URL string
	// SampleRate is the input PCM rate from the pipeline; 0 uses the transport's
	// rate. Audio is resampled from this rate to the 24 kHz moshi expects.
	SampleRate int
	// Language is informational; the model itself is fixed (e.g. en_fr).
	Language language.Language
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }
