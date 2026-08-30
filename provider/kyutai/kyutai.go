// Package kyutai provides Kyutai's self-hosted speech services. Kyutai
// publishes several models; the ones served over a network are here:
//
//   - Kyutai STT and Kyutai TTS, the Delayed Streams Modeling models, served by
//     moshi-server. [NewSTT] and [NewTTS].
//   - Pocket TTS, the small CPU-only model, served by pocket-tts.
//     [NewPocketTTS].
//
// They are separate servers speaking separate protocols, so which one a bot
// speaks with is which constructor it calls. Moshi itself, the full-duplex
// speech-to-speech model the server is named after, is a different thing again
// and is not provided here.
//
// The moshi-server models talk MessagePack over a WebSocket: STT streams 24 kHz
// float32 PCM up and receives word and semantic-VAD messages back; TTS streams
// words up and receives 24 kHz float32 PCM back. The STT service emits
// cumulative interim transcripts as words arrive and a single finalized
// end-of-turn transcript when the server's semantic VAD predicts a pause, so it
// works whether the pipeline runs its own turn detection or leans on that
// end-of-turn signal.
package kyutai

import (
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// defaultSampleRate is the PCM rate the moshi-server models run at, for both the
// audio sent to STT and the audio received from TTS.
const defaultSampleRate = 24000

// defaultToken is moshi-server's default shared API token, sent as the
// kyutai-api-key header. Deployments that set a real token override it via the
// APIKey config field.
const defaultToken = "public_token"

// msgTypeKey is the server's message discriminator field, used as the map key when
// framing outbound audio and text messages.
const msgTypeKey = "type"

// Config configures the Kyutai STT service.
type Config struct {
	// APIKey is moshi-server's shared token (sent as the kyutai-api-key header);
	// empty uses the server's default "public_token".
	APIKey string
	// URL overrides the moshi-server ASR WebSocket endpoint; empty uses localhost.
	URL string
	// SampleRate is the input PCM rate from the pipeline; 0 uses the transport's
	// rate. Audio is resampled from this rate to the 24 kHz the server expects.
	SampleRate int
	// Language is informational; the model itself is fixed (e.g. en_fr).
	Language language.Language

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.DefaultTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }
