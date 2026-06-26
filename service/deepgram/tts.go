package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"

	"github.com/gojargo/jargo/service/tts"
)

const (
	speakURL = "https://api.deepgram.com/v1/speak"
	// defaultTTSModel is a current Aura-2 English voice.
	defaultTTSModel = "aura-2-thalia-en"
	// defaultTTSSampleRate is the PCM rate jargo requests from Aura.
	defaultTTSSampleRate = 24000
	// defaultTTSEncoding is the audio encoding jargo requests from Aura.
	defaultTTSEncoding = "linear16"
)

// TTSConfig configures the Aura TTS service.
type TTSConfig struct {
	// APIKey is the Deepgram API key; empty uses the DEEPGRAM_API_KEY env var.
	APIKey string
	// Model is the Aura voice model; empty uses a default.
	Model string
	// SampleRate is the PCM rate requested from Aura and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "linear16".
	Encoding string
}

// NewTTS builds a Deepgram Aura TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultTTSEncoding
	}
	return tts.New("DeepgramTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
// container=none is required to receive headerless PCM rather than a WAV stream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	q := url.Values{}
	q.Set("model", s.cfg.Model)
	q.Set("encoding", s.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	q.Set("container", "none")

	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, speakURL+"?"+q.Encode(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}
