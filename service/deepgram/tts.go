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
	// ttsSampleRate is the PCM rate jargo requests from Aura.
	ttsSampleRate = 24000
)

// TTSConfig configures the Aura TTS service.
type TTSConfig struct {
	// APIKey is the Deepgram API key; empty uses the DEEPGRAM_API_KEY env var.
	APIKey string
	// Model is the Aura voice model; empty uses a default.
	Model string
}

// NewTTS builds a Deepgram Aura TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	return tts.New("DeepgramTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return ttsSampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
// container=none is required to receive headerless PCM rather than a WAV stream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	q := url.Values{}
	q.Set("model", s.cfg.Model)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(ttsSampleRate))
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
