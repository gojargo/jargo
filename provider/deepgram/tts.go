package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// speakPath is the synthesis route appended to the configured base URL.
	speakPath = "/v1/speak"
	// defaultTTSModel is a current Aura-2 English voice.
	defaultTTSModel = "aura-2-thalia-en"
	// defaultTTSEncoding is the audio encoding jargo requests from Aura.
	defaultTTSEncoding = "linear16"
)

// TTSConfig configures the Aura TTS service.
type TTSConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Deepgram host, for a private or air-gapped
	// deployment. It is taken as given, scheme included; empty uses Deepgram's
	// hosted API.
	BaseURL string
	// Model is the Aura voice model; empty uses a default.
	Model string
	// SampleRate is the PCM rate requested from Aura and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "linear16".
	Encoding string
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Deepgram Aura TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultTTSEncoding
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://" + defaultHost
	}
	return tts.New("DeepgramTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// Metadata reports the Aura voice model synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model}
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
// container=none is required to receive headerless PCM rather than a WAV stream.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	q := url.Values{}
	q.Set("model", s.cfg.Model)
	q.Set("encoding", s.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	q.Set("container", "none")

	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.cfg.BaseURL+speakPath+"?"+q.Encode(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", authToken(s.cfg.APIKey))
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}
