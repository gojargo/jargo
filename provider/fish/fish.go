// Package fish is a streaming text-to-speech service backed by Fish Audio. It
// requests PCM output so raw audio streams straight downstream.
package fish

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	defaultBaseURL    = "https://api.fish.audio"
	defaultModel      = "s2.1-pro"
	defaultSampleRate = 24000
)

// Config configures the Fish Audio TTS service.
type Config struct {
	// APIKey is the Fish Audio API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted API.
	BaseURL string
	// Model is the backend model, sent as the "model" request header; empty uses
	// a current default.
	Model string
	// ReferenceID selects a voice model (a library or cloned voice). Recommended;
	// empty uses the model's default voice.
	ReferenceID string
	// SampleRate is the PCM output rate; empty uses 24 kHz. Supported: 8000,
	// 16000, 24000, 32000, 44100.
	SampleRate int
	// Speed scales the speaking rate; nil leaves it at the default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Fish Audio TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("FishTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	payload := map[string]any{
		"text":        text,
		"format":      "pcm",
		"sample_rate": s.cfg.SampleRate,
	}
	if s.cfg.ReferenceID != "" {
		payload["reference_id"] = s.cfg.ReferenceID
	}
	if s.cfg.Speed != nil {
		payload["prosody"] = map[string]any{"speed": *s.cfg.Speed}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/v1/tts", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("model", s.cfg.Model) // Fish selects the backend via this header
	return tts.StreamResponse(s.http, req, emit)
}
