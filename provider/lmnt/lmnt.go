// Package lmnt is a streaming text-to-speech service backed by LMNT's HTTP
// "speech bytes" endpoint, which streams raw PCM.
package lmnt

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	apiURL       = "https://api.lmnt.com/v1/ai/speech/bytes"
	defaultModel = "blizzard"
	defaultVoice = "leah"
	defaultLang  = "en"
	// defaultSampleRate is the PCM rate jargo requests from LMNT.
	defaultSampleRate = 24000
)

// Config configures the LMNT TTS service.
type Config struct {
	// APIKey is the LMNT API key. Required.
	APIKey string `validate:"required"`
	// Model is the LMNT model; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default.
	Voice string
	// Lang is the spoken language; empty uses "en".
	Lang string
	// SampleRate is the PCM rate requested from LMNT and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds an LMNT TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.Lang == "" {
		cfg.Lang = defaultLang
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("LMNTTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := json.Marshal(map[string]any{
		"text":        text,
		"voice":       s.cfg.Voice,
		"model":       s.cfg.Model,
		"language":    s.cfg.Lang,
		"format":      "raw",
		"sample_rate": s.cfg.SampleRate,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("X-API-Key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}
