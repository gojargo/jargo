// Package rime is a streaming text-to-speech service backed by Rime's HTTP API.
// It requests raw PCM via the Accept header and streams it downstream.
package rime

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"

	"github.com/gojargo/jargo/service/tts"
)

const (
	apiURL         = "https://users.rime.ai/v1/rime-tts"
	defaultModel   = "arcana"
	defaultSpeaker = "astra"
	defaultLang    = "en"
	// sampleRate is the PCM rate jargo requests from Rime.
	sampleRate = 24000
)

// Config configures the Rime TTS service.
type Config struct {
	// APIKey is the Rime API key; empty uses the RIME_API_KEY env var.
	APIKey string
	// Model is the Rime model id; empty uses a default.
	Model string
	// Speaker is the voice id; empty uses a default.
	Speaker string
	// Lang is the spoken language; empty uses "en".
	Lang string
}

// NewTTS builds a Rime TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("RIME_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Speaker == "" {
		cfg.Speaker = defaultSpeaker
	}
	if cfg.Lang == "" {
		cfg.Lang = defaultLang
	}
	return tts.New("RimeTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return sampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := json.Marshal(map[string]any{
		"text":         text,
		"speaker":      s.cfg.Speaker,
		"modelId":      s.cfg.Model,
		"lang":         s.cfg.Lang,
		"samplingRate": sampleRate,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/pcm")
	return tts.StreamResponse(s.http, req, emit)
}
