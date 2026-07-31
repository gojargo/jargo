package lmnt

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

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
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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
