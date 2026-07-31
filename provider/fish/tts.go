package fish

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

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
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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
