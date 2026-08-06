package hume

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

// NewTTS builds a Hume TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return tts.New("HumeTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports Hume's fixed PCM output rate.
func (s *synthesizer) SampleRate() int { return humeSampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	body, err := json.Marshal(s.request(text))
	if err != nil {
		return err
	}
	endpoint := s.cfg.BaseURL + "/v0/tts/stream/file"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("X-Hume-Api-Key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}

// request builds the synthesis body for one utterance. The voice is named by id
// and by nothing else, and instant mode is always on, which is what a voice
// being required buys: it cannot be used without one.
func (s *synthesizer) request(text string) map[string]any {
	utterance := map[string]any{
		"text":  text,
		"voice": map[string]any{"id": s.cfg.VoiceID},
	}
	if s.cfg.Description != "" {
		utterance["description"] = s.cfg.Description
	}
	if s.cfg.Speed != nil {
		utterance["speed"] = *s.cfg.Speed
	}

	payload := map[string]any{
		"utterances":    []any{utterance},
		"format":        map[string]any{"type": "pcm"},
		"strip_headers": true,
		"instant_mode":  true,
	}
	if s.cfg.Version != "" {
		payload["version"] = s.cfg.Version
	}
	return payload
}
