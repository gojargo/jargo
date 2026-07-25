package hume

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/service/tts"
)

// NewTTS builds a Hume TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.VoiceProvider == "" {
		cfg.VoiceProvider = defaultVoiceProvider
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
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	utterance := map[string]any{"text": text}
	if voice := s.voice(); voice != nil {
		utterance["voice"] = voice
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
	}
	if _, ok := utterance["voice"]; ok {
		payload["instant_mode"] = true // requires a specified voice
	}
	if s.cfg.Version != "" {
		payload["version"] = s.cfg.Version
	}

	body, err := json.Marshal(payload)
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

// voice builds the voice selector, preferring an id over a name; nil when no
// voice is configured.
func (s *synthesizer) voice() map[string]any {
	switch {
	case s.cfg.VoiceID != "":
		return map[string]any{"id": s.cfg.VoiceID, "provider": s.cfg.VoiceProvider}
	case s.cfg.VoiceName != "":
		return map[string]any{"name": s.cfg.VoiceName, "provider": s.cfg.VoiceProvider}
	default:
		return nil
	}
}
