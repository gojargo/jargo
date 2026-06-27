package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	defaultTTSModel = "gpt-4o-mini-tts"
	defaultTTSVoice = "alloy"
	// ttsSampleRate is the rate of OpenAI's "pcm" response format.
	ttsSampleRate = 24000
)

// TTSConfig configures the OpenAI TTS service.
type TTSConfig struct {
	// APIKey is the API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base.
	BaseURL string
	// Model is the TTS model; empty uses the default.
	Model string
	// Voice is the voice name; empty uses a default voice.
	Voice string
	// Instructions guide the voice synthesis behavior; empty omits it.
	Instructions string
	// Speed controls the speaking rate (0.25 to 4.0); nil omits it.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds an OpenAI TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	return NewCompatTTS("OpenAITTS", defaultLLMBaseURL, defaultTTSModel, defaultTTSVoice, cfg)
}

// NewCompatTTS builds a TTS service for any endpoint that implements OpenAI's
// /audio/speech API with a "pcm" response format (e.g. a local Kokoro-FastAPI
// server). name labels the service; baseURL, defaultModel and defaultVoice fill
// in the matching Config fields when they are empty.
func NewCompatTTS(name, baseURL, defaultModel, defaultVoice string, cfg TTSConfig) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	return tts.New(name, &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports OpenAI's fixed PCM output rate.
func (s *synthesizer) SampleRate() int { return ttsSampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	payload := map[string]any{
		"model":           s.cfg.Model,
		"voice":           s.cfg.Voice,
		"input":           text,
		"response_format": "pcm",
	}
	if s.cfg.Instructions != "" {
		payload["instructions"] = s.cfg.Instructions
	}
	if s.cfg.Speed != nil {
		payload["speed"] = *s.cfg.Speed
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}
