// Package rime is a streaming text-to-speech service backed by Rime's HTTP API.
// It requests raw PCM via the Accept header and streams it downstream.
package rime

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	apiURL         = "https://users.rime.ai/v1/rime-tts"
	defaultModel   = "arcana"
	defaultSpeaker = "astra"
	defaultLang    = "en"
	// defaultSampleRate is the PCM rate jargo requests from Rime.
	defaultSampleRate = 24000
)

// Config configures the Rime TTS service. The model-specific controls are
// pointers (or empty strings) and are omitted from the request when unset.
type Config struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// Model is the Rime model id; empty uses a default.
	Model string
	// Speaker is the voice id; empty uses a default.
	Speaker string
	// Lang is the spoken language; empty uses "en".
	Lang string
	// SampleRate is the PCM rate requested from Rime and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// RepetitionPenalty penalizes token repetition (arcana only, 1.0-2.0).
	RepetitionPenalty *float64
	// Temperature is the sampling temperature (arcana only, 0.0-1.0).
	Temperature *float64
	// TopP is the cumulative-probability threshold (arcana only, 0.0-1.0).
	TopP *float64
	// SpeedAlpha is the speech-speed multiplier (mistv2 only).
	SpeedAlpha *float64
	// ReduceLatency trades quality for latency (mistv2 only).
	ReduceLatency *bool
	// PauseBetweenBrackets adds pauses between bracketed content (mistv2 only).
	PauseBetweenBrackets *bool
	// PhonemizeBetweenBrackets phonemizes bracketed content (mistv2 only).
	PhonemizeBetweenBrackets *bool
	// InlineSpeedAlpha is inline speed-control markup; empty omits it.
	InlineSpeedAlpha string
	// TimeScaleFactor scales playback speed (arcana, mistv3, coda); >1 slows down.
	TimeScaleFactor *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Rime TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Speaker == "" {
		cfg.Speaker = defaultSpeaker
	}
	if cfg.Lang == "" {
		cfg.Lang = defaultLang
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("RimeTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// requestBody builds the Rime request body for text, including any set
// model-specific controls.
func (s *synthesizer) requestBody(text string) ([]byte, error) {
	m := map[string]any{
		"text":         text,
		"speaker":      s.cfg.Speaker,
		"modelId":      s.cfg.Model,
		"lang":         s.cfg.Lang,
		"samplingRate": s.cfg.SampleRate,
	}
	setFloat(m, "repetition_penalty", s.cfg.RepetitionPenalty)
	setFloat(m, "temperature", s.cfg.Temperature)
	setFloat(m, "top_p", s.cfg.TopP)
	setFloat(m, "speedAlpha", s.cfg.SpeedAlpha)
	setFloat(m, "timeScaleFactor", s.cfg.TimeScaleFactor)
	setBool(m, "reduceLatency", s.cfg.ReduceLatency)
	setBool(m, "pauseBetweenBrackets", s.cfg.PauseBetweenBrackets)
	setBool(m, "phonemizeBetweenBrackets", s.cfg.PhonemizeBetweenBrackets)
	if s.cfg.InlineSpeedAlpha != "" {
		m["inlineSpeedAlpha"] = s.cfg.InlineSpeedAlpha
	}
	return json.Marshal(m)
}

// setFloat sets key only when v is non-nil.
func setFloat(m map[string]any, key string, v *float64) {
	if v != nil {
		m[key] = *v
	}
}

// setBool sets key only when v is non-nil.
func setBool(m map[string]any, key string, v *bool) {
	if v != nil {
		m[key] = *v
	}
}

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := s.requestBody(text)
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
