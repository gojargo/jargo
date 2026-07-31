package rime

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

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
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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
