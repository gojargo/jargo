// Package camb is a streaming text-to-speech service backed by Camb.ai's MARS
// models. For each sentence it posts the text to Camb.ai's HTTP streaming
// endpoint and streams the raw 16-bit PCM response downstream. The model sets
// the native output rate: mars-flash and mars-instruct produce 22.05 kHz,
// mars-pro produces 48 kHz.
package camb

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

const (
	defaultURL     = "https://client.camb.ai/apis/tts-stream"
	defaultModel   = "mars-flash"
	defaultVoiceID = 147320
	// maxTextLen bounds a single request; Camb.ai rejects longer input.
	maxTextLen = 3000
	// instructModel is the only model that honors user instructions.
	instructModel = "mars-instruct"
	// proSampleRate and flashSampleRate are the MARS models' native PCM rates.
	proSampleRate   = 48000
	flashSampleRate = 22050
)

// Config configures the Camb.ai TTS service.
type Config struct {
	// APIKey is the Camb.ai API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the streaming endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the MARS model id; empty uses "mars-flash".
	Model string
	// VoiceID is the numeric voice id; 0 uses a default public voice.
	VoiceID int
	// Language for synthesis; the zero value uses "en-us". Mapped to Camb.ai's
	// lower-case, region-qualified BCP-47 code.
	Language language.Language
	// SampleRate is the PCM rate requested from Camb.ai and emitted downstream;
	// 0 uses the model's native rate (22.05 kHz for mars-flash/mars-instruct,
	// 48 kHz for mars-pro).
	SampleRate int
	// UserInstructions guide delivery for the mars-instruct model only; empty
	// omits them and other models ignore them.
	UserInstructions string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Camb.ai TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.VoiceID == 0 {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = modelSampleRate(cfg.Model)
	}
	return tts.New("CambTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// modelSampleRate returns a MARS model's native PCM rate.
func modelSampleRate(model string) int {
	switch model {
	case "mars-pro", "mars-8.1-pro-beta":
		return proSampleRate
	default:
		return flashSampleRate
	}
}

// requestBody builds the Camb.ai request body for text.
func (s *synthesizer) requestBody(text string) ([]byte, error) {
	m := map[string]any{
		"text":         text,
		"voice_id":     s.cfg.VoiceID,
		"language":     cambLanguage(s.cfg.Language),
		"speech_model": s.cfg.Model,
		"output_configuration": map[string]any{
			"format":      "pcm_s16le",
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if s.cfg.Model == instructModel && s.cfg.UserInstructions != "" {
		m["user_instructions"] = s.cfg.UserInstructions
	}
	return json.Marshal(m)
}

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	if len(text) > maxTextLen {
		text = text[:maxTextLen]
	}
	body, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("x-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}

// cambRegional maps region-specific Languages to Camb.ai codes that differ from
// the base language's default.
//
//nolint:gochecknoglobals // lookup table
var cambRegional = map[language.Language]string{
	language.EnglishGB:    "en-gb",
	language.EnglishAU:    "en-au",
	language.SpanishMX:    "es-mx",
	language.FrenchCA:     "fr-ca",
	language.PortugueseBR: "pt-br",
	language.ChineseTW:    "zh-tw",
}

// cambBase maps a base language code to Camb.ai's region-qualified code.
//
//nolint:gochecknoglobals // lookup table
var cambBase = map[string]string{
	"es": "es-es", "fr": "fr-fr", "de": "de-de", "it": "it-it", "pt": "pt-pt",
	"nl": "nl-nl", "pl": "pl-pl", "ru": "ru-ru", "ja": "ja-jp", "ko": "ko-kr",
	"zh": "zh-cn", "ar": "ar-sa", "hi": "hi-in", "tr": "tr-tr", "vi": "vi-vn",
	"th": "th-th", "id": "id-id", "ms": "ms-my", "sv": "sv-se", "da": "da-dk",
	"no": "no-no", "fi": "fi-fi", "cs": "cs-cz", "el": "el-gr", "he": "he-il",
	"hu": "hu-hu", "ro": "ro-ro", "sk": "sk-sk", "uk": "uk-ua", "bg": "bg-bg",
	"hr": "hr-hr", "ta": "ta-in",
}

// cambLanguage maps a Language to Camb.ai's language code. Camb.ai wants a
// lower-case, region-qualified BCP-47 code; the zero value and unmapped
// languages fall back to "en-us".
func cambLanguage(l language.Language) string {
	if c, ok := cambRegional[l]; ok {
		return c
	}
	if c, ok := cambBase[l.BaseCode()]; ok {
		return c
	}
	return "en-us"
}
