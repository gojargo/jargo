package heygen

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ttsstream"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// defaultHTTPBaseURL is the HeyGen API base.
	defaultHTTPBaseURL = "https://api.heygen.com"
	// httpTTSPath is the streaming speech endpoint, which answers with raw PCM.
	httpTTSPath = "/v3/models/audio/tts/stream"
)

// HTTPTTSConfig configures the HeyGen Voice HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the HeyGen API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses "https://api.heygen.com".
	BaseURL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 24 kHz. A rate
	// HeyGen does not synthesize (it synthesizes 8000, 16000, 22050, 24000, 44100
	// and 48000) is synthesized at 24 kHz and resampled to this one.
	SampleRate int
	// Model is the HeyGen Voice model: "heygen-voice-1", "heygen-voice-1-turbo"
	// or "heygen-voice-1-turbo-free". Empty uses "heygen-voice-1-turbo".
	Model string
	// VoiceID is the HeyGen voice: a public voice, or an instant or professional
	// clone in your workspace. A professional clone needs the "heygen-voice-1"
	// model; the Turbo models refuse it. Empty uses the public voice Nadine.
	VoiceID string
	// Language is the language of the text; the zero value lets HeyGen detect
	// it.
	Language language.Language
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// HTTPTTSSettings is the HeyGen Voice HTTP configuration that can change while
// the pipeline runs: the model, the voice and the language. Each request is
// built from it, so a change applies from the next sentence on.
type HTTPTTSSettings struct {
	settings.TTS
}

// NewHTTPTTS builds a HeyGen Voice TTS service over the HTTP API.
//
// It sends each sentence to HeyGen's streaming speech endpoint and passes the
// raw PCM audio on as it arrives. It defaults to "heygen-voice-1-turbo", the
// lowest-latency model, and the public voice Nadine. Professional voice clones
// need "heygen-voice-1", and are served only by this service.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig) (*tts.Base, *httpSynthesizer) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultHTTPBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{
		cfg:           cfg,
		synthesisRate: synthesisRate(cfg.SampleRate, httpSampleRates),
		live:          newHTTPTTSSettings(cfg),
	}
	return tts.New("HeyGenHTTPTTS", s), s
}

// newHTTPTTSSettings is the starting state, taken from what the service was
// built with.
func newHTTPTTSSettings(cfg HTTPTTSConfig) *HTTPTTSSettings {
	s := &HTTPTTSSettings{}
	s.Model = settings.Set(cmp.Or(cfg.Model, defaultModel))
	s.Voice = settings.Set(cmp.Or(cfg.VoiceID, defaultVoiceID))
	s.Language = languageSetting(cfg.Language)
	return s
}

// httpSynthesizer speaks each sentence with one request to the streaming
// speech endpoint.
type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// synthesisRate is the rate HeyGen synthesizes at, resolved from the
	// service's.
	synthesisRate int
	// live is what a caller may change while the pipeline runs.
	live *HTTPTTSSettings
}

// SampleRate reports the PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the HeyGen model and voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(""), VoiceID: s.live.Voice.Or("")}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *httpSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way HeyGen Voice does, implementing
// tts.LanguageNamer.
func (s *httpSynthesizer) ServiceLanguage(l language.Language) string { return heygenLanguage(l) }

// payload is the streaming speech request for text.
func (s *httpSynthesizer) payload(text string) map[string]any {
	payload := map[string]any{
		"text":     text,
		"voice_id": s.live.Voice.Or(""),
		"model":    s.live.Model.Or(""),
		"output_format": map[string]any{
			"encoding":          "pcm",
			"sample_rate_hertz": s.synthesisRate,
		},
	}
	// HeyGen refuses a language it does not speak, and detects one left out.
	if lang := s.live.Language.Or(""); supportedLanguage(lang) {
		payload["language"] = lang
	}
	return payload
}

// RunTTS requests speech for text and streams the raw PCM as it arrives, at the
// rate HeyGen synthesized it; the base resamples it to the service's and keeps
// the frames sample-aligned. A failure is yielded as an error in the context's
// stream.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("HeyGen TTS error: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *httpSynthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+httpTTSPath, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set(headerAPIKey, s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return yield(frames.NewErrorFrame(fmt.Sprintf("HeyGen TTS error (status: %d): %s",
			resp.StatusCode, errorDetail(bytes.ToValidUTF8(raw, nil)))))
	}
	tts.StartUsageMetrics(ctx, text)

	emit := ttsstream.PCMYielder(yield, s.synthesisRate)
	buf := make([]byte, tts.ChunkSize(s.cfg.SampleRate))
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if err := emit(bytes.Clone(buf[:n])); err != nil {
				return err
			}
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return readErr
		}
	}
}

// languageSetting is the language a service starts with: the one the caller
// gave, which the base names the way HeyGen does, or none, so HeyGen detects
// it.
func languageSetting(l language.Language) settings.Opt[string] {
	if l == "" {
		return settings.Cleared[string]()
	}
	return settings.Set(string(l))
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.LanguageNamer  = (*httpSynthesizer)(nil)
	_ tts.Describer      = (*httpSynthesizer)(nil)
)
