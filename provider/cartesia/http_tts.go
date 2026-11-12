// Cartesia text-to-speech over the HTTP API, which answers a whole synthesis in
// one response rather than streaming it. It is the simpler of the two services:
// no session to hold open, no word timings, and the first audio arrives only
// once the whole thing has been generated. Prefer NewTTS on a call, and this
// where the latency does not matter and a plain request is easier to operate.

package cartesia

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttscartesia "github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	// defaultHTTPTTSBaseURL is the host the synthesis request goes to.
	defaultHTTPTTSBaseURL = "https://api.cartesia.ai"
	// httpTTSPath is the endpoint that answers with the audio itself.
	httpTTSPath = "/tts/bytes"
	// httpTTSErrorBodyLimit bounds how much of a failed response is reported.
	httpTTSErrorBodyLimit = 4096
)

// errHTTPTTSStatus is returned when the synthesis API answers with a status
// other than 200.
//
//nolint:gochecknoglobals // sentinel error
var errHTTPTTSStatus = errors.New("cartesia: tts unexpected status")

// HTTPTTSConfig configures the Cartesia HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Cartesia API key. Required.
	APIKey string `validate:"required"`
	// VoiceID is the voice to speak in. Required: Cartesia has no default.
	VoiceID string `validate:"required"`
	// BaseURL overrides the API host; empty uses the hosted one.
	BaseURL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Model is the Cartesia model id; empty uses a default.
	Model string
	// Language for synthesis; the zero value leaves it unset, which leaves
	// Cartesia on its own default.
	Language language.Language
	// SampleRate is the PCM rate requested from Cartesia and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Container is the audio container; empty uses "raw".
	Container string
	// GenerationConfig guides generation (volume, speed, emotion) on supported
	// models; nil omits it.
	GenerationConfig *GenerationConfig
	// PronunciationDictID applies a custom pronunciation dictionary; empty omits it.
	PronunciationDictID string
	// Accent is the accent a multilingual voice speaks the language with (for
	// example "british"). It must be one the voice supports, as listed in the
	// accents field of Cartesia's Get Voice response. Empty lets Cartesia pick
	// the closest accent the voice supports for the requested language.
	Accent string
	// Headers are sent on every request, on top of the ones the service sets.
	Headers map[string]string
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds a Cartesia TTS service over the HTTP API.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultHTTPTTSBaseURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Container == "" {
		cfg.Container = defaultContainer
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := newHTTPSynthesizer(cfg)
	b := tts.New("CartesiaHTTPTTS", s)
	s.expressionSettings = b.ExpressionSettings
	return b
}

type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings
	// expressionSettings is what the service's expression adapter asks to send
	// with the request being synthesized, in an expressive conversation.
	expressionSettings func() map[string]any
}

// newHTTPSynthesizer builds the synthesizer with its settings seeded from cfg.
func newHTTPSynthesizer(cfg HTTPTTSConfig) *httpSynthesizer {
	return &httpSynthesizer{cfg: cfg, live: newTTSSettings(Config{
		Model:               cfg.Model,
		VoiceID:             cfg.VoiceID,
		Language:            cfg.Language,
		GenerationConfig:    cfg.GenerationConfig,
		PronunciationDictID: cfg.PronunciationDictID,
		Accent:              cfg.Accent,
	})}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Each request is built from it, so a change
// applies from the next sentence on.
func (s *httpSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Cartesia does, implementing
// tts.LanguageNamer.
func (s *httpSynthesizer) ServiceLanguage(l language.Language) string {
	return cartesiaLanguage(l)
}

// ExpressionAdapter implements tts.ExpressionAdapterProvider: Cartesia's
// emotions, laughter and pauses.
func (s *httpSynthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	return ttscartesia.CartesiaExpressionAdapter{}
}

// SampleRate reports the requested PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Cartesia model and voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Model), VoiceID: s.live.Voice.Or(s.cfg.VoiceID)}
}

// RunTTS asks for the whole synthesis and streams the audio it answers with. The
// response is read as it arrives, so playback starts before the download has
// finished even though generation already has.
func (s *httpSynthesizer) RunTTS(
	ctx context.Context, text, _ string, yield func(f frames.Frame) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())

	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, s.cfg.BaseURL+httpTTSPath, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("X-API-Key", s.cfg.APIKey)
	req.Header.Set("Cartesia-Version", s.cfg.Version)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range s.cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, httpTTSErrorBodyLimit))
		return errs.NewHTTPStatusError(resp.StatusCode,
			fmt.Errorf("%w %d: %s", errHTTPTTSStatus, resp.StatusCode, msg))
	}

	// Read in chunks rather than all at once: the audio is raw PCM, so a partial
	// read is playable as it stands.
	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if emitErr := emit(buf[:n]); emitErr != nil {
				return emitErr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// payload is the synthesis request Cartesia takes.
func (s *httpSynthesizer) payload(text string) map[string]any {
	msg := map[string]any{
		"model_id":      s.live.Model.Or(s.cfg.Model),
		fieldTranscript: text,
		"voice":         map[string]any{"mode": "id", "id": s.live.Voice.Or(s.cfg.VoiceID)},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if lang := s.live.Language.Or(cartesiaLanguage(s.cfg.Language)); lang != "" {
		msg["language"] = lang
	}
	if gen := generationConfig(
		s.live.GenerationConfig.Or(s.cfg.GenerationConfig), expression(s.expressionSettings)); gen != nil {
		msg["generation_config"] = gen
	}
	if dict := s.live.PronunciationDictID.Or(s.cfg.PronunciationDictID); dict != "" {
		msg["pronunciation_dict_id"] = dict
	}
	if accent := s.live.Accent.Or(s.cfg.Accent); accent != "" {
		msg["accent"] = accent
	}
	return msg
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.LanguageNamer  = (*httpSynthesizer)(nil)
)
