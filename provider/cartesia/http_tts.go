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
	return tts.New("CartesiaHTTPTTS", &httpSynthesizer{cfg: cfg})
}

type httpSynthesizer struct {
	cfg HTTPTTSConfig
}

// SampleRate reports the requested PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Cartesia model and voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
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
		"model_id":      s.cfg.Model,
		fieldTranscript: text,
		"voice":         map[string]any{"mode": "id", "id": s.cfg.VoiceID},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if lang := cartesiaLanguage(s.cfg.Language); lang != "" {
		msg["language"] = lang
	}
	if s.cfg.GenerationConfig != nil {
		msg["generation_config"] = s.cfg.GenerationConfig
	}
	if s.cfg.PronunciationDictID != "" {
		msg["pronunciation_dict_id"] = s.cfg.PronunciationDictID
	}
	return msg
}
