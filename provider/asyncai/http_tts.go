// Async text-to-speech over the HTTP streaming API. It needs no connection held
// open for the call, and suits an application that prefers plain HTTP requests
// over a WebSocket.

package asyncai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	// defaultHTTPURL is the Async HTTP API base.
	defaultHTTPURL = "https://api.async.com"
	// httpReadChunk is how much of the response is read at a time.
	httpReadChunk = 64 * 1024
)

// errHTTPStatus is returned when the API answers with a status other than 200.
//
//nolint:gochecknoglobals // sentinel error
var errHTTPStatus = errors.New("Async API returned status") //nolint:staticcheck // the provider's name leads

// HTTPTTSConfig configures the Async HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Async API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice id (UUID) to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the HTTP API base; empty uses the hosted API.
	URL string
	// Version sets the API version; empty uses a default.
	Version string
	// Model is the Async model id; empty uses a default.
	Model string
	// Language for synthesis; the zero value leaves it unset. Mapped to Async's
	// language code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Container is the audio container; empty uses "raw".
	Container string
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds an Async TTS service over the HTTP streaming API. Each
// sentence is one request, whose audio is read whole and yielded as one frame.
// Prefer NewTTS on a call, where a connection held open spares each sentence
// the request.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig) (*tts.Base, *httpSynthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultHTTPURL
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
	s := &httpSynthesizer{cfg: cfg, live: &settings.TTS{}}
	s.live.Model = settings.Set(cfg.Model)
	s.live.Voice = settings.Set(cfg.Voice)
	s.live.Language = settings.Cleared[string]()
	if cfg.Language != "" {
		s.live.Language = settings.Set(string(cfg.Language))
	}
	return tts.New("AsyncAIHTTPTTS", s), s
}

// httpSynthesizer speaks each sentence with one request to the HTTP API.
type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *settings.TTS
}

// SampleRate reports the requested PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Async model and voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Each request is built from it, so a change
// applies from the next sentence on.
func (s *httpSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Async does, implementing
// tts.LanguageNamer.
func (s *httpSynthesizer) ServiceLanguage(l language.Language) string { return asyncLanguage(l) }

// payload is the synthesis request Async takes.
func (s *httpSynthesizer) payload(text string) map[string]any {
	return map[string]any{
		"model_id":    s.live.Model.AsAny(),
		keyTranscript: text,
		keyVoice:      map[string]any{"mode": "id", "id": s.live.Voice.AsAny()},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
		keyLanguage: s.live.Language.AsAny(),
	}
}

// RunTTS requests speech for text, reads the whole of the streamed audio, and
// yields it as one frame. A failure is yielded as an error in the context's
// stream.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *httpSynthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	endpoint := strings.TrimRight(s.cfg.URL, "/") + "/text_to_speech/streaming"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("version", s.cfg.Version)
	req.Header.Set("x-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		errorText, _ := io.ReadAll(resp.Body)
		if err := yield(frames.NewErrorFrame("Async API error: " + string(errorText))); err != nil {
			return err
		}
		return errs.NewHTTPStatusError(resp.StatusCode,
			fmt.Errorf("%w %d: %s", errHTTPStatus, resp.StatusCode, errorText))
	}

	// The audio streams in, and is gathered whole before it is yielded.
	var audio bytes.Buffer
	buf := make([]byte, httpReadChunk)
	for {
		n, readErr := resp.Body.Read(buf)
		audio.Write(buf[:n])
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	tts.StartUsageMetrics(ctx, text)
	return yield(frames.NewTTSAudioRawFrame(audio.Bytes(), s.cfg.SampleRate, 1))
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.LanguageNamer  = (*httpSynthesizer)(nil)
	_ tts.Describer      = (*httpSynthesizer)(nil)
)
