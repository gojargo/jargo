// Neuphonic text-to-speech over the HTTP API, which streams each synthesis back
// as server-sent events. It needs no connection held open for the call, and
// suits an application that prefers plain HTTP requests over a WebSocket.

package neuphonic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

// defaultHTTPURL is the Neuphonic HTTP API base.
const defaultHTTPURL = "https://api.neuphonic.com"

// errAPI is returned for an event in which Neuphonic reports errors.
//
//nolint:gochecknoglobals // sentinel error
var errAPI = errors.New("Neuphonic API error") //nolint:staticcheck // the provider's name leads the message

// HTTPTTSConfig configures the Neuphonic HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Neuphonic API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the HTTP API base; empty uses the hosted API.
	URL string
	// VoiceID is the voice id; empty uses Neuphonic's default voice.
	VoiceID string
	// Language for synthesis; the zero value uses English. Mapped to Neuphonic's
	// language code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 22.05 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_linear".
	Encoding string
	// Speed is the speech-speed multiplier; 0 uses 1.0.
	Speed float64
	// Temperature is the randomness introduced into the synthesis, from 0 to 1.
	// Nil leaves it to Neuphonic.
	Temperature *float64
	// IPAKey is the regional variant the sounds of <phoneme:...> tags in the
	// text are read with, either "es-castilian" or "pt-br". Empty leaves it to
	// Neuphonic.
	IPAKey string
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds a Neuphonic TTS service over the HTTP API. Each sentence is
// one request, whose audio streams back as server-sent events. Prefer NewTTS on
// a call, where a connection held open spares each sentence the request.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig) (*tts.Base, *httpSynthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultHTTPURL
	}
	cfg.URL = strings.TrimRight(cfg.URL, "/")
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Speed == 0 {
		cfg.Speed = defaultSpeed
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{cfg: cfg, live: newTTSSettings(Config{
		VoiceID:     cfg.VoiceID,
		Language:    cfg.Language,
		Speed:       cfg.Speed,
		Temperature: cfg.Temperature,
		IPAKey:      cfg.IPAKey,
	})}
	return tts.New("NeuphonicHTTPTTS", s), s
}

// httpSynthesizer speaks each sentence with one request to the HTTP API.
type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings
}

// SampleRate reports the requested PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Each request is built from it, so a change
// applies from the next sentence on.
func (s *httpSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Neuphonic does, implementing
// tts.LanguageNamer.
func (s *httpSynthesizer) ServiceLanguage(l language.Language) string { return neuphonicLanguage(l) }

// payload is the synthesis request Neuphonic takes. The speed and the language
// are always sent; the voice, the temperature and the IPA key only when set.
func (s *httpSynthesizer) payload(text string) map[string]any {
	msg := map[string]any{
		"text":          text,
		"lang_code":     s.live.Language.Ptr(),
		"encoding":      s.cfg.Encoding,
		"sampling_rate": s.cfg.SampleRate,
		"speed":         s.live.Speed.Ptr(),
	}
	if voice, ok := s.live.Voice.Value(); ok && voice != "" {
		msg["voice_id"] = voice
	}
	if temperature, ok := s.live.Temperature.Value(); ok {
		msg["temperature"] = temperature
	}
	if ipaKey, ok := s.live.IPAKey.Value(); ok {
		msg["ipa_key"] = ipaKey
	}
	return msg
}

// RunTTS requests speech for text and streams the audio of each event as it
// arrives. A failure is yielded as an error in the context's stream; one event
// that fails is reported and the rest of the stream is still read.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails before or between the events.
func (s *httpSynthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	endpoint := s.cfg.URL + "/sse/speak/" + s.live.Language.Or("")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("X-API-KEY", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		errorText, _ := io.ReadAll(resp.Body)
		return yield(frames.NewErrorFrame(
			fmt.Sprintf("Neuphonic API error: HTTP %d - %s", resp.StatusCode, errorText)))
	}
	tts.StartUsageMetrics(ctx, text)

	// The stream is read line by line. A line is as long as the audio it
	// carries, so it is read whole rather than through a scanner's bounded
	// buffer.
	r := bufio.NewReader(resp.Body)
	for {
		line, readErr := r.ReadBytes('\n')
		if err := s.handleLine(line, yield); err != nil {
			return err
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return readErr
		}
	}
}

// handleLine yields the audio one line of the stream carries. A line that
// fails is reported in the stream and the rest is still read; only a yield
// that fails ends it.
func (s *httpSynthesizer) handleLine(line []byte, yield func(f frames.Frame) error) error {
	message := strings.ToValidUTF8(string(line), "")
	if strings.TrimSpace(message) == "" {
		return nil
	}
	audio, err := audioOf(parseSSEMessage(message))
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		return yield(ef)
	}
	if audio == nil {
		return nil
	}
	return yield(frames.NewTTSAudioRawFrame(audio, s.cfg.SampleRate, 1))
}

// sseMessage is the subset of a Neuphonic event read.
type sseMessage struct {
	Data *struct {
		Audio *string `json:"audio"`
	} `json:"data"`
	Errors     any `json:"errors"`
	StatusCode any `json:"status_code"`
}

// parseSSEMessage parses one server-sent event. It reports false for anything
// but a data event carrying JSON, and fails for an event that reports errors.
func parseSSEMessage(message string) (sseMessage, bool, error) {
	message = strings.TrimSpace(message)
	if message == "" || !strings.Contains(message, "data") {
		return sseMessage{}, false, nil
	}
	_, data, found := strings.Cut(message, ": ")
	if !found {
		slog.Warn("neuphonic http tts: failed to parse SSE message", "message", message)
		return sseMessage{}, false, nil
	}
	if data == "" || data == "[DONE]" {
		return sseMessage{}, false, nil
	}
	var m sseMessage
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		slog.Warn("neuphonic http tts: failed to parse SSE message", "err", err)
		return sseMessage{}, false, nil
	}
	if m.Errors != nil {
		status := m.StatusCode
		if status == nil {
			status = "unknown"
		}
		return sseMessage{}, false, fmt.Errorf("%w %v: %v", errAPI, status, m.Errors)
	}
	return m, true, nil
}

// audioOf decodes the audio a parsed event carries, or nil when it carries
// none.
func audioOf(m sseMessage, ok bool, err error) ([]byte, error) {
	if err != nil || !ok || m.Data == nil || m.Data.Audio == nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(*m.Data.Audio)
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.LanguageNamer  = (*httpSynthesizer)(nil)
)
