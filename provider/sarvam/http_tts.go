// Sarvam text-to-speech over the HTTP API, which answers each synthesis with
// the whole of its audio. It needs no connection held open for the call, and
// suits an application that prefers plain HTTP requests over a WebSocket.

package sarvam

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// defaultHTTPTTSURL is the Sarvam API base.
	defaultHTTPTTSURL = "https://api.sarvam.ai"
	// wavHeaderSize is the length of the WAV header Sarvam's audio may open
	// with.
	wavHeaderSize = 44
)

// HTTPTTSConfig configures the Sarvam HTTP TTS service. Optional fields modeled
// as pointers fall back to Sarvam's neutral values when unset, and
// model-specific fields are dropped for models that do not support them.
type HTTPTTSConfig struct {
	// APIKey is the Sarvam API subscription key. Required.
	APIKey string `validate:"required"`
	// URL overrides the API base; empty uses "https://api.sarvam.ai".
	URL string
	// Model is the TTS model; empty uses "bulbul:v3". "bulbul:v4-flash" is also
	// available. "bulbul:v2" is deprecated: Sarvam's API rejects it.
	Model string `validate:"omitempty,oneof=bulbul:v2 bulbul:v3-beta bulbul:v3 bulbul:v4-flash"`
	// Voice is the speaker id; empty uses the model's default speaker.
	Voice string
	// Language for synthesis; the zero value uses English (India). Mapped to
	// Sarvam's regional code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream (8000, 16000,
	// 22050, 24000, and, for bulbul:v3, 32000, 44100, 48000); 0 uses the model
	// default (24000 Hz, 22050 Hz for bulbul:v2).
	SampleRate int
	// EnablePreprocessing toggles text preprocessing; nil defaults to off. Always
	// forced on for every model but bulbul:v2.
	EnablePreprocessing *bool
	// Pace multiplies the speaking rate; nil uses 1.0. Clamped to the model's
	// supported range (0.5 to 2.0, 0.3 to 3.0 for bulbul:v2).
	Pace *float64
	// Pitch adjusts voice pitch; nil uses 0.0. Only bulbul:v4-flash and
	// bulbul:v2.
	Pitch *float64
	// Loudness multiplies volume; nil uses 1.0. Only bulbul:v4-flash and
	// bulbul:v2.
	Loudness *float64
	// Temperature controls output randomness (0.01 to 1.0); nil uses 0.6. Only
	// the v3 variants.
	Temperature *float64
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds a Sarvam TTS service over the HTTP API. Each sentence is
// one request, answered with the whole of its audio. Prefer NewTTS on a call,
// where the audio streams back over a connection held open for it.
//
// bulbul:v3, the default, takes a pace from 0.5 to 2.0 and a temperature, but
// no pitch or loudness. bulbul:v4-flash takes pitch, loudness and pace, but no
// temperature, and its own catalog of speakers. Both always preprocess the
// text.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig) (*tts.Base, *httpSynthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultHTTPTTSURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.Model == deprecatedTTSModel {
		slog.Warn("sarvam tts: bulbul:v2 is deprecated, use bulbul:v3 instead")
	}
	mc, ok := ttsModelConfigs[cfg.Model]
	if !ok {
		mc = ttsModelConfigs[defaultTTSModel]
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{
		apiKey:     cfg.APIKey,
		url:        strings.TrimRight(cfg.URL, "/"),
		sampleRate: firstNonZero(cfg.SampleRate, mc.defaultSampleRate),
		client:     cfg.Client,
		mc:         mc,
		live:       newHTTPTTSSettings(cfg, mc),
	}
	return tts.New("SarvamHTTPTTS", s), s
}

// httpSynthesizer speaks each sentence with one request to the HTTP API.
type httpSynthesizer struct {
	apiKey     string
	url        string
	sampleRate int
	client     *http.Client

	// live is the settings store and mc the capabilities of its model. Each
	// request is built from both, so a change applies from the next sentence
	// on.
	live *HTTPTTSSettings
	mc   ttsModelConfig
}

// SampleRate reports the PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.sampleRate }

// Metadata reports the Sarvam model and voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	model, _ := s.live.Model.Value()
	voice, _ := s.live.Voice.Value()
	return tts.Metadata{Model: model, VoiceID: voice}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *httpSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Sarvam does, implementing
// tts.LanguageNamer.
func (s *httpSynthesizer) ServiceLanguage(l language.Language) string { return resolveTTSLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Each
// request is built from the model's capabilities, so a new model swaps them in,
// and, unless the update named a voice too, moves to its default speaker.
func (s *httpSynthesizer) UpdateSettings(_ context.Context, changed settings.Changed) error {
	if !changed.Has("model") {
		return nil
	}
	model, _ := s.live.Model.Value()
	s.mc = ttsModelConfigs[model]
	if !changed.Has("voice") {
		s.live.Voice = settings.Set(s.mc.defaultSpeaker)
	}
	return nil
}

// payload is the synthesis request. A control the model does not support is
// left off; one it supports is sent, at its neutral value when unset.
func (s *httpSynthesizer) payload(text string) map[string]any {
	msg := map[string]any{
		keyText:                text,
		keyTargetLanguageCode:  s.live.Language.AsAny(),
		keySpeaker:             s.live.Voice.AsAny(),
		"speech_sample_rate":   s.sampleRate,
		"enable_preprocessing": s.live.EnablePreprocessing.AsAny(),
		"model":                s.live.Model.AsAny(),
		"pace":                 s.live.Pace.Or(defaultPace),
	}
	if s.mc.supportsPitch {
		msg["pitch"] = s.live.Pitch.Or(neutralPitch)
	}
	if s.mc.supportsLoudness {
		msg["loudness"] = s.live.Loudness.Or(neutralLoudness)
	}
	if s.mc.supportsTemperature {
		msg["temperature"] = s.live.Temperature.Or(neutralTemperature)
	}
	return msg
}

// httpTTSResponse is the subset of Sarvam's answer read.
type httpTTSResponse struct {
	Audios []string `json:"audios"`
}

// RunTTS requests speech for text and yields its audio as one frame. A failure
// is yielded as an error in the context's stream.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Error generating TTS: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *httpSynthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url+"/text-to-speech", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("api-subscription-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		errorText, _ := io.ReadAll(resp.Body)
		return yield(frames.NewErrorFrame("Sarvam API error: " + string(errorText)))
	}
	var answer httpTTSResponse
	if err = json.NewDecoder(resp.Body).Decode(&answer); err != nil {
		return err
	}
	tts.StartUsageMetrics(ctx, text)

	if len(answer.Audios) == 0 {
		return yield(frames.NewErrorFrame("No audio data received"))
	}
	// One text was sent, so there is one audio.
	audio, err := base64.StdEncoding.DecodeString(answer.Audios[0])
	if err != nil {
		return err
	}
	if len(audio) > wavHeaderSize && bytes.HasPrefix(audio, []byte("RIFF")) {
		slog.Debug("sarvam tts: stripping the WAV header from the audio")
		audio = audio[wavHeaderSize:]
	}
	return yield(frames.NewTTSAudioRawFrame(audio, s.sampleRate, 1))
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*httpSynthesizer)(nil)
	_ tts.SettingsUpdater = (*httpSynthesizer)(nil)
	_ tts.LanguageNamer   = (*httpSynthesizer)(nil)
	_ tts.Describer       = (*httpSynthesizer)(nil)
)
