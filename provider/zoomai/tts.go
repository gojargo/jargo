package zoomai

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ttsstream"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/internal/zoomaiutil"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// defaultTTSURL is the Text-to-Speech synthesize endpoint.
	defaultTTSURL = "https://api.zoom.us/v2/aiservices/tts/synthesize"
	// defaultVoice is the voice used when the caller names none.
	defaultVoice = "jenna"
	// ttsSampleRate is the rate Zoom TTS synthesizes at: it returns signed
	// 16-bit little-endian mono PCM at 44.1 kHz only.
	ttsSampleRate = 44100
	// ttsResponseFormat asks for that PCM.
	ttsResponseFormat = "pcm_44100"
	// defaultSampleRate is the PCM rate emitted downstream when the caller
	// names none: the pipeline's default output rate.
	defaultSampleRate = 24000
	// maxInputChars is the longest input one request accepts, in Unicode code
	// points.
	maxInputChars = 4000
)

// HTTPTTSConfig configures the Zoom AI Services HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Zoom AI Services API key or JWT, from https://www.zoom.ai/.
	// Required.
	APIKey string `validate:"required"`
	// URL overrides the Text-to-Speech synthesize endpoint; empty uses
	// "https://api.zoom.us/v2/aiservices/tts/synthesize". A plaintext http://
	// URL to a host other than localhost logs a warning.
	URL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 24 kHz. Zoom TTS
	// returns 44.1 kHz audio, which is resampled to this rate.
	SampleRate int
	// Voice is a Zoom TTS voice ID, such as "jenna" (the default) or "ava". See
	// https://developers.zoom.us/docs/ai-services/tts/#language-and-voices for
	// the list.
	Voice string
	// Speed is the speaking rate multiplier, from 0.5 to 2.0. Nil uses the
	// voice's normal rate.
	Speed *float64
	// Volume is the volume multiplier, greater than 0 and up to 10. Nil uses
	// the voice's normal volume.
	Volume *float64
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// HTTPTTSSettings is the Zoom TTS configuration that can change while the
// pipeline runs. Each request is built from it, so a change applies from the
// next sentence on.
//
// Voice is a Zoom TTS voice ID. The API has no language or model field, so
// Language and Model have no value.
type HTTPTTSSettings struct {
	settings.TTS

	// Speed is the speaking rate multiplier, from 0.5 to 2.0. Without a value
	// the voice's normal rate is used.
	Speed settings.Opt[float64] `settings:"speed"`
	// Volume is the volume multiplier, greater than 0 and up to 10. Without a
	// value the voice's normal volume is used.
	Volume settings.Opt[float64] `settings:"volume"`
}

// HTTPTTSService is the Zoom text-to-speech service NewHTTPTTS builds.
type HTTPTTSService struct {
	*tts.Base
}

// NewHTTPTTS builds a Zoom AI Services TTS service over the HTTP API, in Fast
// mode.
//
// It sends each sentence to the Zoom TTS API in its own request and streams the
// audio into the pipeline as it arrives, so playback starts before synthesis
// finishes. Zoom TTS returns 44.1 kHz audio, which is resampled to the
// service's output rate. An interruption closes the request, which stops
// synthesis on the server.
func NewHTTPTTS(cfg HTTPTTSConfig) *HTTPTTSService {
	s := &HTTPTTSService{}
	s.Base = buildHTTPTTS(cfg, tts.WithSelf(s))
	return s
}

// buildHTTPTTS builds the service NewHTTPTTS returns, registering the
// service opts names as the processor the pipeline runs.
func buildHTTPTTS(cfg HTTPTTSConfig, opts ...tts.Option) *tts.Base {
	b, _ := newHTTPTTS(cfg, opts...)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig, opts ...tts.Option) (*tts.Base, *httpSynthesizer) {
	cfg.URL = zoomaiutil.CheckURL(cmp.Or(cfg.URL, defaultTTSURL), "Zoom TTS", "text")
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{cfg: cfg, live: newHTTPTTSSettings(cfg)}
	return tts.New("ZoomAIHTTPTTS", s, opts...), s
}

// newHTTPTTSSettings is the starting state, taken from what the service was
// built with.
func newHTTPTTSSettings(cfg HTTPTTSConfig) *HTTPTTSSettings {
	s := &HTTPTTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Set(cmp.Or(cfg.Voice, defaultVoice))
	s.Language = settings.Cleared[string]()
	s.Speed = settings.Cleared[float64]()
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	s.Volume = settings.Cleared[float64]()
	if cfg.Volume != nil {
		s.Volume = settings.Set(*cfg.Volume)
	}
	return s
}

// httpSynthesizer speaks each sentence with one request to the synthesize
// endpoint, or several when the sentence is longer than one request accepts.
type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// live is what a caller may change while the pipeline runs.
	live *HTTPTTSSettings
}

// SampleRate reports the PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the voice synthesis is billed against. Zoom TTS has no
// model to name.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.live.Voice.Or("")}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *httpSynthesizer) Settings() any { return s.live }

// payload is the synthesize request for text. A setting with no value is left
// out, which leaves it to Zoom.
func (s *httpSynthesizer) payload(text string) map[string]any {
	payload := map[string]any{"input": text, "response_format": ttsResponseFormat}
	if voice, ok := s.live.Voice.Value(); ok {
		payload["voice"] = voice
	}
	if speed, ok := s.live.Speed.Value(); ok {
		payload["speed"] = speed
	}
	if volume, ok := s.live.Volume.Value(); ok {
		payload["volume"] = volume
	}
	return payload
}

// RunTTS synthesizes text with Zoom TTS and streams its audio, at the rate Zoom
// synthesized it; the base resamples it to the service's and keeps the frames
// sample-aligned. Text longer than one request accepts is sent in several
// requests, split at whitespace.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	for _, piece := range splitText(text, maxInputChars) {
		if err := s.synthesize(ctx, piece, yield); err != nil {
			return err
		}
	}
	return nil
}

// synthesize sends one request and streams its audio. What fails is yielded as
// an error naming the request's tracking ID, and ends this request alone.
func (s *httpSynthesizer) synthesize(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	trackingID := zoomaiutil.NewTrackingID()
	fail := func(message string, cause error, category errclass.Category) error {
		if ctx.Err() != nil {
			// An interruption closed the request, which is how Zoom TTS is told
			// to stop synthesizing.
			return ctx.Err()
		}
		ef := frames.NewErrorFrame(message)
		ef.Err = cause
		ef.Category = category
		return yield(ef)
	}
	requestFailed := func(err error) error {
		return fail(fmt.Sprintf("Zoom TTS request failed (%s %s): %v", headerTrackingID, trackingID, err),
			err, errclass.Unset)
	}

	// Non-ASCII text stays UTF-8 rather than \u escapes, which keeps a request
	// of the longest input within the API's 32 KiB body limit.
	var body bytes.Buffer
	enc := json.NewEncoder(&body)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s.payload(text)); err != nil {
		return requestFailed(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, &body)
	if err != nil {
		return requestFailed(err)
	}
	zoomaiutil.SetRequestHeaders(req.Header, s.cfg.APIKey, trackingID)
	req.Header.Set("Content-Type", "application/json")

	// Leaving this function mid-stream, as an interruption or a failed stream
	// does, closes the connection, which stops synthesis on the server.
	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return requestFailed(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		detail := describeError(resp)
		return fail(fmt.Sprintf("Zoom TTS error: %s (%s %s)", detail, headerTrackingID, trackingID),
			nil, errclass.ClassifyHTTPStatusCode(resp.StatusCode))
	}
	slog.DebugContext(ctx, "zoom tts: synthesizing", headerTrackingID, trackingID)
	tts.StartUsageMetrics(ctx, text)

	emit := ttsstream.PCMYielder(yield, ttsSampleRate)
	buf := make([]byte, tts.ChunkSize(s.cfg.SampleRate))
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if err := emit(bytes.Clone(buf[:n])); err != nil {
				return err
			}
		}
		switch {
		case errors.Is(readErr, io.EOF):
			return nil
		case errors.Is(readErr, io.ErrUnexpectedEOF):
			// Once audio is streaming the status can't change, so a synthesis
			// failure ends the response early instead.
			return fail(fmt.Sprintf("Zoom TTS stream ended early (%s %s): %v",
				headerTrackingID, trackingID, readErr), readErr, errclass.Server)
		case readErr != nil:
			return requestFailed(readErr)
		}
	}
}

// describeError describes an error response, with Zoom's reason and message
// when its body has them.
func describeError(resp *http.Response) string {
	raw, _ := io.ReadAll(resp.Body)
	parts := []string{fmt.Sprintf("HTTP %d", resp.StatusCode)}
	var zoomError struct {
		Reason  any `json:"reason"`
		Message any `json:"message"`
	}
	if err := json.Unmarshal(raw, &zoomError); err != nil {
		// No body, or not a JSON object.
		if text := strings.TrimSpace(string(bytes.ToValidUTF8(raw, nil))); text != "" {
			parts = append(parts, text)
		}
		return strings.Join(parts, " ")
	}
	for _, part := range []any{zoomError.Reason, zoomError.Message} {
		if text := fmt.Sprint(part); part != nil && text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, " ")
}

// splitText splits text into pieces of at most limit code points, at
// whitespace if possible.
func splitText(text string, limit int) []string {
	runes := []rune(text)
	var pieces []string
	for len(runes) > limit {
		cut := -1
		for i := limit; i >= 0; i-- {
			if runes[i] == ' ' {
				cut = i
				break
			}
		}
		if cut <= 0 {
			cut = limit
		}
		pieces = append(pieces, string(runes[:cut]))
		runes = []rune(strings.TrimLeftFunc(string(runes[cut:]), isSpace))
	}
	if len(runes) > 0 {
		pieces = append(pieces, string(runes))
	}
	return pieces
}

// isSpace reports whether r is white space as the text is split on.
func isSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r', '\v', '\f':
		return true
	}
	return false
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.Describer      = (*httpSynthesizer)(nil)
)
