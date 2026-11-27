package bland

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
	"github.com/gojargo/jargo/internal/ttsstream"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

// defaultHTTPURL is the Bland API base.
const defaultHTTPURL = "https://api.bland.ai/v2"

// HTTPTTSConfig configures the Bland HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Bland API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the API base; empty uses "https://api.bland.ai/v2".
	URL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 48 kHz. A rate Bland
	// does not render (it renders 8000, 16000, 24000, 44100 and 48000) is
	// synthesized at 48 kHz and resampled to this one.
	SampleRate int
	// Voice is the voice id, which also sets the model; empty uses Bland's
	// default voice.
	Voice string
	// Expressiveness is from 0.0 to 1.0: higher values produce more varied
	// intonation. Nil keeps Bland's default.
	Expressiveness *float64
	// Stability is from 0.0 to 1.0: higher values produce more consistent
	// delivery. Nil keeps Bland's default.
	Stability *float64
	// AutoFormatting rewrites numbers into the form the voice reads best before
	// synthesis. Nil leaves it off.
	AutoFormatting *bool
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// HTTPTTSService is the Bland text-to-speech service NewHTTPTTS builds.
type HTTPTTSService struct {
	*tts.Base
}

// NewHTTPTTS builds a Bland TTS service over the /v2/tts endpoint, which takes
// the complete text in one request. Voice agents should prefer NewTTS, which
// streams text into a realtime session.
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
	if cfg.URL == "" {
		cfg.URL = defaultHTTPURL
	}
	cfg.URL = strings.TrimRight(cfg.URL, "/")
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{
		cfg:       cfg,
		blandRate: resolveSampleRate(cfg.SampleRate),
		live:      newTTSSettings(cfg.Voice, cfg.Expressiveness, cfg.Stability, cfg.AutoFormatting),
	}
	return tts.New("BlandHTTPTTS", s, opts...), s
}

// httpSynthesizer speaks each sentence with one request to /v2/tts.
type httpSynthesizer struct {
	cfg HTTPTTSConfig
	// blandRate is the rate Bland synthesizes at, resolved from the service's.
	blandRate int
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings
}

// SampleRate reports the PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Bland voice synthesis is billed against.
func (s *httpSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.live.Voice.Or("")}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Each request is built from it, so a change
// applies from the next sentence on.
func (s *httpSynthesizer) Settings() any { return s.live }

// payload is the /v2/tts body for text.
func (s *httpSynthesizer) payload(text string) map[string]any {
	payload := map[string]any{
		"text":   text,
		keyVoice: s.live.Voice.AsAny(),
		"audio": map[string]any{
			"encoding":    encoding,
			"sample_rate": s.blandRate,
			// The body is streamed straight into audio frames, so a
			// container's header would be read as the first samples.
			"container": "raw",
		},
	}
	applyRequestOptions(payload, s.live)
	return payload
}

// RunTTS requests speech for text and streams the audio as it arrives, at the
// rate Bland synthesized it; the base resamples it to the service's. A failure
// is yielded as an error in the context's stream.
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL+"/tts", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return yield(frames.NewErrorFrame(errorMessage(resp)))
	}
	tts.StartUsageMetrics(ctx, text)

	emit := ttsstream.PCMYielder(yield, s.blandRate)
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

// errorMessage unwraps the v2 {"error": {"code", "message"}} envelope of a
// failed response, falling back to the raw body.
func errorMessage(resp *http.Response) string {
	raw, _ := io.ReadAll(resp.Body)
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		body := strings.ToValidUTF8(string(raw), "")
		return fmt.Sprintf("Error getting audio (status: %d, error: %s)", resp.StatusCode, body)
	}
	if detail := errorDetail(payload); detail != "" {
		return fmt.Sprintf("Error getting audio (status: %d, error: %s)", resp.StatusCode, detail)
	}
	return fmt.Sprintf("Error getting audio (status: %d, error: %v)", resp.StatusCode, payload)
}

// errorDetail is the code and the message of a v2 error envelope, joined, or
// "" when payload is not one or carries neither.
func errorDetail(payload any) string {
	envelope, _ := payload.(map[string]any)
	detail, _ := envelope["error"].(map[string]any)
	var parts []string
	for _, key := range []string{"code", "message"} {
		if v, ok := detail[key]; ok && v != nil && v != "" {
			parts = append(parts, fmt.Sprint(v))
		}
	}
	return strings.Join(parts, ": ")
}

// Compile-time checks that the HTTP synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*httpSynthesizer)(nil)
	_ tts.Describer      = (*httpSynthesizer)(nil)
)
