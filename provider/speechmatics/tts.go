// Speechmatics text-to-speech over its HTTP streaming API. The service posts the
// text and streams the raw 16 kHz PCM response downstream.

package speechmatics

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	ttsDefaultURL   = "https://preview.tts.speechmatics.com"
	ttsDefaultVoice = "sarah"
	// ttsSampleRate is the fixed PCM rate of Speechmatics TTS audio.
	ttsSampleRate = 16000
)

// TTSConfig configures the Speechmatics TTS service.
type TTSConfig struct {
	// APIKey is the Speechmatics API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the TTS endpoint; empty uses the hosted endpoint.
	BaseURL string
	// Voice is the voice id in the request path; empty uses "sarah".
	Voice string
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Speechmatics TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = ttsDefaultURL
	}
	if cfg.Voice == "" {
		cfg.Voice = ttsDefaultVoice
	}
	return tts.New("SpeechmaticsTTS", &ttsSynthesizer{cfg: cfg, http: &http.Client{}})
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports Speechmatics' fixed 16 kHz PCM rate.
func (s *ttsSynthesizer) SampleRate() int { return ttsSampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *ttsSynthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := json.Marshal(map[string]any{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}

// endpoint builds the generate URL for the configured voice and output format.
func (s *ttsSynthesizer) endpoint() string {
	q := url.Values{}
	q.Set("output_format", "pcm_"+strconv.Itoa(ttsSampleRate))
	return s.cfg.BaseURL + "/generate/" + url.PathEscape(s.cfg.Voice) + "?" + q.Encode()
}
