// Package fal is a batch speech-to-text service backed by Fal's Wizper API. A
// turn detector upstream delimits each utterance; the whole segment is uploaded
// as a WAV data URI and transcribed in one request.
package fal

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("fal: unexpected status")

const defaultEndpoint = "https://fal.run/fal-ai/wizper"

// Config configures the Fal Wizper transcription service.
type Config struct {
	// APIKey is the Fal API key. Required.
	APIKey string `validate:"required"`
	// Endpoint overrides the Wizper run endpoint; empty uses the hosted endpoint.
	Endpoint string
	// Language of the audio; the zero value omits it (auto-detect). Mapped to the
	// base code.
	Language language.Language
	// Task selects the operation ("transcribe" or "translate"); empty omits it.
	Task string
	// Version pins the Wizper model version; empty omits it.
	Version string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewSTT builds a Fal Wizper transcription service. It is segmented: a turn
// detector upstream delimits each utterance, transcribed in one request.
func NewSTT(cfg Config) *stt.SegmentService {
	if cfg.Endpoint == "" {
		cfg.Endpoint = defaultEndpoint
	}
	return stt.NewSegment("FalSTT", &transcriber{cfg: cfg, http: &http.Client{}}, cfg.SampleRate)
}

type transcriber struct {
	cfg  Config
	http *http.Client
}

// Transcribe uploads the segment as a WAV data URI and returns the transcript.
func (t *transcriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	wav := stt.WAV(audio, sampleRate, 1)
	payload := map[string]any{
		"audio_url": "data:audio/x-wav;base64," + base64.StdEncoding.EncodeToString(wav),
	}
	if lang := t.cfg.Language.BaseCode(); lang != "" {
		payload["language"] = lang
	}
	if t.cfg.Task != "" {
		payload["task"] = t.cfg.Task
	}
	if t.cfg.Version != "" {
		payload["version"] = t.cfg.Version
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.cfg.Endpoint, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Key "+t.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return strings.TrimSpace(out.Text), nil
}
