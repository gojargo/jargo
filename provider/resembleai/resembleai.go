// Package resembleai is a streaming text-to-speech service backed by Resemble
// AI's WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream. The request sets
// no_audio_header so the audio arrives as headerless 16-bit PCM.
package resembleai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// errProtocol is returned when Resemble reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("resembleai: protocol error")

const (
	defaultURL        = "wss://websocket.cluster.resemble.ai/stream"
	defaultSampleRate = 22050
	defaultPrecision  = "PCM_16"
)

// Config configures the Resemble AI TTS service.
type Config struct {
	// APIKey is the Resemble AI API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice UUID to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 22.05 kHz.
	SampleRate int
	// Precision is the audio precision; empty uses "PCM_16".
	Precision string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Resemble AI TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Precision == "" {
		cfg.Precision = defaultPrecision
	}
	return tts.New("ResembleTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// wsMessage is the subset of a Resemble WebSocket message we read.
type wsMessage struct {
	Type         string `json:"type"`
	AudioContent string `json:"audio_content"`
	Message      string `json:"message"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

func (s *synthesizer) request(ctx context.Context, conn *websocket.Conn, text string) error {
	// A fresh connection carries one request, so a fixed request_id is fine.
	msg := map[string]any{
		"voice_uuid":      s.cfg.Voice,
		"data":            text,
		"binary_response": false,
		"request_id":      1,
		"output_format":   "wav",
		"sample_rate":     s.cfg.SampleRate,
		"precision":       s.cfg.Precision,
		"no_audio_header": true,
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

func (s *synthesizer) receive(ctx context.Context, conn *websocket.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m wsMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		switch m.Type {
		case "audio":
			pcm, err := base64.StdEncoding.DecodeString(m.AudioContent)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "audio_end":
			return nil
		case "error":
			return fmt.Errorf("%w: %s", errProtocol, m.Message)
		}
	}
}
