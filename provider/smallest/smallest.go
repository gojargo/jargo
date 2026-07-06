// Package smallest is a streaming text-to-speech service backed by Smallest AI's
// Waves WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream.
package smallest

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

// errProtocol is returned when Smallest reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("smallest: protocol error")

const (
	defaultURL        = "wss://api.smallest.ai/waves/v1/tts/live"
	defaultModel      = "lightning_v3.1_pro"
	defaultVoice      = "meher"
	defaultLanguage   = "en"
	defaultSampleRate = 24000
)

// Config configures the Smallest AI TTS service.
type Config struct {
	// APIKey is the Smallest AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the Smallest model id; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default voice.
	Voice string
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
	// Speed multiplies the speaking rate; nil uses the service default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Smallest AI TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("SmallestTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// wsMessage is the subset of a Smallest WebSocket message we read.
type wsMessage struct {
	Status string `json:"status"`
	Data   struct {
		Audio string `json:"audio"`
	} `json:"data"`
	Error string `json:"error"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	header.Set("X-Source", "jargo")

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
	msg := map[string]any{
		"text":          text,
		"voice_id":      s.cfg.Voice,
		"model":         s.cfg.Model,
		"language":      s.cfg.Language,
		"sample_rate":   s.cfg.SampleRate,
		"output_format": "pcm",
	}
	if s.cfg.Speed != nil {
		msg["speed"] = *s.cfg.Speed
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
		switch m.Status {
		case "chunk":
			pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "complete":
			return nil
		case "error":
			return fmt.Errorf("%w: %s", errProtocol, m.Error)
		}
	}
}
