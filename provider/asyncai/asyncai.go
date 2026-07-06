// Package asyncai is a streaming text-to-speech service backed by Async's TTS
// WebSocket. For each sentence it opens a connection, sends an init message and
// the transcript, and streams the raw PCM audio chunks downstream.
package asyncai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	defaultURL        = "wss://api.async.com/text_to_speech/websocket/ws"
	defaultVersion    = "v1"
	defaultModel      = "async_flash_v1.0"
	defaultLanguage   = "en"
	defaultEncoding   = "pcm_s16le"
	defaultContainer  = "raw"
	defaultSampleRate = 24000
)

// Config configures the Async TTS service.
type Config struct {
	// APIKey is the Async API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice id (UUID) to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the API version; empty uses a default.
	Version string
	// Model is the Async model id; empty uses a default.
	Model string
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds an Async TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("AsyncAITTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// wsMessage is the subset of an Async WebSocket message we read.
type wsMessage struct {
	Final bool   `json:"final"`
	Audio string `json:"audio"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	endpoint := s.cfg.URL + "?api_key=" + url.QueryEscape(s.cfg.APIKey) + "&version=" + url.QueryEscape(s.cfg.Version)
	conn, err := wsutil.Dial(ctx, endpoint, http.Header{}, wsutil.DefaultReadLimit)
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
	init := map[string]any{
		"model_id": s.cfg.Model,
		"voice":    map[string]any{"mode": "id", "id": s.cfg.Voice},
		"output_format": map[string]any{
			"container":   defaultContainer,
			"encoding":    defaultEncoding,
			"sample_rate": s.cfg.SampleRate,
		},
		"language": s.cfg.Language,
	}
	if err := writeJSON(ctx, conn, init); err != nil {
		return err
	}
	// force flushes the single sentence immediately rather than waiting for more.
	return writeJSON(ctx, conn, map[string]any{"transcript": text, "context_id": "jargo", "force": true})
}

func writeJSON(ctx context.Context, conn *websocket.Conn, msg map[string]any) error {
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
		if m.Audio != "" {
			pcm, err := base64.StdEncoding.DecodeString(m.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		}
		if m.Final {
			return nil
		}
	}
}
