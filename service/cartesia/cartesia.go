// Package cartesia is a streaming text-to-speech service backed by Cartesia's
// TTS WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream.
package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/tts"
)

// errProtocol is returned when Cartesia reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("cartesia: protocol error")

const (
	wsURL = "wss://api.cartesia.ai/tts/websocket"
	// apiVersion pins the Cartesia API version; it is required on every request.
	apiVersion   = "2025-04-16"
	defaultModel = "sonic-2"
	// defaultVoiceID is a public Cartesia voice.
	defaultVoiceID = "694f9389-aac1-45b6-b726-9d9369183238"
	// sampleRate is the PCM rate jargo requests from Cartesia.
	sampleRate = 24000
	// readLimit bounds a single WebSocket message; audio chunks arrive base64.
	readLimit = 1 << 20
)

// Config configures the Cartesia TTS service.
type Config struct {
	// APIKey is the Cartesia API key; empty uses the CARTESIA_API_KEY env var.
	APIKey string
	// Model is the Cartesia model id; empty uses a default.
	Model string
	// VoiceID is the voice id; empty uses a default public voice.
	VoiceID string
}

// NewTTS builds a Cartesia TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("CARTESIA_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	return tts.New("CartesiaTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return sampleRate }

// wsMessage is the subset of a Cartesia WebSocket message we read.
type wsMessage struct {
	Type    string `json:"type"`
	Data    string `json:"data"`
	Message string `json:"message"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	header := http.Header{}
	header.Set("X-API-Key", s.cfg.APIKey)
	header.Set("Cartesia-Version", apiVersion)

	conn, resp, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()
	conn.SetReadLimit(readLimit)

	if err := s.request(ctx, conn, text); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

func (s *synthesizer) request(ctx context.Context, conn *websocket.Conn, text string) error {
	payload, err := json.Marshal(map[string]any{
		"model_id":   s.cfg.Model,
		"transcript": text,
		"voice":      map[string]any{"mode": "id", "id": s.cfg.VoiceID},
		"output_format": map[string]any{
			"container":   "raw",
			"encoding":    "pcm_s16le",
			"sample_rate": sampleRate,
		},
		"context_id": "jargo",
		"continue":   false,
	})
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
		case "chunk":
			pcm, err := base64.StdEncoding.DecodeString(m.Data)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "done":
			return nil
		case "error":
			return fmt.Errorf("%w: %s", errProtocol, m.Message)
		}
	}
}
