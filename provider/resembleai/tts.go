package resembleai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

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
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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
