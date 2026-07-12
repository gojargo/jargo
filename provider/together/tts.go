package together

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

const (
	ttsURL          = "wss://api.together.ai/v1/audio/speech/websocket"
	defaultTTSModel = "hexgrad/Kokoro-82M"
	defaultTTSVoice = "af_heart"
	// ttsSampleRate is Together's fixed TTS output rate; it streams 24 kHz PCM.
	ttsSampleRate = 24000
)

// errTTS wraps an error reported by the Together TTS session.
//
//nolint:gochecknoglobals // sentinel error
var errTTS = errors.New("together: tts error")

// TTSConfig configures the Together AI streaming TTS service, which streams 24 kHz
// mono PCM over an OpenAI-compatible realtime WebSocket.
type TTSConfig struct {
	// APIKey is the Together AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the TTS model; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default.
	Voice string
	// MaxPartialLength caps the partial text length for streaming; nil omits it.
	MaxPartialLength *int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Together AI streaming TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = ttsURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultTTSVoice
	}
	return tts.New("TogetherTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg TTSConfig
}

// SampleRate reports Together's fixed PCM output rate.
func (s *synthesizer) SampleRate() int { return ttsSampleRate }

// ttsEvent is the subset of a Together TTS event we read.
type ttsEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// endpoint builds the TTS WebSocket URL with model and voice query parameters.
func (s *synthesizer) endpoint() string {
	url := fmt.Sprintf("%s?model=%s&voice=%s", s.cfg.URL, s.cfg.Model, s.cfg.Voice)
	if s.cfg.MaxPartialLength != nil {
		url += fmt.Sprintf("&max_partial_length=%d", *s.cfg.MaxPartialLength)
	}
	return url
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.endpoint(), header, wsutil.DefaultReadLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

// request pins the voice, then buffers the text and commits it for synthesis.
func (s *synthesizer) request(ctx context.Context, conn *websocket.Conn, text string) error {
	msgs := []map[string]any{
		{msgType: "tts_session.updated", "session": map[string]any{"voice": s.cfg.Voice}},
		{msgType: "input_text_buffer.append", "text": text},
		{msgType: "input_text_buffer.commit"},
	}
	for _, m := range msgs {
		payload, err := json.Marshal(m)
		if err != nil {
			return err
		}
		if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
			return err
		}
	}
	return nil
}

// receive streams audio deltas until the session reports completion.
func (s *synthesizer) receive(ctx context.Context, conn *websocket.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var evt ttsEvent
		if json.Unmarshal(data, &evt) != nil {
			continue
		}
		switch evt.Type {
		case "conversation.item.audio_output.delta":
			if evt.Delta == "" {
				continue
			}
			pcm, err := base64.StdEncoding.DecodeString(evt.Delta)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "conversation.item.audio_output.done":
			return nil
		case "conversation.item.tts.failed", "error":
			return fmt.Errorf("%w: %s", errTTS, evt.Error.Message)
		}
	}
}
