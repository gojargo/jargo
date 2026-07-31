package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// errTTSProtocol is returned when Gradium reports a TTS error message.
//
//nolint:gochecknoglobals // sentinel error
var errTTSProtocol = errors.New("gradium: tts protocol error")

const (
	defaultTTSURL = "wss://api.gradium.ai/api/speech/tts"
	// defaultVoiceID is Gradium's default voice.
	defaultVoiceID = "_6Aslh2DxfmnRLmP"
	// ttsSampleRate is the fixed PCM rate of Gradium's TTS output.
	ttsSampleRate = 48000
	// ttsClientReqID labels the synthesis context on the wire.
	ttsClientReqID = "jargo"
)

// TTSConfig configures the Gradium TTS service.
type TTSConfig struct {
	// APIKey is the Gradium API key, sent as the x-api-key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice identifier; empty uses a default voice.
	VoiceID string
	// JSONConfig is an optional JSON configuration string forwarded to the model;
	// empty omits it.
	JSONConfig string
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Gradium TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultTTSURL
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	return tts.New("GradiumTTS", &ttsSynthesizer{cfg: cfg})
}

type ttsSynthesizer struct {
	cfg TTSConfig
}

// SampleRate reports Gradium's fixed 48 kHz PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return ttsSampleRate }

// ttsMessage is the subset of a Gradium TTS WebSocket message we read.
type ttsMessage struct {
	Type    string `json:"type"`
	Audio   string `json:"audio"`
	Message string `json:"message"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks
// until the server signals end-of-stream.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	header := http.Header{}
	header.Set("x-api-key", s.cfg.APIKey)

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

// request sends the setup, text, and end-of-stream messages for one sentence.
func (s *ttsSynthesizer) request(ctx context.Context, conn *websocket.Conn, text string) error {
	setup := map[string]any{
		msgType:           "setup",
		"output_format":   encPCM,
		"voice_id":        s.cfg.VoiceID,
		"close_ws_on_eos": false,
		keyClientReqID:    ttsClientReqID,
	}
	if s.cfg.JSONConfig != "" {
		setup["json_config"] = s.cfg.JSONConfig
	}
	if err := s.send(ctx, conn, setup); err != nil {
		return err
	}

	txt := map[string]any{
		msgType:        msgText,
		msgText:        text,
		keyClientReqID: ttsClientReqID,
	}
	if err := s.send(ctx, conn, txt); err != nil {
		return err
	}

	// Signal that no more text is coming so the server flushes and closes the
	// context with an end_of_stream message.
	eos := map[string]any{msgType: msgEndStream, keyClientReqID: ttsClientReqID}
	return s.send(ctx, conn, eos)
}

func (s *ttsSynthesizer) send(ctx context.Context, conn *websocket.Conn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// receive streams audio chunks to emit until end-of-stream or an error.
func (s *ttsSynthesizer) receive(ctx context.Context, conn *websocket.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m ttsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case msgAudio:
			pcm, err := base64.StdEncoding.DecodeString(m.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case msgEndStream:
			return nil
		case msgError:
			return fmt.Errorf("%w: %s", errTTSProtocol, m.Message)
		}
	}
}
