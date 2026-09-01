package smallest

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

// NewTTS builds a Smallest AI TTS service.
func NewTTS(cfg Config) *tts.Base {
	return tts.New("SmallestTTS", &synthesizer{cfg: withDefaults(cfg)})
}

// withDefaults fills in what the caller left unset. The voice is resolved from
// the model rather than fixed, since the two models serve different voice
// catalogs and one default for both would name a voice the other does not
// serve. A model with no default of its own leaves the voice to Smallest.
func withDefaults(cfg Config) Config {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = modelDefaultVoices[cfg.Model]
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return cfg
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
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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

func (s *synthesizer) request(ctx context.Context, conn *wsutil.Conn, text string) error {
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

func (s *synthesizer) receive(ctx context.Context, conn *wsutil.Conn, emit func(pcm []byte) error) error {
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
