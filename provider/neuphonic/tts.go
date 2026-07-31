package neuphonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// NewTTS builds a Neuphonic TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Speed == 0 {
		cfg.Speed = defaultSpeed
	}
	return tts.New("NeuphonicTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// endpoint builds the /speak WebSocket URL with the configured query params.
func (s *synthesizer) endpoint() string {
	lang := neuphonicLanguage(s.cfg.Language)
	q := url.Values{}
	q.Set("lang_code", lang)
	q.Set("speed", strconv.FormatFloat(s.cfg.Speed, 'g', -1, 64))
	q.Set("encoding", s.cfg.Encoding)
	q.Set("sampling_rate", strconv.Itoa(s.cfg.SampleRate))
	if s.cfg.VoiceID != "" {
		q.Set("voice_id", s.cfg.VoiceID)
	}
	return fmt.Sprintf("%s/speak/%s?%s", s.cfg.URL, lang, q.Encode())
}

// wsMessage is the subset of a Neuphonic WebSocket message we read.
type wsMessage struct {
	Data   *wsData  `json:"data"`
	Errors []string `json:"errors"`
}

// wsData carries a single audio chunk; Stop marks the final message.
type wsData struct {
	Audio string `json:"audio"`
	Stop  bool   `json:"stop"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	header := http.Header{}
	header.Set("x-api-key", s.cfg.APIKey)

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

func (s *synthesizer) request(ctx context.Context, conn *websocket.Conn, text string) error {
	payload, err := json.Marshal(map[string]any{"text": text + " <STOP>"})
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
		if len(m.Errors) > 0 {
			return fmt.Errorf("%w: %s", errProtocol, strings.Join(m.Errors, "; "))
		}
		if m.Data == nil {
			continue
		}
		if m.Data.Audio != "" {
			pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		}
		if m.Data.Stop {
			return nil
		}
	}
}

// neuphonicLanguage maps a Language to Neuphonic's language code. The zero value
// falls back to English; unmapped languages fall back to their base code.
func neuphonicLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "":
		return "en"
	case "hi":
		return "HI"
	default:
		return base
	}
}
