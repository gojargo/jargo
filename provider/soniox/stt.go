package soniox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// NewSTT builds a Soniox streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return stt.NewStream("SonioxSTT", &connector{cfg: cfg, live: newSTTSettings(cfg)}, cfg.SampleRate)
}

type connector struct {
	cfg Config
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Soniox treats as changeable and that jargo's
	// configuration carries.
	live *settings.STT
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg Config) *settings.STT {
	s := &settings.STT{}
	s.Model = settings.Set(cfg.Model)
	s.Language = settings.Set(cfg.Language.BaseCode())
	return s
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Soniox is told all of this in the handshake that opens the session, so a
// change reaches it only by opening another.
func (c *connector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ServiceLanguage names a language the way Soniox does, by its base code.
func (c *connector) ServiceLanguage(l language.Language) string {
	return l.BaseCode()
}

// Connect dials the WebSocket and sends the config handshake (which carries the
// API key).
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	conn, resp, err := websocket.Dial(ctx, c.cfg.URL, nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)

	if err := conn.Write(ctx, websocket.MessageText, c.config(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "config failed")
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx}, nil
}

func (c *connector) config(sampleRate int) []byte {
	cfg := map[string]any{
		"api_key":                   c.cfg.APIKey,
		"model":                     c.live.Model.Or(c.cfg.Model),
		"audio_format":              "s16le",
		"sample_rate":               sampleRate,
		"num_channels":              1,
		"enable_endpoint_detection": c.cfg.EnableEndpointDetection == nil || *c.cfg.EnableEndpointDetection,
	}
	if lang := c.live.Language.Or(c.cfg.Language.BaseCode()); lang != "" {
		cfg["language_hints"] = []string{lang}
	}
	b, _ := json.Marshal(cfg) //nolint:errchkjson // map of known-serializable values
	return b
}

type stream struct {
	conn     *websocket.Conn
	ctx      context.Context
	writeMu  sync.Mutex
	finalBuf string
}

// Send writes a chunk of PCM as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next batch of tokens. Final tokens accumulate; a "<end>" or
// "<fin>" marker flushes the utterance with EndOfTurn set. Provisional tokens
// surface as an interim result.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		if m.ErrorCode != 0 {
			return nil, fmt.Errorf("%w: %s", errServer, m.ErrorMessage)
		}
		if m.Finished {
			return nil, io.EOF
		}

		var provisional strings.Builder
		endOfTurn := false
		for _, t := range m.Tokens {
			switch {
			case t.Text == endToken || t.Text == finToken:
				endOfTurn = endOfTurn || t.IsFinal
			case t.IsFinal:
				s.finalBuf += t.Text
			default:
				provisional.WriteString(t.Text)
			}
		}

		if endOfTurn && s.finalBuf != "" {
			text := strings.TrimSpace(s.finalBuf)
			s.finalBuf = ""
			return []stt.Result{{Text: text, Final: true, EndOfTurn: true}}, nil
		}
		if interim := s.finalBuf + provisional.String(); strings.TrimSpace(interim) != "" {
			return []stt.Result{{Text: strings.TrimSpace(interim), Final: false}}, nil
		}
	}
}

// Close signals end of audio and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	// An empty frame tells Soniox the audio is complete.
	_ = s.conn.Write(context.Background(), websocket.MessageBinary, []byte{})
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// message is the subset of a Soniox result message we use.
type message struct {
	Tokens       []token `json:"tokens"`
	Finished     bool    `json:"finished"`
	ErrorCode    int     `json:"error_code"`
	ErrorMessage string  `json:"error_message"`
}

type token struct {
	Text    string `json:"text"`
	IsFinal bool   `json:"is_final"`
}
