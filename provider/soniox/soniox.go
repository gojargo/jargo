// Package soniox provides Soniox real-time streaming speech-to-text over its
// WebSocket API. Soniox streams tokens, each marked final or provisional, and
// signals end-of-turn with a special "<end>" token; this service surfaces
// provisional text as interims and the finalized utterance with EndOfTurn set.
package soniox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

const (
	defaultURL   = "wss://stt-rt.soniox.com/transcribe-websocket"
	defaultModel = "stt-rt-v5"
	// readLimit bounds a single inbound message.
	readLimit = 1 << 20
	// endToken and finToken are Soniox's in-band turn-boundary markers.
	endToken = "<end>"
	finToken = "<fin>"
)

// errServer wraps an error reported by Soniox.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("soniox: server error")

// Config configures the Soniox STT service.
type Config struct {
	// APIKey is the Soniox API key, sent in the config handshake. Required.
	APIKey string `validate:"required"`
	// URL overrides the real-time endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a current default.
	Model string
	// Language hints the spoken language; the zero value lets Soniox auto-detect.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// EnableEndpointDetection emits an end-of-turn marker on detected silence;
	// nil defaults to true.
	EnableEndpointDetection *bool
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewSTT builds a Soniox streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return stt.NewStream("SonioxSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
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
		"model":                     c.cfg.Model,
		"audio_format":              "s16le",
		"sample_rate":               sampleRate,
		"num_channels":              1,
		"enable_endpoint_detection": c.cfg.EnableEndpointDetection == nil || *c.cfg.EnableEndpointDetection,
	}
	if lang := c.cfg.Language.BaseCode(); lang != "" {
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
