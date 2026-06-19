// Package gladia is a streaming speech-to-text service backed by Gladia's Live
// STT v2 API. A session is opened with a REST call that returns a tokenized
// WebSocket URL; audio then streams over the socket and transcripts come back as
// interim and final utterances.
package gladia

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
)

// errStatus is returned when the session REST call responds with a non-2xx
// status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("gladia: unexpected status")

const (
	liveURL = "https://api.gladia.io/v2/live"
	model   = "solaria-1"
	// readLimit bounds a single WebSocket message.
	readLimit = 1 << 20
)

// Config configures the Gladia STT service.
type Config struct {
	// APIKey is the Gladia API key; empty uses the GLADIA_API_KEY env var.
	APIKey string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// NewSTT builds a Gladia streaming STT service. It works best behind a turn
// detector: Gladia finalizes per utterance rather than per turn.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("GLADIA_API_KEY")
	}
	return stt.NewStream("GladiaSTT", &connector{cfg: cfg, http: &http.Client{}}, cfg.SampleRate)
}

type connector struct {
	cfg  Config
	http *http.Client
}

// Connect initializes a session over REST then dials the returned WebSocket.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	wsURL, err := c.initSession(ctx, sampleRate)
	if err != nil {
		return nil, err
	}
	conn, resp, err := websocket.Dial(ctx, wsURL, nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)
	return &stream{conn: conn, ctx: ctx}, nil
}

func (c *connector) initSession(ctx context.Context, sampleRate int) (string, error) {
	body, err := json.Marshal(map[string]any{
		"encoding":    "wav/pcm",
		"sample_rate": sampleRate,
		"bit_depth":   16,
		"channels":    1,
		"model":       model,
		"messages_config": map[string]any{
			"receive_partial_transcripts": true,
			"receive_final_transcripts":   true,
		},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, liveURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("x-gladia-key", c.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var out struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.URL, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// message is the subset of a Gladia transcript message we read.
type message struct {
	Type string `json:"type"`
	Data struct {
		IsFinal   bool `json:"is_final"`
		Utterance struct {
			Text     string `json:"text"`
			Language string `json:"language"`
		} `json:"utterance"`
	} `json:"data"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcript message and maps it to a result.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "transcript" || m.Data.Utterance.Text == "" {
			continue
		}
		return []stt.Result{{
			Text:      m.Data.Utterance.Text,
			Final:     m.Data.IsFinal,
			EndOfTurn: m.Data.IsFinal,
			Language:  m.Data.Utterance.Language,
		}}, nil
	}
}

// Close stops the session and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"stop_recording"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
