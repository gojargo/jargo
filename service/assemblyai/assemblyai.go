// Package assemblyai is a streaming speech-to-text service backed by
// AssemblyAI's Universal-Streaming (v3) WebSocket. It streams audio and pushes
// InterimTranscriptionFrames and finalized TranscriptionFrames downstream.
//
// A turn is finalized once AssemblyAI reports a formatted end-of-turn, which
// marks the end of the user's speech.
package assemblyai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
)

const (
	wsBase = "wss://streaming.assemblyai.com/v3/ws"
	// readLimit bounds a single WebSocket message; long turns carry many words.
	readLimit = 1 << 20
)

// Config configures the AssemblyAI STT service.
type Config struct {
	// APIKey is the AssemblyAI API key; empty uses the ASSEMBLYAI_API_KEY env var.
	APIKey string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// NewSTT builds an AssemblyAI streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ASSEMBLYAI_API_KEY")
	}
	return stt.NewStream("AssemblyAISTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// Connect opens the streaming WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := url.Values{}
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("encoding", "pcm_s16le")
	q.Set("format_turns", "true")

	header := http.Header{}
	header.Set("Authorization", c.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, wsBase+"?"+q.Encode(), &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)
	return &stream{conn: conn, ctx: ctx}, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// turnMessage is the subset of a v3 Turn message we read.
type turnMessage struct {
	Type            string `json:"type"`
	Transcript      string `json:"transcript"`
	EndOfTurn       bool   `json:"end_of_turn"`
	TurnIsFormatted bool   `json:"turn_is_formatted"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next Turn message and maps it to a result. A turn is final only
// once it is both ended and formatted; the unformatted end-of-turn and ongoing
// turns are interim.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m turnMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "Turn" || m.Transcript == "" {
			continue
		}
		final := m.EndOfTurn && m.TurnIsFormatted
		return []stt.Result{{Text: m.Transcript, Final: final, EndOfTurn: final}}, nil
	}
}

// Close terminates the session and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"Terminate"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
