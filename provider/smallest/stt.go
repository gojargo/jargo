package smallest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	defaultSTTURL      = "wss://api.smallest.ai"
	defaultSTTEncoding = "linear16"
)

// STTConfig configures the Smallest AI streaming STT (Pulse) service.
type STTConfig struct {
	// APIKey is the Smallest AI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the STT WebSocket base; empty uses the hosted endpoint.
	BaseURL string
	// Language is the audio language code; empty lets the service auto-detect.
	Language string
	// Encoding is the PCM encoding sent to the API; empty uses "linear16".
	Encoding string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Smallest AI streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultSTTURL
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultSTTEncoding
	}
	return stt.NewStream("SmallestSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg STTConfig
}

// Connect opens a Pulse streaming session for the given input sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := url.Values{}
	if c.cfg.Language != "" {
		q.Set("language", c.cfg.Language)
	}
	q.Set("encoding", c.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	endpoint := c.cfg.BaseURL + "/waves/v1/pulse/get_text?" + q.Encode()

	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("X-Source", "jargo")

	conn, err := wsutil.Dial(ctx, endpoint, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &sttStream{ctx: ctx, conn: conn}, nil
}

type sttStream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// sttMessage is the subset of a Pulse transcription message we read.
type sttMessage struct {
	IsFinal    bool   `json:"is_final"`
	Transcript string `json:"transcript"`
	Language   string `json:"language"`
}

// Send writes a chunk of 16-bit mono PCM to the session.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcription; a final transcript marks the end of turn.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Transcript == "" {
			continue
		}
		return []stt.Result{{
			Text:      m.Transcript,
			Final:     m.IsFinal,
			EndOfTurn: m.IsFinal,
			Language:  m.Language,
		}}, nil
	}
}

// Close tears down the session.
func (s *sttStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
