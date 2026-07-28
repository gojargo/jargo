package cartesia

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	defaultTurnsURL   = "wss://api.cartesia.ai/stt/turns/websocket"
	defaultTurnsModel = "ink-2"
	// turnsTTFSP99 is the time-to-final-segment P99 latency reported downstream.
	// The server ends a turn itself, so the final transcript arrives with the
	// boundary rather than after a separate finalize.
	turnsTTFSP99 = 400 * time.Millisecond
)

// Cartesia turn-detection message types.
const (
	// turnsConnected acknowledges the session.
	turnsConnected = "connected"
	// turnsStart marks the beginning of a user turn.
	turnsStart = "turn.start"
	// turnsUpdate carries the turn's transcript so far. Transcripts are
	// cumulative, so each update supersedes the previous one.
	turnsUpdate = "turn.update"
	// turnsEnd marks the end of a user turn and carries its final transcript.
	turnsEnd = "turn.end"
	// turnsEagerEnd is a predicted end of turn, which turnsResume may retract.
	turnsEagerEnd = "turn.eager_end"
	// turnsResume retracts an eager end: the user carried on speaking.
	turnsResume = "turn.resume"
	// turnsError reports a server-side failure.
	turnsError = "error"
)

// TurnsSTTConfig configures the Cartesia turn-detecting STT service.
type TurnsSTTConfig struct {
	// APIKey is the Cartesia API key, sent as the X-API-Key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the turn-detection WebSocket endpoint; empty uses the hosted
	// endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Model is the transcription model; empty uses a default.
	Model string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TurnsSTTConfig) Validate() error { return validate.Struct(c) }

// NewTurnsSTT builds a Cartesia turn-detecting STT service. Unlike NewSTT, the
// server decides where a turn begins and ends and reports those boundaries, so
// the pipeline does not need its own end-of-turn detection. The service
// recommends external turn strategies accordingly.
func NewTurnsSTT(cfg TurnsSTTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultTurnsURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultTurnsModel
	}
	return stt.NewStream("CartesiaTurnsSTT", &turnsConnector{cfg: cfg}, cfg.SampleRate)
}

type turnsConnector struct {
	cfg TurnsSTTConfig
}

// Metadata tells downstream processors the service detects turns itself, so the
// user aggregator adopts external turn strategies rather than running its own.
func (c *turnsConnector) Metadata() stt.Metadata {
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnExternal,
		TTFSP99:              turnsTTFSP99,
		Model:                c.cfg.Model,
	}
}

// Connect opens a turn-detection session and waits for the server to
// acknowledge it.
func (c *turnsConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := url.Values{}
	q.Set("model", c.cfg.Model)
	q.Set("encoding", defaultSTTEncoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))

	header := http.Header{}
	header.Set("X-API-Key", c.cfg.APIKey)
	header.Set("Cartesia-Version", c.cfg.Version)

	conn, err := wsutil.Dial(ctx, c.cfg.URL+"?"+q.Encode(), header, readLimit)
	if err != nil {
		return nil, err
	}
	return &turnsStream{conn: conn, ctx: ctx}, nil
}

type turnsStream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// turnsMessage is the subset of a Cartesia turn message we read.
type turnsMessage struct {
	Type string `json:"type"`
	// Transcript is the turn's text so far. It is cumulative, not a delta.
	Transcript string `json:"transcript"`
	Message    string `json:"message"`
}

// Send writes a chunk of PCM as a binary frame.
func (s *turnsStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next turn event. An update carries the running transcript as an
// interim result; the turn's end finalizes it. An eager end is a prediction the
// server may retract with a resume, so it is not treated as the turn ending.
func (s *turnsStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m turnsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case turnsUpdate:
			if m.Transcript == "" {
				continue
			}
			return []stt.Result{{Text: m.Transcript, Final: false}}, nil
		case turnsEnd:
			if m.Transcript == "" {
				continue
			}
			return []stt.Result{{Text: m.Transcript, Final: true, EndOfTurn: true}}, nil
		case turnsError:
			return nil, fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
		case turnsConnected, turnsStart, turnsEagerEnd, turnsResume:
			// Session and turn-boundary bookkeeping with no transcript to emit.
			continue
		}
	}
}

// Close tears the session down. Cartesia ends a turn session by closing the
// socket; there is no finalize command.
func (s *turnsStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
