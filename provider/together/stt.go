package together

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	sttBaseURL      = "wss://api.together.ai/v1"
	defaultSTTModel = "openai/whisper-large-v3"
	// defaultTurnDetection lets Together segment utterances server-side. The
	// streaming STT contract forwards audio but has no per-turn commit signal,
	// so the service relies on the server to detect turn boundaries.
	defaultTurnDetection = "server_vad"
)

// errSTT wraps an error reported by the Together transcription session.
//
//nolint:gochecknoglobals // sentinel error
var errSTT = errors.New("together: stt error")

// STTConfig configures the Together AI streaming STT service, which uses an
// OpenAI-compatible realtime transcription WebSocket.
type STTConfig struct {
	// APIKey is the Together AI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the realtime WebSocket base; empty uses the hosted endpoint.
	BaseURL string
	// Model is the transcription model; empty uses a default.
	Model string
	// Language hints the spoken language; reported on results. The zero value
	// leaves it unknown.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// TurnDetection selects server-side endpointing; empty uses "server_vad".
	TurnDetection string

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.TogetherTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Together AI streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = sttBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.TurnDetection == "" {
		cfg.TurnDetection = defaultTurnDetection
	}
	return stt.NewStream("TogetherSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg STTConfig
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.TogetherTTFSP99)}
}

// Connect dials the realtime transcription WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	url := fmt.Sprintf(
		"%s/realtime?intent=transcription&model=%s&turn_detection=%s&input_audio_format=pcm_s16le_%d",
		c.cfg.BaseURL, c.cfg.Model, c.cfg.TurnDetection, sampleRate,
	)
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("OpenAI-Beta", "realtime=v1")

	conn, err := wsutil.Dial(ctx, url, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx, lang: c.cfg.Language.Code()}, nil
}

type stream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
	lang    string
}

// sttEvent is the subset of a Together transcription event we read.
type sttEvent struct {
	Type       string `json:"type"`
	Transcript string `json:"transcript"`
	Error      struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Send appends a chunk of PCM to the input buffer as base64.
func (s *stream) Send(audio []byte) error {
	msg, err := json.Marshal(map[string]string{
		msgType: "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(audio),
	})
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, msg)
}

// Recv reads the next result. A transcription delta surfaces as an interim; a
// completed transcription is final and marks the end of the user's turn.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var evt sttEvent
		if json.Unmarshal(data, &evt) != nil {
			continue
		}
		switch evt.Type {
		case "conversation.item.input_audio_transcription.delta":
			if text := strings.TrimSpace(evt.Transcript); text != "" {
				return []stt.Result{{Text: text, Final: false, Language: s.lang}}, nil
			}
		case "conversation.item.input_audio_transcription.completed":
			if text := strings.TrimSpace(evt.Transcript); text != "" {
				return []stt.Result{{Text: text, Final: true, EndOfTurn: true, Language: s.lang}}, nil
			}
		case "conversation.item.input_audio_transcription.failed", "error":
			return nil, fmt.Errorf("%w: %s", errSTT, evt.Error.Message)
		}
	}
}

// Close closes the session socket.
func (s *stream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
