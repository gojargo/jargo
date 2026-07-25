package speechmatics

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// NewSTT builds a Speechmatics streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Language == "" {
		cfg.Language = language.EnglishUS
	}
	if cfg.OperatingPoint == "" {
		cfg.OperatingPoint = defaultOperatingPoint
	}
	if cfg.MaxDelay == 0 {
		cfg.MaxDelay = defaultMaxDelay
	}
	return stt.NewStream("SpeechmaticsSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// Connect dials the real-time WebSocket, sends StartRecognition, and waits for
// the RecognitionStarted acknowledgement before audio flows.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, c.cfg.URL, &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)

	s := &stream{conn: conn, ctx: ctx}
	if err := conn.Write(ctx, websocket.MessageText, c.startRecognition(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
		return nil, err
	}
	if err := s.awaitStarted(); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
		return nil, err
	}
	return s, nil
}

func (c *connector) startRecognition(sampleRate int) []byte {
	tc := map[string]any{
		"language":        c.cfg.Language.BaseCode(),
		"operating_point": c.cfg.OperatingPoint,
		"enable_partials": c.cfg.EnablePartials == nil || *c.cfg.EnablePartials,
		"max_delay":       c.cfg.MaxDelay,
	}
	eou := defaultEndOfUtteranceSilence
	if c.cfg.EndOfUtteranceSilence != nil {
		eou = *c.cfg.EndOfUtteranceSilence
	}
	if eou > 0 {
		tc["conversation_config"] = map[string]any{"end_of_utterance_silence_trigger": eou}
	}
	msg := map[string]any{
		"message": "StartRecognition",
		"audio_format": map[string]any{
			"type":        "raw",
			"encoding":    "pcm_s16le",
			"sample_rate": sampleRate,
		},
		"transcription_config": tc,
	}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	return b
}

type stream struct {
	conn     *websocket.Conn
	ctx      context.Context
	writeMu  sync.Mutex
	seqNo    atomic.Uint64
	finalBuf string
}

// awaitStarted reads until RecognitionStarted, failing on an error message.
func (s *stream) awaitStarted() error {
	for {
		var m message
		if err := s.read(&m); err != nil {
			return err
		}
		switch m.Message {
		case "RecognitionStarted":
			return nil
		case "Error":
			return fmt.Errorf("%w: %s", errServer, m.Reason)
		}
	}
}

// Send writes a chunk of PCM as a binary frame (Speechmatics' AddAudio).
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(s.ctx, websocket.MessageBinary, audio); err != nil {
		return err
	}
	s.seqNo.Add(1)
	return nil
}

// Recv reads the next transcript. Partials are interim; finalized segments
// accumulate and are emitted together at end-of-utterance with EndOfTurn set.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		var m message
		if err := s.read(&m); err != nil {
			return nil, err
		}
		switch m.Message {
		case "AddPartialTranscript":
			if t := m.Metadata.Transcript; t != "" {
				return []stt.Result{{Text: t, Final: false}}, nil
			}
		case "AddTranscript":
			if t := m.Metadata.Transcript; t != "" {
				if s.finalBuf != "" {
					s.finalBuf += " "
				}
				s.finalBuf += t
			}
		case "EndOfUtterance":
			if s.finalBuf != "" {
				text := s.finalBuf
				s.finalBuf = ""
				return []stt.Result{{Text: text, Final: true, EndOfTurn: true}}, nil
			}
		case "EndOfTranscript":
			return nil, io.EOF
		case "Error":
			return nil, fmt.Errorf("%w: %s", errServer, m.Reason)
		}
	}
}

func (s *stream) read(m *message) error {
	_, data, err := s.conn.Read(s.ctx)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, m)
}

// Close sends EndOfStream and closes the socket.
func (s *stream) Close() error {
	end, _ := json.Marshal(map[string]any{ //nolint:errchkjson // known-serializable values
		"message":     "EndOfStream",
		"last_seq_no": s.seqNo.Load(),
	})
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, end)
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// message is the subset of Speechmatics' real-time messages we use.
type message struct {
	Message  string `json:"message"`
	Reason   string `json:"reason"`
	Metadata struct {
		Transcript string `json:"transcript"`
	} `json:"metadata"`
}
