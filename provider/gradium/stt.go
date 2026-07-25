package gradium

import (
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

// errSTTProtocol is returned when Gradium reports an STT error or an unexpected
// handshake reply.
//
//nolint:gochecknoglobals // sentinel error
var errSTTProtocol = errors.New("gradium: stt protocol error")

const (
	defaultSTTURL   = "wss://api.gradium.ai/api/speech/asr"
	defaultSTTModel = "default"
	// defaultDelayInFrames delays text generation by 12 frames (960 ms), trading
	// latency for context.
	defaultDelayInFrames = 12
	// sttChunkMS is the audio chunk size streamed to the server.
	sttChunkMS = 80
	// sttTTFSP99 is the time-to-final-segment P99 latency reported to downstream.
	sttTTFSP99 = 620 * time.Millisecond
)

// STTConfig configures the Gradium STT service.
type STTConfig struct {
	// APIKey is the Gradium API key, sent as the x-api-key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the ASR WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a default.
	Model string
	// Encoding is the base audio encoding ("pcm", "wav", or "opus"); empty uses
	// "pcm". For PCM the sample rate is appended to form the input format (for
	// example "pcm_16000").
	Encoding string
	// Language hints the spoken language; the zero value lets the server decide.
	// Verified codes are de, en, es, fr, and pt; other languages fall back to
	// the base code.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// DelayInFrames delays text generation by this many 80 ms frames; nil uses a
	// default. Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
	DelayInFrames *int `validate:"omitempty,oneof=7 8 10 12 14 16 20 24 36 48"`
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Gradium streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultSTTURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Encoding == "" {
		cfg.Encoding = encPCM
	}
	return stt.NewStream("GradiumSTT", &sttConnector{cfg: cfg}, cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata reports Gradium's time-to-final-segment latency to downstream
// processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: sttTTFSP99}
}

// inputFormat builds Gradium's input_format from the encoding and sample rate.
// For PCM the sample rate is appended; other encodings are used as-is.
func (c *sttConnector) inputFormat(sampleRate int) string {
	if c.cfg.Encoding != encPCM {
		return c.cfg.Encoding
	}
	switch sampleRate {
	case 8000:
		return "pcm_8000"
	case 24000:
		return "pcm_24000"
	default:
		return "pcm_16000"
	}
}

// gradiumLanguage maps a Language to Gradium's language code. Gradium wants the
// base code; the empty language yields "" so the caller omits the hint.
func gradiumLanguage(l language.Language) string { return l.BaseCode() }

// Connect dials the WebSocket, sends the setup handshake, and waits for the
// server's ready acknowledgement.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("x-api-key", c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}

	if err := conn.Write(ctx, websocket.MessageText, c.setup(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return nil, err
	}
	if err := awaitReady(ctx, conn); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return nil, err
	}

	chunkBytes := sttChunkMS * sampleRate * 2 / 1000
	return &sttStream{conn: conn, ctx: ctx, chunkBytes: chunkBytes, lang: c.cfg.Language.Code()}, nil
}

// setup builds the initial setup message carrying the model, input format, and
// optional language and delay configuration.
func (c *sttConnector) setup(sampleRate int) []byte {
	msg := map[string]any{
		msgType:        "setup",
		"model_name":   c.cfg.Model,
		"input_format": c.inputFormat(sampleRate),
	}
	jsonConfig := map[string]any{}
	if lang := gradiumLanguage(c.cfg.Language); lang != "" {
		jsonConfig["language"] = lang
	}
	delay := defaultDelayInFrames
	if c.cfg.DelayInFrames != nil {
		delay = *c.cfg.DelayInFrames
	}
	if delay != 0 {
		jsonConfig["delay_in_frames"] = delay
	}
	if len(jsonConfig) > 0 {
		msg["json_config"] = jsonConfig
	}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	return b
}

// awaitReady reads the handshake reply and requires a "ready" message.
func awaitReady(ctx context.Context, conn *websocket.Conn) error {
	_, data, err := conn.Read(ctx)
	if err != nil {
		return err
	}
	var m sttMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	switch m.Type {
	case "ready":
		return nil
	case "error":
		return fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
	default:
		return fmt.Errorf("%w: unexpected first message %q", errSTTProtocol, m.Type)
	}
}

type sttStream struct {
	conn       *websocket.Conn
	ctx        context.Context
	chunkBytes int
	// lang is the configured language hint echoed on results, or "".
	lang string

	writeMu sync.Mutex
	buf     []byte

	accumulated []string
}

// sttMessage is the subset of a Gradium ASR message we use.
type sttMessage struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Message string `json:"message"`
}

// Send buffers audio and streams it in fixed 80 ms chunks, base64-encoded.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.buf = append(s.buf, audio...)
	for len(s.buf) >= s.chunkBytes {
		chunk := s.buf[:s.chunkBytes]
		s.buf = s.buf[s.chunkBytes:]
		msg := map[string]any{
			msgType:  msgAudio,
			msgAudio: base64.StdEncoding.EncodeToString(chunk),
		}
		b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
		if err := s.conn.Write(s.ctx, websocket.MessageText, b); err != nil {
			return err
		}
	}
	return nil
}

// Recv reads the next message. Each "text" fragment accumulates and surfaces the
// running transcript as an interim result; a "flushed" or "end_of_stream" marker
// finalizes the accumulated transcript with EndOfTurn set.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case msgText:
			s.accumulated = append(s.accumulated, m.Text)
			text := strings.Join(s.accumulated, " ")
			return []stt.Result{{Text: text, Final: false, Language: s.lang}}, nil
		case "flushed", msgEndStream:
			if len(s.accumulated) == 0 {
				continue
			}
			text := strings.Join(s.accumulated, " ")
			s.accumulated = nil
			return []stt.Result{{Text: text, Final: true, EndOfTurn: true, Language: s.lang}}, nil
		case msgError:
			return nil, fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
		}
	}
}

// Close tears the session down.
func (s *sttStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
