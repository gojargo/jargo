package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	return stt.Metadata{TTFSP99: sttTTFSP99, Model: c.cfg.Model}
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
	s := &sttStream{
		conn:       conn,
		ctx:        ctx,
		chunkBytes: chunkBytes,
		lang:       c.cfg.Language.Code(),
		reads:      make(chan sttRead, 8),
		done:       make(chan struct{}),
	}
	go s.readLoop()
	return s, nil
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
func awaitReady(ctx context.Context, conn *wsutil.Conn) error {
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

// transcriptAggregationDelay is how long the transcript stays open after a
// flush. The flush only says the buffered audio has been processed: the words at
// the end of it can still be on their way, so settling the transcript the moment
// it arrives drops the last of the utterance, and those words then turn up at
// the front of the next one.
const transcriptAggregationDelay = 100 * time.Millisecond

type sttStream struct {
	conn       *wsutil.Conn
	ctx        context.Context
	chunkBytes int
	// lang is the configured language hint echoed on results, or "".
	lang string

	writeMu sync.Mutex
	buf     []byte

	accumulated []string
	// reads carries what the reader goroutine has read. Waiting for trailing
	// text on a timer needs a read that can be given up on, and giving up on a
	// read by canceling its context closes the connection under it, so the read
	// runs on its own and the wait happens here instead.
	reads chan sttRead
	// pending holds a message taken off reads while waiting for trailing text
	// that was not text itself, so it is acted on rather than dropped.
	pending *sttRead
	// done stops the reader when the stream is closed.
	done      chan struct{}
	closeOnce sync.Once
}

// sttRead is one message from the reader goroutine, or the failure that ended
// it.
type sttRead struct {
	msg sttMessage
	err error
}

// readLoop reads messages until the connection fails or the stream is closed. It
// reads under the stream's own context, which outlives any single wait.
func (s *sttStream) readLoop() {
	defer close(s.reads)
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			select {
			case s.reads <- sttRead{err: err}:
			case <-s.done:
			}
			return
		}
		var m sttMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		select {
		case s.reads <- sttRead{msg: m}:
		case <-s.done:
			return
		}
	}
}

// next returns the next message, preferring one already taken off the channel
// while waiting for trailing text.
func (s *sttStream) next() (sttRead, bool) {
	if p := s.pending; p != nil {
		s.pending = nil
		return *p, true
	}
	r, ok := <-s.reads
	return r, ok
}

// aggregateTrailing keeps the transcript open for a moment after a flush,
// folding in the words still arriving. Anything else read in that window is held
// for the next call rather than dropped.
func (s *sttStream) aggregateTrailing() {
	timer := time.NewTimer(transcriptAggregationDelay)
	defer timer.Stop()
	for {
		select {
		case r, ok := <-s.reads:
			if !ok {
				return
			}
			if r.err != nil || r.msg.Type != msgText {
				s.pending = &r
				return
			}
			s.accumulated = append(s.accumulated, r.msg.Text)
		case <-timer.C:
			return
		}
	}
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
		r, ok := s.next()
		if !ok {
			return nil, io.EOF
		}
		if r.err != nil {
			return nil, r.err
		}
		switch r.msg.Type {
		case msgText:
			s.accumulated = append(s.accumulated, r.msg.Text)
			text := strings.Join(s.accumulated, " ")
			return []stt.Result{{Text: text, Final: false, Language: s.lang}}, nil
		case msgFlushed, msgEndStream:
			if len(s.accumulated) == 0 {
				continue
			}
			// The words that end the utterance can still be arriving.
			s.aggregateTrailing()
			text := strings.Join(s.accumulated, " ")
			s.accumulated = nil
			return []stt.Result{{Text: text, Final: true, EndOfTurn: true, Language: s.lang}}, nil
		case msgError:
			return nil, fmt.Errorf("%w: %s", errSTTProtocol, r.msg.Message)
		}
	}
}

// Close tears the session down and stops the reader.
func (s *sttStream) Close() error {
	s.closeOnce.Do(func() { close(s.done) })
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
