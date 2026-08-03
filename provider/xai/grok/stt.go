package grok

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultSTTURL is the hosted real-time transcription WebSocket.
	defaultSTTURL = "wss://api.x.ai/v1/stt"
	// defaultSTTEncoding is signed 16-bit little-endian PCM, the format the
	// pipeline carries.
	defaultSTTEncoding = "pcm"
	// sttUserAgent identifies the client to xAI on the handshake.
	sttUserAgent = "xAI/1.0 (integration=jargo)"
	// sttTTFSP99 is the time-to-final-segment P99 latency reported downstream.
	sttTTFSP99 = 2140 * time.Millisecond
)

// xAI STT server event types.
const (
	// sttEventCreated acknowledges the session; audio sent before it is dropped.
	sttEventCreated = "transcript.created"
	// sttEventPartial carries a transcript, interim or endpointed.
	sttEventPartial = "transcript.partial"
	// sttEventDone carries the final transcript for a closing session.
	sttEventDone = "transcript.done"
	// sttEventError reports a server-side failure.
	sttEventError = "error"
)

// errSTTServer wraps an error reported by the xAI STT service.
//
//nolint:gochecknoglobals // sentinel error
var errSTTServer = errors.New("xai: stt server error")

// errSTTChannels is returned when multichannel transcription is asked for
// without saying how many channels the audio interleaves.
//
//nolint:gochecknoglobals // sentinel error
var errSTTChannels = errors.New("xai: stt multichannel requires channels")

// STTConfig configures the xAI streaming STT service.
type STTConfig struct {
	// APIKey is the xAI API key, sent as a Bearer token on the WebSocket
	// handshake. Required.
	APIKey string `validate:"required"`
	// URL overrides the transcription WebSocket endpoint; empty uses the hosted
	// endpoint.
	URL string
	// Encoding is the audio encoding ("pcm", "mulaw" or "alaw"); empty uses
	// "pcm", signed 16-bit little-endian.
	Encoding string `validate:"omitempty,oneof=pcm mulaw alaw"`
	// Language hints the spoken language and enables inverse text
	// normalization; the zero value lets the server decide. xAI takes the base
	// two-letter code, so a regional language is sent as its base.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// xAI accepts 8000, 16000, 22050, 24000, 44100 and 48000.
	SampleRate int `validate:"omitempty,oneof=8000 16000 22050 24000 44100 48000"`
	// InterimResults emits partial transcripts roughly every 500 ms; nil
	// defaults to true.
	InterimResults *bool
	// Endpointing is the silence duration in milliseconds that ends a turn; nil
	// uses the server default of 10 ms.
	Endpointing *int `validate:"omitempty,min=0,max=5000"`
	// Multichannel transcribes each interleaved channel independently. It
	// requires Channels to be set.
	Multichannel *bool
	// Channels is the number of interleaved channels; nil sends none. Required
	// when Multichannel is true.
	Channels *int `validate:"omitempty,min=2,max=8"`
	// Diarize attaches a detected speaker to each word; nil sends none.
	Diarize *bool
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.Multichannel != nil && *c.Multichannel && c.Channels == nil {
		return fmt.Errorf("%w", errSTTChannels)
	}
	return nil
}

// NewSTT builds an xAI streaming speech-to-text service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultSTTURL
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultSTTEncoding
	}
	return stt.NewStream("XAISTT", &sttConnector{cfg: cfg}, cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata reports xAI's time-to-final-segment latency to downstream processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: sttTTFSP99}
}

// Connect opens a transcription session. The session is configured entirely
// through query parameters on the handshake, so there is no setup message.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("User-Agent", sttUserAgent)

	conn, err := wsutil.Dial(ctx, c.endpoint(sampleRate), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &sttStream{conn: conn, ctx: ctx, lang: c.cfg.Language.BaseCode()}, nil
}

// endpoint builds the session URL. xAI takes the base two-letter language code,
// and booleans as lowercase strings.
func (c *sttConnector) endpoint(sampleRate int) string {
	q := url.Values{}
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("encoding", c.cfg.Encoding)
	query.SetStrOpt(q, "language", c.cfg.Language.BaseCode())
	query.SetBoolTrue(q, "interim_results", c.cfg.InterimResults)
	query.SetIntOpt(q, "endpointing", c.cfg.Endpointing)
	query.SetBoolOpt(q, "multichannel", c.cfg.Multichannel)
	query.SetIntOpt(q, "channels", c.cfg.Channels)
	query.SetBoolOpt(q, "diarize", c.cfg.Diarize)
	return c.cfg.URL + "?" + q.Encode()
}

type sttStream struct {
	conn *wsutil.Conn
	ctx  context.Context
	// lang is the configured language hint echoed on results, or "".
	lang string

	writeMu sync.Mutex
	// ready gates audio on the session acknowledgement, which arrives on the
	// read side. Audio sent before it would be rejected, so it is dropped.
	ready atomic.Bool
}

// sttMessage is the subset of an xAI STT event we read.
type sttMessage struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// IsFinal marks a transcript the server will not revise.
	IsFinal bool `json:"is_final"`
	// SpeechFinal marks the end of the user's turn, detected by endpointing.
	SpeechFinal bool   `json:"speech_final"`
	Message     string `json:"message"`
}

// Send writes a chunk of PCM as a binary frame, once the session is ready.
func (s *sttStream) Send(audio []byte) error {
	if !s.ready.Load() {
		return nil
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcript. A partial with is_final set is a finalized
// transcription, and speech_final on top of it ends the turn.
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
		case sttEventCreated:
			s.ready.Store(true)
		case sttEventPartial:
			if m.Text == "" {
				continue
			}
			return []stt.Result{{
				Text:      m.Text,
				Final:     m.IsFinal,
				EndOfTurn: m.IsFinal && m.SpeechFinal,
				Language:  s.lang,
			}}, nil
		case sttEventDone:
			if m.Text == "" {
				continue
			}
			return []stt.Result{{Text: m.Text, Final: true, EndOfTurn: true, Language: s.lang}}, nil
		case sttEventError:
			return nil, fmt.Errorf("%w: %s", errSTTServer, m.Message)
		}
	}
}

// Close tells xAI the audio is complete, then tears the session down.
func (s *sttStream) Close() error {
	s.writeMu.Lock()
	// Sent on the background context: s.ctx is already canceled by the time the
	// service closes the session, and this is the signal that flushes the
	// server's final transcript.
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"audio.done"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
