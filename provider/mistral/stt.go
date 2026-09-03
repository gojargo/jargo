// Mistral realtime speech-to-text over the Voxtral transcription WebSocket.
// Audio streams up as base64 append messages; the server streams incremental
// text deltas (surfaced as interims) and a final transcription for each segment.
//
// A segment closes when the client flushes the audio it has sent, which happens
// as the VAD reports the speech ended. The session stays open across the flush,
// so one connection carries every utterance of the call.

package mistral

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	sttDefaultURL   = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
	sttDefaultModel = "voxtral-mini-transcribe-realtime-2602"
	// sttReadLimit bounds a single inbound message.
	sttReadLimit = 1 << 20

	// msgType is the discriminator key shared by client and server frames.
	msgType = "type"

	// Client message types.
	sttMsgSessionUpdate = "session.update"
	sttMsgAudioAppend   = "input_audio.append"
	sttMsgAudioFlush    = "input_audio.flush"

	// Server event types.
	sttEventTextDelta = "transcription.text.delta"
	sttEventDone      = "transcription.done"
	sttEventLanguage  = "transcription.language"
	sttEventError     = "error"
)

// errSTTServer wraps an error reported by the Mistral realtime STT API.
//
//nolint:gochecknoglobals // sentinel error
var errSTTServer = errors.New("mistral: stt server error")

// STTConfig configures the Mistral realtime STT service.
type STTConfig struct {
	// APIKey is the Mistral API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// URL overrides the realtime endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a current default.
	Model string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// TargetStreamingDelayMs trades latency for accuracy: higher values may
	// improve accuracy at the cost of latency. 0 uses the server default.
	TargetStreamingDelayMs int

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.MistralTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Mistral realtime streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = sttDefaultURL
	}
	if cfg.Model == "" {
		cfg.Model = sttDefaultModel
	}
	return stt.NewStream("MistralSTT", &sttConnector{cfg: cfg}, cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata reports the model in use and the transcript latency the turn
// strategies size their wait by.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.MistralTTFSP99), Model: c.cfg.Model}
}

// Connect dials the realtime WebSocket and sends the session configuration. The
// server's session.created acknowledgement is consumed by the read loop.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	endpoint := c.cfg.URL + "?model=" + url.QueryEscape(c.cfg.Model)
	conn, err := wsutil.Dial(ctx, endpoint, header, sttReadLimit)
	if err != nil {
		return nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, c.sessionUpdate(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "session update failed")
		return nil, err
	}
	return &sttStream{conn: conn, ctx: ctx}, nil
}

func (c *sttConnector) sessionUpdate(sampleRate int) []byte {
	session := map[string]any{
		"audio_format": map[string]any{
			"encoding":    "pcm_s16le",
			"sample_rate": sampleRate,
		},
	}
	if c.cfg.TargetStreamingDelayMs > 0 {
		session["target_streaming_delay_ms"] = c.cfg.TargetStreamingDelayMs
	}
	b, _ := json.Marshal(map[string]any{ //nolint:errchkjson // map of known-serializable values
		msgType:   sttMsgSessionUpdate,
		"session": session,
	})
	return b
}

type sttStream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
	// textMu guards the transcript being built, which the read loop extends and
	// the start of an utterance clears.
	textMu      sync.Mutex
	accumulated string
	language    string
}

// Send appends a chunk of PCM as a base64 audio message.
func (s *sttStream) Send(audio []byte) error {
	b, _ := json.Marshal(map[string]any{ //nolint:errchkjson // map of known-serializable values
		msgType: sttMsgAudioAppend,
		"audio": base64.StdEncoding.EncodeToString(audio),
	})
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, b)
}

// Finalize flushes the audio sent so far, which closes the segment being
// transcribed: the server answers with the done event carrying the transcript
// for it and keeps the session open for the next utterance. Without it a segment
// is only closed at the end of the stream, and the whole call comes back as one
// transcript long after the turn it belonged to.
func (s *sttStream) Finalize() error {
	b, _ := json.Marshal(map[string]any{msgType: sttMsgAudioFlush}) //nolint:errchkjson // map of known-serializable values
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, b)
}

// SpeechStarted drops the transcript built for the utterance before this one.
// The done event clears it as each segment closes, and this covers the segment
// that never closed: the deltas for it would otherwise be read as the opening
// words of the utterance beginning now.
func (s *sttStream) SpeechStarted() {
	s.textMu.Lock()
	defer s.textMu.Unlock()
	s.accumulated = ""
}

// Recv reads the next transcription event. Text deltas accumulate and surface as
// interim results; a done event closes the segment as a final result. Language
// events update the language reported on the next final.
//
// A final is not marked as the end of the turn. The done event closes the
// segment the flush asked about, but it says nothing about whether the user has
// finished speaking, and the turn strategies read that flag as exactly that.
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
		case sttEventTextDelta:
			s.textMu.Lock()
			s.accumulated += m.Text
			text := s.accumulated
			s.textMu.Unlock()
			if text != "" {
				return []stt.Result{{Text: text, Final: false}}, nil
			}
		case sttEventDone:
			text := m.Text
			s.textMu.Lock()
			s.accumulated = ""
			s.textMu.Unlock()
			if text != "" {
				return []stt.Result{{Text: text, Final: true, Language: s.language}}, nil
			}
		case sttEventLanguage:
			s.language = m.AudioLanguage
		case sttEventError:
			return nil, fmt.Errorf("%w: %s", errSTTServer, m.Error.Message)
		}
	}
}

// Close ends the session. Nothing is asked of the server on the way out: each
// utterance was flushed as its speech ended, and a request made here would be
// answered after the reader that would have carried the transcript has gone.
func (s *sttStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// sttMessage is the subset of Mistral realtime STT events we use.
type sttMessage struct {
	Type          string `json:"type"`
	Text          string `json:"text"`
	AudioLanguage string `json:"audio_language"`
	Error         struct {
		Message string `json:"message"`
	} `json:"error"`
}
