// Mistral realtime speech-to-text over the Voxtral transcription WebSocket.
// Audio streams up as base64 append messages; the server streams incremental
// text deltas (surfaced as interims) and a final transcription for each segment.

package mistral

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"

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
	sttMsgAudioEnd      = "input_audio.end"

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
	conn        *wsutil.Conn
	ctx         context.Context
	writeMu     sync.Mutex
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

// Recv reads the next transcription event. Text deltas accumulate and surface as
// interim results; a done event flushes the segment as a final result with
// EndOfTurn set. Language events update the language reported on the next final.
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
			s.accumulated += m.Text
			if s.accumulated != "" {
				return []stt.Result{{Text: s.accumulated, Final: false}}, nil
			}
		case sttEventDone:
			text := m.Text
			s.accumulated = ""
			if text != "" {
				return []stt.Result{{Text: text, Final: true, EndOfTurn: true, Language: s.language}}, nil
			}
		case sttEventLanguage:
			s.language = m.AudioLanguage
		case sttEventError:
			return nil, fmt.Errorf("%w: %s", errSTTServer, m.Error.Message)
		}
	}
}

// Close signals end of audio and closes the socket.
func (s *sttStream) Close() error {
	end, _ := json.Marshal(map[string]any{msgType: sttMsgAudioEnd}) //nolint:errchkjson // known-serializable values
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, end)
	s.writeMu.Unlock()
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
