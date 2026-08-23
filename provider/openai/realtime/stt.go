package realtime

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultSTTModel is a current low-latency streaming transcription model.
	defaultSTTModel = "gpt-realtime-whisper"
	// sttSampleRate is the rate the Realtime API exchanges audio at. The
	// transcription session is configured for it, so run the transport's input
	// at 24 kHz.
	sttSampleRate = 24000
)

// Transcription session message types.
const (
	// sttEventDelta is a partial transcript for the utterance in progress.
	sttEventDelta = "conversation.item.input_audio_transcription.delta"
	// sttEventCompleted is the finalized transcript for an utterance.
	sttEventCompleted = "conversation.item.input_audio_transcription.completed"
	// sttEventFailed reports that an utterance could not be transcribed.
	sttEventFailed = "conversation.item.input_audio_transcription.failed"
	// sttEventError reports a session-level failure.
	sttEventError = "error"
)

// STTConfig configures the OpenAI Realtime transcription service. The session is
// transcription-only: the model produces no conversational reply.
type STTConfig struct {
	// APIKey is the OpenAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Realtime WebSocket endpoint.
	BaseURL string
	// Model is the transcription model; empty uses a current default.
	Model string
	// Language of the audio, as a base code; the zero value lets the model
	// decide.
	Language language.Language
	// Prompt guides the transcription style or supplies keyword hints; empty
	// sends none. The default streaming model does not support it.
	Prompt string
	// NoiseReduction filters the input audio: "near_field" for a close
	// microphone, "far_field" for a distant one. Empty disables it.
	NoiseReduction string `validate:"omitempty,oneof=near_field far_field"`
	// SilenceMS is how long the server waits through silence before ending an
	// utterance; 0 uses the server default.
	SilenceMS int `validate:"omitempty,min=0"`

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.OpenAIRealtimeTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds an OpenAI Realtime transcription service. The server detects
// utterance boundaries itself, so the pipeline does not need its own end-of-turn
// detection and the service recommends external turn strategies.
//
// The Realtime API exchanges 24 kHz audio, which is what the session is
// configured for, so run the transport's input sample rate at 24000.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	return stt.NewStream("OpenAIRealtimeSTT", &sttConnector{cfg: cfg}, sttSampleRate)
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata tells downstream processors the session detects turns itself.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{
		UserTurnStrategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
		TTFSP99:            cmp.Or(c.cfg.TTFSP99, stt.OpenAIRealtimeTTFSP99),
		Model:              c.cfg.Model,
	}
}

// Connect opens a transcription session and configures it.
func (c *sttConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("OpenAI-Beta", "realtime=v1")

	// The intent selects a transcription session rather than a conversational
	// one; without it the model would answer instead of only transcribing.
	conn, err := wsutil.Dial(ctx, c.cfg.BaseURL+"?intent=transcription", header, readLimit)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(c.sessionUpdate())
	if err != nil {
		_ = conn.Close(websocket.StatusInternalError, "session encode failed")
		return nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "session update failed")
		return nil, err
	}
	return &sttStream{conn: conn, ctx: ctx, lang: c.cfg.Language.BaseCode()}, nil
}

// sessionUpdate configures the transcription session: the audio format, the
// model, and the server-side turn detection that delimits each utterance.
func (c *sttConnector) sessionUpdate() map[string]any {
	transcription := map[string]any{"model": c.cfg.Model}
	if lang := c.cfg.Language.BaseCode(); lang != "" {
		transcription["language"] = lang
	}
	if c.cfg.Prompt != "" {
		transcription["prompt"] = c.cfg.Prompt
	}

	turnDetection := map[string]any{keyType: "server_vad"}
	if c.cfg.SilenceMS > 0 {
		turnDetection["silence_duration_ms"] = c.cfg.SilenceMS
	}

	input := map[string]any{
		"format":         map[string]any{keyType: "audio/pcm", "rate": sttSampleRate},
		"transcription":  transcription,
		"turn_detection": turnDetection,
	}
	if c.cfg.NoiseReduction != "" {
		input["noise_reduction"] = map[string]any{keyType: c.cfg.NoiseReduction}
	}

	return map[string]any{
		keyType: msgSessionUpdate,
		"session": map[string]any{
			keyType:  "transcription",
			keyAudio: map[string]any{"input": input},
		},
	}
}

type sttStream struct {
	conn *wsutil.Conn
	ctx  context.Context
	// lang is the configured language echoed on results, or "".
	lang    string
	writeMu sync.Mutex
	// partial accumulates the utterance in progress. The API streams each
	// interim as a fragment rather than the running text, and every other jargo
	// STT service reports interims cumulatively, so the fragments are joined
	// here. It is cleared when the utterance finalizes.
	partial string
}

// sttMessage is the subset of a transcription session event we read.
type sttMessage struct {
	Type       string `json:"type"`
	Delta      string `json:"delta"`
	Transcript string `json:"transcript"`
	Error      struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Send appends a chunk of input PCM to the model's audio buffer.
func (s *sttStream) Send(audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	payload, err := json.Marshal(map[string]any{
		keyType:  "input_audio_buffer.append",
		keyAudio: base64.StdEncoding.EncodeToString(audio),
	})
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, payload)
}

// Recv reads the next transcript. A delta is the utterance so far; the completed
// event finalizes it and ends the turn, since the server's own VAD delimited it.
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
		case sttEventDelta:
			if m.Delta == "" {
				continue
			}
			s.partial += m.Delta
			return []stt.Result{{Text: s.partial, Final: false, Language: s.lang}}, nil
		case sttEventCompleted:
			s.partial = ""
			if m.Transcript == "" {
				continue
			}
			return []stt.Result{{Text: m.Transcript, Final: true, EndOfTurn: true, Language: s.lang}}, nil
		case sttEventFailed, sttEventError:
			s.partial = ""
			return nil, fmt.Errorf("%w: %s", errServer, m.Error.Message)
		}
	}
}

// Close tears the session down.
func (s *sttStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
