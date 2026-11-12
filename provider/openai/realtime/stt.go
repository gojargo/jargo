package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultSTTModel is a current low-latency streaming transcription model.
	defaultSTTModel = "gpt-realtime-whisper"
	// sttSampleRate is the rate the Realtime API exchanges audio at. The
	// transcription session is configured for it, and the pipeline's audio is
	// resampled to it before it is sent.
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
	// sttEventSpeechStarted reports that the server's own detection heard the
	// user begin speaking.
	sttEventSpeechStarted = "input_audio_buffer.speech_started"
	// sttEventSpeechStopped reports that it heard the user stop.
	sttEventSpeechStopped = "input_audio_buffer.speech_stopped"
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
	// sends none. The default streaming model does not support it, so with that
	// model it is dropped with a warning.
	Prompt string
	// NoiseReduction filters the input audio: "near_field" for a close
	// microphone, "far_field" for a distant one. Empty disables it.
	NoiseReduction string `validate:"omitempty,oneof=near_field far_field"`
	// Delay is how long the model waits before emitting transcription text,
	// trading latency for accuracy: "minimal", "low", "medium", "high" or
	// "xhigh". Empty leaves the field off. Only gpt-realtime-whisper, the default
	// model, honors it.
	Delay string `validate:"omitempty,oneof=minimal low medium high xhigh"`
	// TurnDetection is the server-side VAD configuration. Nil, the default,
	// turns it off and relies on a local VAD in the pipeline: when the VAD
	// reports the user stopped, the service commits the audio buffer so the
	// server transcribes the finished segment. A configuration turns it on, at
	// minimum &events.TurnDetection{Type: events.TurnDetectionServerVAD}; the
	// server then delimits each utterance, the service proposes those turn
	// boundaries, and it recommends the external user turn strategies that
	// resolve them, so no separate VAD should run in the pipeline. An empty
	// &events.TurnDetection{} leaves the field off the session, keeping the
	// session's own default, which detects turns. Server-side detection needs a
	// model that supports it: gpt-transcribe does, the default
	// gpt-realtime-whisper does not.
	TurnDetection *events.TurnDetection
	// ShouldInterrupt barges in when the server-side VAD hears the user speak;
	// nil enables it. It only applies with TurnDetection on, and is passed along
	// to the user turn strategies this service recommends, which own the
	// interruption; strategies the application configures itself override the
	// recommendation and this setting with it.
	ShouldInterrupt *bool

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.OpenAIRealtimeTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds an OpenAI Realtime transcription service. By default the
// pipeline's local VAD delimits each utterance and the service commits the
// audio for it when the VAD reports the user stopped; with
// STTConfig.TurnDetection set the server detects the turns instead. See
// STTConfig.TurnDetection.
//
// The Realtime API exchanges 24 kHz audio, which is what the session is
// configured for. The service resamples the pipeline's input audio to it, so
// the transport may run at any input rate.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	cfg = omitUnsupportedPrompt(cfg)
	return stt.NewStream("OpenAIRealtimeSTT", &sttConnector{cfg: cfg}, 0)
}

// omitUnsupportedPrompt drops a prompt the configured model does not accept.
func omitUnsupportedPrompt(cfg STTConfig) STTConfig {
	if cfg.Model == defaultSTTModel && cfg.Prompt != "" {
		slog.Warn("openai realtime stt: the model does not support a prompt; omitting it from the session",
			"model", cfg.Model)
		cfg.Prompt = ""
	}
	return cfg
}

// serverVAD reports whether the server detects the turns: any turn detection
// configuration that does not turn it off, including the session's own default.
func (c STTConfig) serverVAD() bool {
	return c.TurnDetection != nil && !c.TurnDetection.Disabled
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata recommends the external turn strategies when the server-side VAD
// drives the turns: the server detects the speech boundaries and the service
// proposes turns from them, so the user aggregator resolves those rather than
// running its own detection. With the server-side VAD off the server reports no
// boundaries, and the pipeline's defaults are left in place.
func (c *sttConnector) Metadata() stt.Metadata {
	md := stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.OpenAIRealtimeTTFSP99),
		Model:   c.cfg.Model,
	}
	if c.cfg.serverVAD() {
		md.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{
			EnableInterruptions: c.cfg.ShouldInterrupt,
		})
	}
	return md
}

// Connect opens a transcription session and configures it. Audio is resampled
// from the pipeline's rate to the 24 kHz the session is configured for.
// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *sttConnector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	rs, err := resample.New(sampleRate, sttSampleRate, 1)
	if err != nil {
		return nil, err
	}

	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("OpenAI-Beta", "realtime=v1")

	// The intent selects a transcription session rather than a conversational
	// one; without it the model would answer instead of only transcribing.
	conn, err := c.cfg.Websocket.Dial(ctx, c.cfg.BaseURL+"?intent=transcription", header, readLimit)
	if err != nil {
		rs.Close()
		return nil, err
	}

	payload, err := json.Marshal(c.sessionUpdate())
	if err != nil {
		rs.Close()
		_ = conn.Close(websocket.StatusInternalError, "session encode failed")
		return nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
		rs.Close()
		_ = conn.Close(websocket.StatusInternalError, "session update failed")
		return nil, err
	}
	return &sttStream{
		conn: conn, ctx: ctx, lang: c.cfg.Language.BaseCode(), rs: rs, serverVAD: c.cfg.serverVAD(),
	}, nil
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
	if c.cfg.Delay != "" {
		transcription["delay"] = c.cfg.Delay
	}

	input := map[string]any{
		"format":        map[string]any{keyType: "audio/pcm", "rate": sttSampleRate},
		"transcription": transcription,
	}
	switch td := c.cfg.TurnDetection; {
	case td == nil || td.Disabled:
		// Off: sent as null, so the local VAD delimits the utterances.
		input["turn_detection"] = nil
	case *td != (events.TurnDetection{}):
		input["turn_detection"] = td
	}
	// An empty configuration leaves the field off, keeping the session default.
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
	lang string
	// serverVAD reports whether the server delimits the utterances. When it
	// does not, the service commits the audio buffer on the local VAD's stop.
	serverVAD bool
	// rs converts the pipeline's audio to the 24 kHz the session takes; it
	// passes the audio through when the pipeline already runs at that rate.
	// writeMu guards it.
	rs      *resample.Resampler
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

// Send resamples a chunk of input PCM to 24 kHz and appends it to the model's
// audio buffer.
func (s *sttStream) Send(audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	audio = s.rs.Process(audio)
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
	return s.conn.Write(s.ctx, websocket.MessageText, payload)
}

// Recv reads the next transcript. A delta is the utterance so far; the completed
// event finalizes it. With the server-side VAD on it also ends the turn, since
// that VAD delimited it; with it off the pipeline's turn strategies decide.
//
// The server-side VAD also reports where the speech begins and ends, and those
// boundaries are carried on the results as proposals for the pipeline's turn
// strategies to resolve. The server sends the stop ahead of the transcript it
// belongs to, so the strategy resolving it waits for the text.
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
			return []stt.Result{{Text: m.Transcript, Final: true, EndOfTurn: s.serverVAD, Language: s.lang}}, nil
		case sttEventSpeechStarted:
			return []stt.Result{{Speech: stt.SpeechStarted}}, nil
		case sttEventSpeechStopped:
			return []stt.Result{{Speech: stt.SpeechStopped}}, nil
		case sttEventFailed, sttEventError:
			s.partial = ""
			return nil, fmt.Errorf("%w: %s", errServer, m.Error.Message)
		}
	}
}

// Finalize commits the audio buffer when the local VAD reports the user
// stopped, so the server transcribes the finished segment. With the server-side
// VAD on, the server commits the buffer itself and this does nothing.
func (s *sttStream) Finalize() error {
	if s.serverVAD {
		return nil
	}
	payload, err := json.Marshal(map[string]any{keyType: "input_audio_buffer.commit"})
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, payload)
}

// Close tears the session down and releases the resampler.
func (s *sttStream) Close() error {
	s.writeMu.Lock()
	s.rs.Close()
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
