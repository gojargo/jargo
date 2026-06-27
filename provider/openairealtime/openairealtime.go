// Package openairealtime is a speech-to-speech service built on OpenAI's
// Realtime API. Unlike the cascaded STT -> LLM -> TTS pipeline, a single
// bidirectional WebSocket carries the conversation: input audio streams up, and
// the model streams its spoken reply, its transcript, and server-side
// voice-activity events back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The Realtime API exchanges 16-bit mono PCM at 24 kHz, so run
// the pipeline at that rate (set the transport's input and output sample rates
// to 24000); audio at other rates is sent through unchanged and will sound
// wrong.
//
// The model's server VAD drives turn-taking: on detected user speech the service
// emits an InterruptionFrame (barge-in) so the output transport drops buffered
// bot audio. Tool calling is not yet wired up.
package openairealtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
)

const (
	defaultBaseURL            = "wss://api.openai.com/v1/realtime"
	defaultModel              = "gpt-realtime"
	defaultVoice              = "alloy"
	defaultTranscriptionModel = "whisper-1"
	// sampleRate is the fixed rate of the Realtime API's pcm16 audio format.
	sampleRate = 24000
	// readLimit bounds a single inbound WebSocket message; audio deltas are far
	// larger than the library's 32 KiB default.
	readLimit = 1 << 24
)

// errNotConnected is returned when audio is sent before the socket is open.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("openairealtime: not connected")

// errServer wraps an error event reported by the Realtime API.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("openairealtime: server error")

// Config configures the OpenAI Realtime service.
type Config struct {
	// APIKey is the OpenAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Realtime WebSocket endpoint.
	BaseURL string
	// Model is the realtime model id; empty uses a current default.
	Model string
	// Voice is the model voice (e.g. alloy, echo, shimmer); empty uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// TranscriptionModel transcribes the user's audio; empty uses whisper-1. Set
	// to "-" to disable input transcription.
	TranscriptionModel string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Service is the Realtime speech-to-speech processor.
type Service struct {
	*processor.Base
	cfg Config

	mu      sync.Mutex
	conn    *websocket.Conn
	connCtx context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex
	wg      sync.WaitGroup
}

// New builds a Realtime service.
func New(cfg Config) *Service {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.TranscriptionModel == "" {
		cfg.TranscriptionModel = defaultTranscriptionModel
	}
	s := &Service{cfg: cfg}
	s.Base = processor.New("OpenAIRealtime", s)
	return s
}

// ProcessFrame opens the session on StartFrame, forwards input audio up to the
// model, and tears the session down when the pipeline ends.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "openai realtime connect failed", err, true)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			s.sendAudio(fr.Audio)
			return nil // The model consumes the audio; it does not flow on.
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the session and stops the read loop.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

// connect dials the Realtime WebSocket, configures the session, and starts the
// read loop.
func (s *Service) connect(ctx context.Context) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	header.Set("OpenAI-Beta", "realtime=v1")

	conn, resp, err := websocket.Dial(ctx, s.cfg.BaseURL+"?model="+s.cfg.Model, &websocket.DialOptions{
		HTTPHeader: header,
	})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return err
	}
	conn.SetReadLimit(readLimit)

	connCtx, cancel := context.WithCancel(context.Background())
	s.mu.Lock()
	s.conn = conn
	s.connCtx = connCtx
	s.cancel = cancel
	s.mu.Unlock()

	if err := s.send(s.sessionUpdate()); err != nil {
		cancel()
		_ = conn.Close(websocket.StatusInternalError, "session update failed")
		return err
	}

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// sessionUpdateMsg configures the session at the start of the connection.
type sessionUpdateMsg struct {
	Type    string         `json:"type"`
	Session map[string]any `json:"session"`
}

// audioAppendMsg appends a chunk of input PCM to the model's input buffer.
type audioAppendMsg struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// sessionUpdate is the initial session configuration message.
func (s *Service) sessionUpdate() sessionUpdateMsg {
	session := map[string]any{
		"modalities":          []string{"audio", "text"},
		"voice":               s.cfg.Voice,
		"input_audio_format":  "pcm16",
		"output_audio_format": "pcm16",
		"turn_detection":      map[string]any{"type": "server_vad"},
	}
	if s.cfg.Instructions != "" {
		session["instructions"] = s.cfg.Instructions
	}
	if s.cfg.TranscriptionModel != "-" {
		session["input_audio_transcription"] = map[string]any{"model": s.cfg.TranscriptionModel}
	}
	return sessionUpdateMsg{Type: "session.update", Session: session}
}

// sendAudio appends a chunk of input PCM to the model's input buffer.
func (s *Service) sendAudio(pcm []byte) {
	if len(pcm) == 0 {
		return
	}
	_ = s.send(audioAppendMsg{
		Type:  "input_audio_buffer.append",
		Audio: base64.StdEncoding.EncodeToString(pcm),
	})
}

// send marshals v and writes it as a text frame, serializing concurrent writes.
func (s *Service) send(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.mu.Lock()
	conn, connCtx := s.conn, s.connCtx
	s.mu.Unlock()
	if conn == nil {
		return errNotConnected
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(connCtx, websocket.MessageText, data)
}

// disconnect cancels the session context, closes the socket, and waits for the
// read loop to exit. It is safe to call more than once.
func (s *Service) disconnect() {
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
	s.wg.Wait()
}

// serverEvent is the subset of Realtime server events the service handles. The
// delta field carries base64 PCM for audio events and plain text for transcript
// events.
type serverEvent struct {
	Type       string `json:"type"`
	Delta      string `json:"delta"`
	Transcript string `json:"transcript"`
	Error      struct {
		Message string `json:"message"`
	} `json:"error"`
}

// readLoop reads server events until the connection is closed or canceled.
func (s *Service) readLoop(conn *websocket.Conn, connCtx context.Context) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(connCtx)
		if err != nil {
			if connCtx.Err() == nil {
				slog.Debug("openai realtime read ended", "err", err)
			}
			return
		}
		var ev serverEvent
		if json.Unmarshal(data, &ev) != nil {
			continue
		}
		s.handleEvent(connCtx, ev)
	}
}

// handleEvent maps a server event onto downstream pipeline frames.
func (s *Service) handleEvent(ctx context.Context, ev serverEvent) {
	switch ev.Type {
	case "input_audio_buffer.speech_started":
		// Server VAD detected user speech: barge in so buffered bot audio drops.
		_ = s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
		_ = s.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
	case "input_audio_buffer.speech_stopped":
		_ = s.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
	case "response.created":
		_ = s.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	case "response.audio.delta":
		if pcm, err := base64.StdEncoding.DecodeString(ev.Delta); err == nil && len(pcm) > 0 {
			_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, sampleRate, 1), processor.Downstream)
		}
	case "response.audio_transcript.delta":
		if ev.Delta != "" {
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(ev.Delta), processor.Downstream)
		}
	case "response.done":
		_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	case "conversation.item.input_audio_transcription.completed":
		if ev.Transcript != "" {
			_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(ev.Transcript, "", ""), processor.Downstream)
		}
	case "error":
		s.PushError(ctx, "openai realtime error: "+ev.Error.Message, fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	}
}
