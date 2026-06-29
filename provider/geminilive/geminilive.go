// Package geminilive is a speech-to-speech service built on Google's Gemini
// Live API (BidiGenerateContent). A single bidirectional WebSocket carries the
// conversation: input audio streams up and the model streams its spoken reply,
// its transcript, and the user's transcript back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The Live API takes 16 kHz mono PCM in and returns 24 kHz
// mono PCM out, so run the transport input at 16000 and output at 24000.
//
// The model's server VAD drives turn-taking: when it reports an interruption the
// service emits an InterruptionFrame (barge-in) so the output transport drops
// buffered bot audio.
package geminilive

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/url"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
)

const (
	defaultBaseURL = "wss://generativelanguage.googleapis.com/ws/" +
		"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	defaultModel = "gemini-live-2.5-flash-native-audio"
	defaultVoice = "Puck"
	// outputSampleRate is the fixed rate of the Live API's output audio.
	outputSampleRate = 24000
	// readLimit bounds a single inbound message; audio parts are large.
	readLimit = 1 << 24
)

// Config configures the Gemini Live service.
type Config struct {
	// APIKey is the Google AI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Live API WebSocket endpoint.
	BaseURL string
	// Model is the Live model id (without the "models/" prefix); empty uses a
	// current default.
	Model string
	// Voice is the prebuilt voice name (e.g. "Puck", "Kore"); empty uses a
	// default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Service is the Gemini Live speech-to-speech processor.
type Service struct {
	*processor.Base
	cfg Config

	mu       sync.Mutex
	conn     *websocket.Conn
	connCtx  context.Context
	cancel   context.CancelFunc
	writeMu  sync.Mutex
	wg       sync.WaitGroup
	ready    atomic.Bool
	speaking bool
}

// New builds a Gemini Live service.
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
	s := &Service{cfg: cfg}
	s.Base = processor.New("GeminiLive", s)
	return s
}

// ProcessFrame opens the session on StartFrame, forwards input audio to the
// model, and tears the session down when the pipeline ends.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "gemini live connect failed", err, true)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			s.sendAudio(fr.Audio, fr.SampleRate)
			return nil // the model consumes the audio; it does not flow on
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

// connect dials the Live WebSocket, sends the setup message, and starts the read
// loop.
func (s *Service) connect(ctx context.Context) error {
	endpoint := s.cfg.BaseURL + "?key=" + url.QueryEscape(s.cfg.APIKey)
	conn, resp, err := websocket.Dial(ctx, endpoint, nil)
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

	if err := s.send(s.setup()); err != nil {
		cancel()
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return err
	}

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// setup is the initial session-configuration message.
func (s *Service) setup() map[string]any {
	setup := map[string]any{
		"model": "models/" + s.cfg.Model,
		"generationConfig": map[string]any{
			"responseModalities": []string{"AUDIO"},
			"speechConfig": map[string]any{
				"voiceConfig": map[string]any{
					"prebuiltVoiceConfig": map[string]any{"voiceName": s.cfg.Voice},
				},
			},
		},
		"inputAudioTranscription":  map[string]any{},
		"outputAudioTranscription": map[string]any{},
	}
	if s.cfg.Instructions != "" {
		setup["systemInstruction"] = map[string]any{
			"parts": []any{map[string]any{"text": s.cfg.Instructions}},
		}
	}
	return map[string]any{"setup": setup}
}

// sendAudio streams a chunk of input PCM to the model once the session is ready.
func (s *Service) sendAudio(pcm []byte, sampleRate int) {
	if len(pcm) == 0 || !s.ready.Load() {
		return
	}
	_ = s.send(map[string]any{
		"realtimeInput": map[string]any{
			"audio": map[string]any{
				"data":     base64.StdEncoding.EncodeToString(pcm),
				"mimeType": fmt.Sprintf("audio/pcm;rate=%d", sampleRate),
			},
		},
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
		return nil
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(connCtx, websocket.MessageText, data)
}

// disconnect cancels the session, closes the socket, and waits for the read loop.
func (s *Service) disconnect() {
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.mu.Unlock()
	s.ready.Store(false)
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
	s.wg.Wait()
}

// serverMessage is the subset of Live API server messages the service handles.
// The JSON field names below are Gemini's wire protocol (camelCase), so the
// snake_case house style does not apply.

type serverMessage struct {
	SetupComplete *json.RawMessage `json:"setupComplete"` //nolint:tagliatelle // Gemini wire field
	ServerContent *serverContent   `json:"serverContent"` //nolint:tagliatelle // Gemini wire field
}

type serverContent struct {
	ModelTurn *struct {
		Parts []part `json:"parts"`
	} `json:"modelTurn"` //nolint:tagliatelle // Gemini wire field
	InputTranscription  *textPayload `json:"inputTranscription"`  //nolint:tagliatelle // Gemini wire field
	OutputTranscription *textPayload `json:"outputTranscription"` //nolint:tagliatelle // Gemini wire field
	Interrupted         bool         `json:"interrupted"`
	GenerationComplete  bool         `json:"generationComplete"` //nolint:tagliatelle // Gemini wire field
}

type part struct {
	Text       string `json:"text"`
	InlineData *struct {
		MimeType string `json:"mimeType"` //nolint:tagliatelle // Gemini wire field
		Data     string `json:"data"`
	} `json:"inlineData"` //nolint:tagliatelle // Gemini wire field
}

type textPayload struct {
	Text string `json:"text"`
}

// readLoop reads server messages until the connection is closed or canceled.
func (s *Service) readLoop(conn *websocket.Conn, connCtx context.Context) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(connCtx)
		if err != nil {
			if connCtx.Err() == nil {
				slog.Debug("gemini live read ended", "err", err)
			}
			return
		}
		var msg serverMessage
		if json.Unmarshal(data, &msg) != nil {
			continue
		}
		s.handle(connCtx, msg)
	}
}

// handle maps a server message onto downstream pipeline frames.
func (s *Service) handle(ctx context.Context, msg serverMessage) {
	if msg.SetupComplete != nil {
		s.ready.Store(true)
	}
	sc := msg.ServerContent
	if sc == nil {
		return
	}
	if sc.Interrupted {
		s.setSpeaking(ctx, false)
		_ = s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
		_ = s.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
	}
	if sc.InputTranscription != nil && sc.InputTranscription.Text != "" {
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(sc.InputTranscription.Text, "", ""), processor.Downstream)
	}
	if sc.OutputTranscription != nil && sc.OutputTranscription.Text != "" {
		_ = s.PushFrame(ctx, frames.NewLLMTextFrame(sc.OutputTranscription.Text), processor.Downstream)
	}
	if sc.ModelTurn != nil {
		for _, p := range sc.ModelTurn.Parts {
			s.handlePart(ctx, p)
		}
	}
	if sc.GenerationComplete {
		s.setSpeaking(ctx, false)
	}
}

// handlePart emits the audio and any text carried by one model-turn part.
func (s *Service) handlePart(ctx context.Context, p part) {
	if p.Text != "" {
		_ = s.PushFrame(ctx, frames.NewLLMTextFrame(p.Text), processor.Downstream)
	}
	if p.InlineData == nil {
		return
	}
	pcm, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
	if err != nil || len(pcm) == 0 {
		return
	}
	s.setSpeaking(ctx, true)
	_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, outputSampleRate, 1), processor.Downstream)
}

// setSpeaking emits a bot-speaking transition frame on a change of state.
func (s *Service) setSpeaking(ctx context.Context, speaking bool) {
	s.mu.Lock()
	changed := s.speaking != speaking
	s.speaking = speaking
	s.mu.Unlock()
	if !changed {
		return
	}
	if speaking {
		_ = s.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	} else {
		_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	}
}
