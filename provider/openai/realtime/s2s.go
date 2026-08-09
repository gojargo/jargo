package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"sync"

	"github.com/coder/websocket"
	realtimeadapter "github.com/gojargo/jargo/adapter/realtime"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
)

// Service is the Realtime speech-to-speech processor.
type Service struct {
	// adapter converts the conversation into what a session takes from it.
	adapter realtimeadapter.Adapter
	*processor.Base
	cfg Config

	mu      sync.Mutex
	conn    *wsutil.Conn
	connCtx context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex
	wg      sync.WaitGroup

	// tools and toolChoice are the function-calling configuration currently
	// advertised to the session. The model generates continuously, so it does not
	// re-read the context between turns: every change must be pushed to it with a
	// session.update. They are guarded by mu.
	tools      frames.ToolsSchema
	toolChoice frames.ToolChoice

	connector Connector
}

// Connector customizes how the session is addressed and authorized, so a
// deployment with a different endpoint or auth scheme (Azure OpenAI, which
// addresses a model deployment per resource and authorizes with an api-key
// header) can reuse this implementation.
type Connector interface {
	// Endpoint returns the WebSocket URL to dial and the headers to dial it
	// with. It takes a context because a scheme may have to mint a credential
	// first.
	Endpoint(ctx context.Context) (string, http.Header, error)
}

// bearerConnector is the standard OpenAI Realtime addressing: the model travels
// as a query parameter and the key as a bearer token.
type bearerConnector struct {
	baseURL string
	model   string
	apiKey  string
}

func (c bearerConnector) Endpoint(context.Context) (string, http.Header, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.apiKey)
	header.Set("OpenAI-Beta", "realtime=v1")
	return c.baseURL + "?model=" + url.QueryEscape(c.model), header, nil
}

// New builds a Realtime service.
func New(cfg Config) *Service {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return NewWithConnector("OpenAIRealtime", bearerConnector{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		apiKey:  cfg.APIKey,
	}, cfg)
}

// NewWithConnector builds a Realtime service that dials through conn. It is the
// base for deployments that do not use OpenAI's own endpoint or bearer auth;
// name is the processor label.
func NewWithConnector(name string, conn Connector, cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.TranscriptionModel == "" {
		cfg.TranscriptionModel = defaultTranscriptionModel
	}
	s := &Service{cfg: cfg, connector: conn}
	s.Base = processor.New(name, s)
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
	case *frames.LLMContextFrame:
		// Seed the function-calling configuration from the shared context.
		if fr.Context != nil {
			s.syncTools(fr.Context.ToolsSchema(), fr.Context.ToolChoice())
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMSetToolsFrame:
		// The toolset changed mid-conversation. A text LLM would pick this up on
		// its next run; this model is generating continuously, so tell it now.
		s.syncTools(frames.ToolsSchema{Standard: fr.Tools}, s.currentToolChoice())
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMSetToolChoiceFrame:
		s.syncTools(s.currentTools(), fr.ToolChoice)
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
	endpoint, header, err := s.connector.Endpoint(ctx)
	if err != nil {
		return err
	}

	conn, err := wsutil.Dial(ctx, endpoint, header, readLimit)
	if err != nil {
		return err
	}

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
		"modalities":          []string{keyAudio, "text"},
		"voice":               s.cfg.Voice,
		"input_audio_format":  "pcm16",
		"output_audio_format": "pcm16",
		"turn_detection":      map[string]any{keyType: "server_vad"},
	}
	if s.cfg.Instructions != "" {
		session["instructions"] = s.cfg.Instructions
	}
	if s.cfg.TranscriptionModel != "-" {
		session["input_audio_transcription"] = map[string]any{"model": s.cfg.TranscriptionModel}
	}
	maps.Copy(session, s.toolSession(s.currentTools(), s.currentToolChoice()))
	return sessionUpdateMsg{Type: msgSessionUpdate, Session: session}
}

// toolSession renders the function-calling part of a session payload. It is
// nil when the conversation advertises no tools.
func (s *Service) toolSession(
	schema frames.ToolsSchema, choice frames.ToolChoice,
) map[string]any {
	return s.adapter.SessionParams(schema, choice).Session()
}

// currentTools returns the toolset currently advertised to the session.
func (s *Service) currentTools() frames.ToolsSchema {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tools
}

// currentToolChoice returns the tool choice currently advertised to the session.
func (s *Service) currentToolChoice() frames.ToolChoice {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.toolChoice
}

// syncTools records the function-calling configuration and, when it differs from
// what the live session was told, pushes a session.update so the continuously
// running model picks the change up. It is a no-op before the session connects;
// the initial sessionUpdate carries whatever has been recorded by then.
func (s *Service) syncTools(schema frames.ToolsSchema, choice frames.ToolChoice) {
	s.mu.Lock()
	if sameTools(s.tools, schema) && s.toolChoice == choice {
		s.mu.Unlock()
		return
	}
	s.tools = schema
	s.toolChoice = choice
	live := s.conn != nil
	s.mu.Unlock()

	if !live {
		return
	}
	session := s.toolSession(schema, choice)
	if session == nil {
		// Clearing the toolset still has to reach the model.
		session = map[string]any{"tools": []map[string]any{}}
	}
	if err := s.send(sessionUpdateMsg{Type: msgSessionUpdate, Session: session}); err != nil {
		slog.Warn("openai realtime tool update failed", "err", err)
	}
}

// sameTools reports whether two toolsets are equivalent for session purposes.
// A custom tool is compared by identity of the slice it came in, which is enough
// to tell a toolset that was replaced from one that was not.
func sameTools(a, b frames.ToolsSchema) bool {
	if len(a.Standard) != len(b.Standard) || len(a.Custom) != len(b.Custom) {
		return false
	}
	for i := range a.Standard {
		if a.Standard[i].Name != b.Standard[i].Name ||
			a.Standard[i].Description != b.Standard[i].Description ||
			!bytes.Equal(a.Standard[i].Parameters, b.Standard[i].Parameters) {
			return false
		}
	}
	for k, av := range a.Custom {
		if len(b.Custom[k]) != len(av) {
			return false
		}
	}
	return true
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
// events; response carries the token accounting on the response.done event.
type serverEvent struct {
	Type       string          `json:"type"`
	Delta      string          `json:"delta"`
	Transcript string          `json:"transcript"`
	Response   *responseObject `json:"response"`
	Error      struct {
		Message string `json:"message"`
	} `json:"error"`
}

// responseObject is the completed-response payload on a response.done event; the
// service reads only its usage.
type responseObject struct {
	Usage *usage `json:"usage"`
}

// usage is the Realtime API's per-response token accounting. The *_token_details
// break the input and output token counts down by modality (text vs audio),
// which is how a speech-to-speech model exposes its audio-token billing.
type usage struct {
	TotalTokens        int64        `json:"total_tokens"`         //nolint:tagliatelle // OpenAI wire field
	InputTokens        int64        `json:"input_tokens"`         //nolint:tagliatelle // OpenAI wire field
	OutputTokens       int64        `json:"output_tokens"`        //nolint:tagliatelle // OpenAI wire field
	InputTokenDetails  tokenDetails `json:"input_token_details"`  //nolint:tagliatelle // OpenAI wire field
	OutputTokenDetails tokenDetails `json:"output_token_details"` //nolint:tagliatelle // OpenAI wire field
}

// tokenDetails is the per-modality (and cache) breakdown of one direction's
// token count.
type tokenDetails struct {
	TextTokens   int64 `json:"text_tokens"`   //nolint:tagliatelle // OpenAI wire field
	AudioTokens  int64 `json:"audio_tokens"`  //nolint:tagliatelle // OpenAI wire field
	CachedTokens int64 `json:"cached_tokens"` //nolint:tagliatelle // OpenAI wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:      u.InputTokens,
		CompletionTokens:  u.OutputTokens,
		TotalTokens:       u.TotalTokens,
		CacheReadTokens:   u.InputTokenDetails.CachedTokens,
		InputAudioTokens:  u.InputTokenDetails.AudioTokens,
		OutputAudioTokens: u.OutputTokenDetails.AudioTokens,
		InputTextTokens:   u.InputTokenDetails.TextTokens,
		OutputTextTokens:  u.OutputTokenDetails.TextTokens,
	}
}

// readLoop reads server events until the connection is closed or canceled.
func (s *Service) readLoop(conn *wsutil.Conn, connCtx context.Context) {
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
		s.reportUsage(ctx, ev)
		_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	case "conversation.item.input_audio_transcription.completed":
		if ev.Transcript != "" {
			_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(ev.Transcript, "", ""), processor.Downstream)
		}
	case "error":
		s.PushError(ctx, "openai realtime error: "+ev.Error.Message, fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	}
}

// reportUsage forwards the token accounting on a response.done event to metrics
// and telemetry, when usage metrics are enabled.
func (s *Service) reportUsage(ctx context.Context, ev serverEvent) {
	if ev.Response == nil || ev.Response.Usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	_ = s.PushTokenUsage(ctx, s.cfg.Model, ev.Response.Usage.tokenUsage())
}

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }
