package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Service is the xAI Realtime speech-to-speech processor.
type Service struct {
	*processor.Base
	cfg Config

	mu      sync.Mutex
	conn    *websocket.Conn
	connCtx context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex
	wg      sync.WaitGroup

	// established reports whether the server has opened the conversation. xAI
	// only accepts session configuration from that point on, so a tool change
	// arriving earlier is folded into the initial session update instead.
	established bool

	// tools is the function-calling configuration currently advertised to the
	// session. The model generates continuously, so it does not re-read the
	// context between turns: every change must be pushed to it with a
	// session.update. It is guarded by mu.
	tools []frames.Tool
}

// New builds an xAI Realtime service.
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
	s.Base = processor.New("XAIRealtime", s)
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
			s.PushError(ctx, "xai realtime connect failed", err, true)
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
			s.syncTools(fr.Context.Tools())
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMSetToolsFrame:
		// The toolset changed mid-conversation. A text LLM would pick this up on
		// its next run; this model is generating continuously, so tell it now.
		s.syncTools(fr.Tools)
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return s.PushFrame(ctx, f, dir)
	default:
		// A tool-choice change is not forwarded: the session has no equivalent
		// control, so the model always decides for itself.
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the session and stops the read loop.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

// connect dials the Realtime WebSocket and starts the read loop. The session is
// configured once the server opens the conversation, not here.
func (s *Service) connect(ctx context.Context) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	endpoint := s.cfg.BaseURL + "?model=" + url.QueryEscape(s.cfg.Model)
	conn, resp, err := websocket.Dial(ctx, endpoint, &websocket.DialOptions{HTTPHeader: header})
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
	s.established = false
	s.mu.Unlock()

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// sessionUpdateMsg configures the session.
type sessionUpdateMsg struct {
	Type    string         `json:"type"`
	Session map[string]any `json:"session"`
}

// audioAppendMsg appends a chunk of input PCM to the model's input buffer.
type audioAppendMsg struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// audioFormat renders the session's PCM format. xAI nests the format under an
// input/output audio object rather than naming each direction's format at the
// top level.
func (s *Service) audioFormat() map[string]any {
	format := map[string]any{
		"format": map[string]any{keyType: pcmFormat, "rate": s.cfg.sampleRate()},
	}
	return map[string]any{"input": format, "output": format}
}

// sessionUpdate is the session configuration message. The model is not part of
// it: xAI selects the model on the handshake.
func (s *Service) sessionUpdate() sessionUpdateMsg {
	session := map[string]any{
		"voice": s.cfg.Voice,
		"audio": s.audioFormat(),
	}
	if s.cfg.Instructions != "" {
		session["instructions"] = s.cfg.Instructions
	}
	if s.cfg.serverVAD() {
		session["turn_detection"] = map[string]any{keyType: "server_vad"}
	} else {
		session["turn_detection"] = nil
	}
	if tools := s.toolSpecs(s.currentTools()); tools != nil {
		session["tools"] = tools
	}
	return sessionUpdateMsg{Type: "session.update", Session: session}
}

// toolSpecs renders the tool list: xAI's built-in search tools from the config,
// then the function tools the pipeline advertises. It is nil when there are
// none, so a session without tools is configured without the field.
func (s *Service) toolSpecs(tools []frames.Tool) []map[string]any {
	specs := make([]map[string]any, 0, len(tools)+3)
	if s.cfg.WebSearch {
		specs = append(specs, map[string]any{keyType: "web_search"})
	}
	if s.cfg.XSearch {
		spec := map[string]any{keyType: "x_search"}
		if len(s.cfg.XSearchHandles) > 0 {
			spec["allowed_x_handles"] = s.cfg.XSearchHandles
		}
		specs = append(specs, spec)
	}
	if fs := s.cfg.FileSearch; fs != nil {
		spec := map[string]any{keyType: "file_search", "vector_store_ids": fs.VectorStoreIDs}
		if fs.MaxResults > 0 {
			spec["max_num_results"] = fs.MaxResults
		}
		specs = append(specs, spec)
	}
	for _, t := range tools {
		spec := map[string]any{keyType: "function", "name": t.Name}
		if t.Description != "" {
			spec["description"] = t.Description
		}
		if len(t.Parameters) > 0 {
			spec["parameters"] = t.Parameters
		}
		specs = append(specs, spec)
	}
	if len(specs) == 0 {
		return nil
	}
	return specs
}

// currentTools returns the function tools currently advertised to the session.
func (s *Service) currentTools() []frames.Tool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tools
}

// syncTools records the function-calling configuration and, when it differs from
// what the live session was told, pushes a session.update so the continuously
// running model picks the change up. Before the conversation opens it only
// records: the initial session update carries whatever has been recorded by then.
func (s *Service) syncTools(tools []frames.Tool) {
	s.mu.Lock()
	if sameTools(s.tools, tools) {
		s.mu.Unlock()
		return
	}
	s.tools = tools
	live := s.established
	s.mu.Unlock()

	if !live {
		return
	}
	specs := s.toolSpecs(tools)
	if specs == nil {
		// Clearing the toolset still has to reach the model.
		specs = []map[string]any{}
	}
	if err := s.send(sessionUpdateMsg{
		Type:    "session.update",
		Session: map[string]any{"tools": specs},
	}); err != nil {
		slog.Warn("xai realtime tool update failed", "err", err)
	}
}

// sameTools reports whether two toolsets are equivalent for session purposes.
func sameTools(a, b []frames.Tool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Name != b[i].Name ||
			a[i].Description != b[i].Description ||
			!bytes.Equal(a[i].Parameters, b[i].Parameters) {
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
	s.established = false
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
// events. Token accounting arrives on response.done, either at the top level or
// nested in the response.
type serverEvent struct {
	Type       string          `json:"type"`
	Delta      string          `json:"delta"`
	Transcript string          `json:"transcript"`
	Usage      *usage          `json:"usage"`
	Response   *responseObject `json:"response"`
	Error      struct {
		Message string `json:"message"`
	} `json:"error"`
}

// responseObject is the completed-response payload on a response.done event.
type responseObject struct {
	Status string `json:"status"`
	Usage  *usage `json:"usage"`
}

// usage is the per-response token accounting. xAI reports totals only, with no
// per-modality breakdown, so the audio and text token fields stay zero.
type usage struct {
	TotalTokens  int64 `json:"total_tokens"`  //nolint:tagliatelle // xAI wire field
	InputTokens  int64 `json:"input_tokens"`  //nolint:tagliatelle // xAI wire field
	OutputTokens int64 `json:"output_tokens"` //nolint:tagliatelle // xAI wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
}

// readLoop reads server events until the connection is closed or canceled.
func (s *Service) readLoop(conn *websocket.Conn, connCtx context.Context) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(connCtx)
		if err != nil {
			if connCtx.Err() == nil {
				slog.Debug("xai realtime read ended", "err", err)
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
	case "conversation.created":
		s.configureSession(ctx)
	case "input_audio_buffer.speech_started":
		// Server VAD detected user speech: barge in so buffered bot audio drops.
		_ = s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
		_ = s.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
	case "input_audio_buffer.speech_stopped":
		_ = s.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
	case "response.created":
		_ = s.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	case "response.output_audio.delta":
		if pcm, err := base64.StdEncoding.DecodeString(ev.Delta); err == nil && len(pcm) > 0 {
			_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, s.cfg.sampleRate(), 1), processor.Downstream)
		}
	case "response.output_audio_transcript.delta":
		if ev.Delta != "" {
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(ev.Delta), processor.Downstream)
		}
	case "response.done":
		s.finishResponse(ctx, ev)
	case "conversation.item.input_audio_transcription.completed":
		if ev.Transcript != "" {
			_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(ev.Transcript, "", ""), processor.Downstream)
		}
	case "error":
		s.PushError(ctx, "xai realtime error: "+ev.Error.Message, fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	}
}

// configureSession sends the session configuration once the server has opened
// the conversation, which is the point from which xAI accepts it. Later tool
// changes push their own updates.
func (s *Service) configureSession(ctx context.Context) {
	s.mu.Lock()
	s.established = true
	s.mu.Unlock()
	if err := s.send(s.sessionUpdate()); err != nil {
		s.PushError(ctx, "xai realtime session update failed", err, false)
	}
}

// finishResponse closes out a completed response: it reports the token
// accounting, ends the bot's turn, and surfaces a response the model failed to
// produce.
func (s *Service) finishResponse(ctx context.Context, ev serverEvent) {
	s.reportUsage(ctx, ev)
	_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	if ev.Response != nil && ev.Response.Status == "failed" {
		s.PushError(ctx, "xai realtime response failed", errServer, false)
	}
}

// reportUsage forwards the token accounting on a response.done event to metrics
// and telemetry, when usage metrics are enabled. xAI reports it at the top level
// on some responses and inside the response object on others.
func (s *Service) reportUsage(ctx context.Context, ev serverEvent) {
	if !s.UsageMetricsEnabled() {
		return
	}
	u := ev.Usage
	if u == nil && ev.Response != nil {
		u = ev.Response.Usage
	}
	if u == nil || u.TotalTokens == 0 {
		return
	}
	_ = s.PushTokenUsage(ctx, s.cfg.Model, u.tokenUsage())
}
