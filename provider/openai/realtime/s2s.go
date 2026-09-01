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
	*llm.Base
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

	// convo is the conversation this session is part of, as the pipeline last
	// reported it. The model generates continuously and so never re-reads it,
	// but a tool call is answered into it and its results are read back out of
	// it here. Guarded by mu.
	convo *frames.LLMContext
	// pendingCalls are the tool calls the model has announced but not finished
	// naming the arguments for, by call id. Guarded by mu.
	pendingCalls map[string]string
	// sentResults names the tool calls whose result has already gone to the
	// model, so a conversation reported again does not send one twice. Guarded
	// by mu.
	sentResults map[string]bool

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
	s := &Service{
		cfg:          cfg,
		connector:    conn,
		pendingCalls: map[string]string{},
		sentResults:  map[string]bool{},
	}
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New(name, s, llm.WithContinuousGeneration())
	s.SetModel(cfg.Model)
	return s
}

// Generate is never called: this service generates continuously from the audio
// it is sent, and says so with llm.WithContinuousGeneration, so the base never
// asks it to answer a conversation. It exists because the base identifies the
// service it belongs to by the generator it was built with.
func (s *Service) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return errNotGenerator
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
		// The conversation the session is part of: the toolset it advertises, and
		// the tool results it has gained since it was last reported.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMSetToolsFrame:
		// The toolset changed mid-conversation. A text LLM would pick this up on
		// its next run; this model is generating continuously, so tell it now.
		// The handlers the new tools carry are registered here for the same
		// reason: the base does that when a conversation arrives, and this
		// service does not get one per turn.
		s.syncTools(frames.ToolsSchema{Standard: fr.Tools}, s.currentToolChoice())
		s.SyncToolHandlers(ctx, fr.Tools)
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

	setup := s.sessionUpdate()
	s.traceSetup(ctx, setup.Session)
	if err := s.send(setup); err != nil {
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

// handleContext takes the conversation the pipeline reported.
//
// The model generates continuously and never re-reads it, so a conversation
// arriving means two things and only two: the toolset it advertises may have
// changed, and it may have gained the result of a tool call the model asked for.
// Both are pushed to the session here.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	s.mu.Unlock()

	s.syncTools(convo.ToolsSchema(), convo.ToolChoice())

	// The results already in the first conversation were produced before this
	// session existed, so they are recorded as known rather than sent: the model
	// never asked for them and has nothing to do with them.
	s.processCompletedCalls(ctx, convo, !first)
}

// processCompletedCalls finds the tool results the model has not been given yet
// and, when send is set, hands each to the session and asks for a reply.
//
// The results are read out of the conversation rather than taken from the
// handler directly, because that is where a call's result lands however it was
// produced: a handler that answered at once, one that answered later, or one the
// application answered for.
func (s *Service) processCompletedCalls(ctx context.Context, convo *frames.LLMContext, send bool) {
	sent := false
	for _, m := range convo.Messages() {
		for _, r := range m.ToolResults {
			if s.resultSeen(r.ID) {
				continue
			}
			// A call still running has a placeholder standing in for its result.
			// It is not an answer, so it is neither sent nor recorded: the real
			// one has still to come.
			if r.Content == frames.ToolResultInProgress {
				continue
			}
			if async, ok := frames.ParseAsyncToolMessage(m); ok && !s.asyncResultSendable(ctx, async) {
				s.recordResult(r.ID)
				continue
			}
			s.recordResult(r.ID)
			if send {
				s.sendToolResult(ctx, r.ID, r.Content)
				sent = true
			}
		}
	}
	if sent {
		// The model has been told what its calls returned, so it can speak the
		// answer. It is generating from audio otherwise, and nothing else would
		// prompt it.
		s.createResponse(ctx)
	}
}

// asyncResultSendable reports whether an asynchronous tool's message carries a
// result this session can take. Only the final one does: the session has no way
// to receive a result in parts, and the marker that opens the call says nothing
// the model does not already know, having made the call itself.
func (s *Service) asyncResultSendable(ctx context.Context, m frames.AsyncToolMessage) bool {
	switch m.Kind {
	case frames.AsyncToolFinal:
		return true
	case frames.AsyncToolIntermediate:
		msg := "openai realtime takes no streamed result from an asynchronous tool"
		slog.ErrorContext(ctx, msg, "function", m.ToolCallID)
		s.PushError(ctx, msg, nil, false)
		return false
	default:
		return false
	}
}

// resultSeen reports whether the model has already been given this call's
// result.
func (s *Service) resultSeen(toolCallID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sentResults[toolCallID]
}

// recordResult marks a call's result as settled, so a conversation reported
// again does not send it twice.
func (s *Service) recordResult(toolCallID string) {
	s.mu.Lock()
	s.sentResults[toolCallID] = true
	s.mu.Unlock()
}

// sendToolResult hands one call's result to the session.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	slog.DebugContext(ctx, "sending a tool result to the realtime session",
		"service", s.Name(), "tool_call_id", toolCallID)
	if err := s.send(map[string]any{
		keyType: "conversation.item.create",
		"item": map[string]any{
			keyType:   "function_call_output",
			"call_id": toolCallID,
			"output":  result,
		},
	}); err != nil {
		slog.ErrorContext(ctx, "sending a tool result failed", "service", s.Name(), "err", err)
	}
}

// createResponse asks the model to speak. The session generates from audio on
// its own, so this is for the turns nothing spoken prompted: the reply to a tool
// result.
func (s *Service) createResponse(ctx context.Context) {
	if err := s.send(map[string]any{keyType: "response.create"}); err != nil {
		slog.ErrorContext(ctx, "asking for a response failed", "service", s.Name(), "err", err)
	}
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
	// Reconfiguring a live session is the same operation as configuring it at the
	// start, so it is recorded the same way: a trace shows every toolset the
	// model was given, not just the one it opened with.
	s.traceSetup(context.Background(), session)
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
	// CallID and Arguments carry a tool call the model finished naming. The
	// arguments arrive as the JSON text the model wrote, not as a decoded
	// object.
	CallID    string `json:"call_id"` //nolint:tagliatelle // OpenAI wire field
	Arguments string `json:"arguments"`
	// Item is the conversation item an item event is about. A tool call is
	// announced as one of these before its arguments are finished.
	Item  *conversationItem `json:"item"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

// conversationItem is the subset of a conversation item we read: enough to
// recognize a tool call and name it.
type conversationItem struct {
	Type   string `json:"type"`
	Name   string `json:"name"`
	CallID string `json:"call_id"` //nolint:tagliatelle // OpenAI wire field
}

// responseObject is the completed-response payload on a response.done event: the
// token accounting the turn is billed on, and what the model produced.
type responseObject struct {
	ID     string               `json:"id"`
	Status string               `json:"status"`
	Usage  *usage               `json:"usage"`
	Output []responseOutputItem `json:"output"`
}

// responseOutputItem is one item the model produced: a spoken message, or a
// function it asked to have called.
type responseOutputItem struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Name    string `json:"name"`
	CallID  string `json:"call_id"` //nolint:tagliatelle // OpenAI wire field
	Content []struct {
		Transcript string `json:"transcript"`
	} `json:"content"`
}

// usage is the Realtime API's per-response token accounting. The *_token_details
// break the input and output token counts down by modality (text vs audio),
// which is how a speech-to-speech model exposes its audio-token billing.
type usage struct {
	TotalTokens        int64         `json:"total_tokens"`         //nolint:tagliatelle // OpenAI wire field
	InputTokens        int64         `json:"input_tokens"`         //nolint:tagliatelle // OpenAI wire field
	OutputTokens       int64         `json:"output_tokens"`        //nolint:tagliatelle // OpenAI wire field
	InputTokenDetails  *tokenDetails `json:"input_token_details"`  //nolint:tagliatelle // OpenAI wire field
	OutputTokenDetails *tokenDetails `json:"output_token_details"` //nolint:tagliatelle // OpenAI wire field
}

// tokenDetails is the per-modality (and cache) breakdown of one direction's
// token count. The counts are pointers because the model reports only the ones
// that apply to it, and a breakdown it omits is not the same as one it reports
// as zero.
type tokenDetails struct {
	TextTokens   *int64 `json:"text_tokens"`   //nolint:tagliatelle // OpenAI wire field
	AudioTokens  *int64 `json:"audio_tokens"`  //nolint:tagliatelle // OpenAI wire field
	CachedTokens *int64 `json:"cached_tokens"` //nolint:tagliatelle // OpenAI wire field
	// CachedTokenDetails splits the cache-read count by modality. Cached audio
	// is priced apart from cached text, so it is reported separately.
	CachedTokenDetails *struct {
		TextTokens  *int64 `json:"text_tokens"`  //nolint:tagliatelle // OpenAI wire field
		AudioTokens *int64 `json:"audio_tokens"` //nolint:tagliatelle // OpenAI wire field
	} `json:"cached_tokens_details"` //nolint:tagliatelle // OpenAI wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	out := frames.LLMTokenUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
	if d := u.InputTokenDetails; d != nil {
		out.CacheReadTokens = d.CachedTokens
		out.InputAudioTokens = d.AudioTokens
		out.InputTextTokens = d.TextTokens
		if c := d.CachedTokenDetails; c != nil {
			out.CacheReadAudioTokens = c.AudioTokens
		}
	}
	if d := u.OutputTokenDetails; d != nil {
		out.OutputAudioTokens = d.AudioTokens
		out.OutputTextTokens = d.TextTokens
	}
	return out
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
	case "conversation.item.added":
		s.trackToolCall(ev.Item)
	case "response.function_call_arguments.done":
		s.runToolCall(ctx, ev)
	case "error":
		s.PushError(ctx, "openai realtime error: "+ev.Error.Message, fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	}
}

// trackToolCall records a tool call the model has announced. The call is named
// here and its arguments finish arriving later, so the name has to be kept until
// they do.
func (s *Service) trackToolCall(item *conversationItem) {
	if item == nil || item.Type != "function_call" || item.CallID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, tracked := s.pendingCalls[item.CallID]; tracked {
		slog.Warn("the realtime session announced a tool call it had already announced",
			"service", s.Name(), "tool_call_id", item.CallID)
		return
	}
	s.pendingCalls[item.CallID] = item.Name
}

// runToolCall runs a call whose arguments the model has finished writing.
//
// It runs on the arguments rather than waiting for the response to complete,
// because a response that only calls a tool may never report itself done.
func (s *Service) runToolCall(ctx context.Context, ev serverEvent) {
	s.mu.Lock()
	name, tracked := s.pendingCalls[ev.CallID]
	// Taken out first, so a repeated event cannot run the same call twice.
	delete(s.pendingCalls, ev.CallID)
	convo := s.convo
	s.mu.Unlock()

	if !tracked {
		slog.WarnContext(ctx, "the realtime session finished a tool call it never announced",
			"service", s.Name(), "tool_call_id", ev.CallID)
		return
	}
	if convo == nil {
		// A call's result is written into the conversation and read back out of
		// it, so without one there is nowhere for the answer to go.
		slog.ErrorContext(ctx, "the model called a tool before any conversation reached the service",
			"service", s.Name(), "function", name)
		return
	}

	args := json.RawMessage(ev.Arguments)
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	call := frames.ToolCall{ID: ev.CallID, Name: name, Args: args}
	if err := s.RunFunctionCalls(ctx, convo, []frames.ToolCall{call}); err != nil {
		slog.ErrorContext(ctx, "running a tool call failed", "service", s.Name(),
			"function", name, "err", err)
	}
}

// reportUsage records the completed turn and the token accounting it was billed
// on. The accounting arrives with the response it belongs to, so the span
// covering that response is opened here and the usage recorded against it.
func (s *Service) reportUsage(ctx context.Context, ev serverEvent) {
	spanCtx, end := s.traceResponse(ctx, ev.Response)
	defer end()
	if ev.Response == nil || ev.Response.Usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	_ = s.PushTokenUsage(spanCtx, ev.Response.Usage.tokenUsage())
}

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }
