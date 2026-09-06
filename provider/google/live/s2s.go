package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/google/uuid"
)

// Service is the Gemini Live speech-to-speech processor.
type Service struct {
	// adapter renders the tools and their results the way the Live API takes
	// them, which is the same way the Gemini API does.
	adapter geminiadapter.Adapter
	*llm.Base
	cfg Config

	mu        sync.Mutex
	conn      *wsutil.Conn
	connCtx   context.Context
	cancel    context.CancelFunc
	writeMu   sync.Mutex
	wg        sync.WaitGroup
	ready     atomic.Bool
	speaking  bool
	connector Connector

	// convo is the conversation this session is part of, as the pipeline last
	// reported it. The model generates continuously and so never re-reads it,
	// but a tool call is answered into it and its results are read back out of
	// it here. Guarded by mu.
	convo *frames.LLMContext
	// tools are the tools the session was opened with. The Live API takes them
	// in the setup message and nowhere else, so a conversation bringing tools
	// to a session that has none is applied by opening another. Guarded by mu.
	tools frames.ToolsSchema
	// toolNames maps a call id to the function it called. A result has to name
	// the function as well as the call, and by the time it comes back only the
	// id is at hand. Guarded by mu.
	toolNames map[string]string
	// sentResults names the tool calls whose result has already gone to the
	// model, so a conversation reported again does not send one twice. Guarded
	// by mu.
	sentResults map[string]bool
}

// Connector customizes how the Live session is addressed and authorized, so a
// deployment with a different endpoint or auth scheme (Vertex AI, which
// addresses models per project and location and authorizes with an OAuth token)
// can reuse this implementation.
type Connector interface {
	// Endpoint returns the WebSocket URL to dial and the headers to dial it
	// with. It takes a context because a scheme may have to mint or refresh a
	// token first.
	Endpoint(ctx context.Context) (string, http.Header, error)
	// ModelPath returns the resource name the setup message identifies the model
	// by.
	ModelPath(model string) string
}

// apiKeyConnector is the standard Gemini Live addressing: the api key travels as
// a query parameter and the model is named relative to the API.
type apiKeyConnector struct {
	baseURL string
	apiKey  string
}

func (c apiKeyConnector) Endpoint(context.Context) (string, http.Header, error) {
	return c.baseURL + "?key=" + url.QueryEscape(c.apiKey), nil, nil
}

func (apiKeyConnector) ModelPath(model string) string { return "models/" + model }

// New builds a Gemini Live service.
func New(cfg Config) *Service {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return NewWithConnector("GeminiLive", apiKeyConnector{baseURL: cfg.BaseURL, apiKey: cfg.APIKey}, cfg)
}

// NewWithConnector builds a Gemini Live service that dials through conn. It is
// the base for deployments that do not use the Gemini API's own endpoint or
// api-key auth; name is the processor label.
func NewWithConnector(name string, conn Connector, cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	s := &Service{
		cfg:         cfg,
		connector:   conn,
		toolNames:   map[string]string{},
		sentResults: map[string]bool{},
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

// ProcessFrame opens the session on StartFrame, forwards input audio to the
// model, and tears the session down when the pipeline ends.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if audio, ok := f.(*frames.InputAudioRawFrame); ok && dir == processor.Downstream {
		// The model consumes the audio, so it does not travel on. Only the
		// processor bookkeeping runs: the LLM base would forward it.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sendAudio(audio.Audio, audio.SampleRate)
		return nil
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "gemini live connect failed", err, true)
		}
		return nil
	case *frames.LLMContextFrame:
		// The conversation the session is part of: the toolset it advertises, and
		// the tool results it has gained since it was last reported. The base
		// holds a conversation back from a service generating continuously, so
		// this is where it travels on.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return nil
	default:
		return nil
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

	s.traceSetup(ctx)
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
		"model": s.connector.ModelPath(s.cfg.Model),
		"generationConfig": map[string]any{
			"responseModalities": []string{modalityAudio},
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
	s.mu.Lock()
	schema := s.tools
	s.mu.Unlock()
	if tools := s.adapter.ToProviderToolsFormat(schema); len(tools) > 0 {
		setup["tools"] = tools
	}
	return map[string]any{"setup": setup}
}

// handleContext takes the conversation the pipeline reported.
//
// The model generates continuously and never re-reads it, so a conversation
// arriving means two things and only two: the toolset it advertises, and the
// results of the tool calls the model asked for.
//
// The toolset is only settled once. The Live API takes it in the setup message
// and offers no way to change it afterwards, so a conversation bringing tools to
// a session opened without them is applied by opening another session with them.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	had := len(s.tools.Standard) > 0 || len(s.tools.Custom) > 0
	s.mu.Unlock()

	schema := convo.ToolsSchema()
	if first && !had && (len(schema.Standard) > 0 || len(schema.Custom) > 0) {
		s.mu.Lock()
		s.tools = schema
		s.mu.Unlock()
		s.reconnectWithTools(ctx)
	}

	// The results already in the first conversation were produced before this
	// session existed, so they are recorded as known rather than sent: the model
	// never asked for them and has nothing to do with them.
	s.processCompletedCalls(ctx, convo, !first)
}

// reconnectWithTools opens the session again so its setup carries the tools. The
// conversation so far is the model's own, and it is lost with the session: the
// Live API keeps it server-side and jargo has no way to replay it.
func (s *Service) reconnectWithTools(ctx context.Context) {
	slog.InfoContext(ctx, "reopening the live session to declare its tools",
		"service", s.Name())
	s.disconnect()
	if err := s.connect(ctx); err != nil {
		s.PushError(ctx, "gemini live reconnect failed", err, true)
	}
}

// processCompletedCalls finds the tool results the model has not been given yet
// and, when send is set, hands each to the session.
//
// The results are read out of the conversation rather than taken from the
// handler directly, because that is where a call's result lands however it was
// produced: a handler that answered at once, one that answered later, or one the
// application answered for.
//
// Nothing prompts a reply afterwards: the Live API answers a tool response on
// its own, unlike the sessions that have to be asked.
func (s *Service) processCompletedCalls(ctx context.Context, convo *frames.LLMContext, send bool) {
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
			}
		}
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
		msg := "gemini live takes no streamed result from an asynchronous tool"
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

// sendToolResult hands one call's result to the session. The response names the
// function as well as the call, so the name the call went out under is looked
// back up here.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	s.mu.Lock()
	name := s.toolNames[toolCallID]
	s.mu.Unlock()

	slog.DebugContext(ctx, "sending a tool result to the live session",
		"service", s.Name(), "tool_call_id", toolCallID, "function", name)
	if err := s.send(map[string]any{
		"toolResponse": map[string]any{
			"functionResponses": []any{map[string]any{
				"id":       toolCallID,
				"name":     name,
				"response": geminiadapter.FunctionResponseDict(result),
			}},
		},
	}); err != nil {
		slog.ErrorContext(ctx, "sending a tool result failed", "service", s.Name(), "err", err)
	}
}

// runToolCalls runs the calls the model asked for in one message.
func (s *Service) runToolCalls(ctx context.Context, tc *toolCall) {
	if tc == nil || len(tc.FunctionCalls) == 0 {
		return
	}
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		// A call's result is written into the conversation and read back out of
		// it, so without one there is nowhere for the answer to go.
		slog.ErrorContext(ctx, "the model called a tool before any conversation reached the service",
			"service", s.Name())
		return
	}

	calls := make([]frames.ToolCall, 0, len(tc.FunctionCalls))
	for _, fc := range tc.FunctionCalls {
		id := fc.ID
		if id == "" {
			// Vertex AI sends no id of its own, so one is made here. It only has
			// to tell this turn's calls apart and pair each with its result.
			id = uuid.NewString()
		}
		args := fc.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		s.mu.Lock()
		s.toolNames[id] = fc.Name
		s.mu.Unlock()
		calls = append(calls, frames.ToolCall{ID: id, Name: fc.Name, Args: args})
	}
	if err := s.RunFunctionCalls(ctx, convo, calls); err != nil {
		slog.ErrorContext(ctx, "running a tool call failed", "service", s.Name(), "err", err)
	}
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
	UsageMetadata *usageMetadata   `json:"usageMetadata"` //nolint:tagliatelle // Gemini wire field
	ToolCall      *toolCall        `json:"toolCall"`      //nolint:tagliatelle // Gemini wire field
}

// toolCall is the model asking for one or more functions to be called. They
// arrive together, as one turn's worth.
type toolCall struct {
	FunctionCalls []functionCall `json:"functionCalls"` //nolint:tagliatelle // Gemini wire field
}

// functionCall is one call: the function, the arguments the model wrote, and the
// id pairing it with its result. Vertex AI sends no id.
type functionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// usageMetadata is the per-turn token accounting the Live API sends alongside a
// completed turn. The *Details lists break the prompt and response token counts
// down by modality (TEXT vs AUDIO), which is how a native-audio model exposes
// how many of its billed tokens were speech.
type usageMetadata struct {
	PromptTokenCount        int64                `json:"promptTokenCount"`        //nolint:tagliatelle // Gemini wire field
	ResponseTokenCount      int64                `json:"responseTokenCount"`      //nolint:tagliatelle // Gemini wire field
	TotalTokenCount         int64                `json:"totalTokenCount"`         //nolint:tagliatelle // Gemini wire field
	CachedContentTokenCount int64                `json:"cachedContentTokenCount"` //nolint:tagliatelle // Gemini wire field
	PromptTokensDetails     []modalityTokenCount `json:"promptTokensDetails"`     //nolint:tagliatelle // Gemini wire field
	ResponseTokensDetails   []modalityTokenCount `json:"responseTokensDetails"`   //nolint:tagliatelle // Gemini wire field
	// CacheTokensDetails splits the cached-content count by modality. Cached
	// audio is priced apart from cached text.
	CacheTokensDetails []modalityTokenCount `json:"cacheTokensDetails"` //nolint:tagliatelle // Gemini wire field
	// ThoughtsTokenCount is the number of completion tokens the model spent
	// reasoning. It is absent on a model that does not reason.
	ThoughtsTokenCount *int64 `json:"thoughtsTokenCount"` //nolint:tagliatelle // Gemini wire field
}

// modalityTokenCount is a token count attributed to one modality (e.g. TEXT or
// AUDIO) within a prompt or response.
type modalityTokenCount struct {
	Modality   string `json:"modality"`
	TokenCount int64  `json:"tokenCount"` //nolint:tagliatelle // Gemini wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape,
// folding the per-modality detail lists into the audio and text breakdowns.
func (u usageMetadata) tokenUsage() frames.LLMTokenUsage {
	usage := frames.LLMTokenUsage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.ResponseTokenCount,
		TotalTokens:      u.TotalTokenCount,
		CacheReadTokens:  new(u.CachedContentTokenCount),
		ReasoningTokens:  u.ThoughtsTokenCount,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	// A modality with no tokens is left out of the breakdown entirely, and the
	// breakdown itself is absent when the model reports none, so a modality that
	// does not appear is unaccounted for rather than zero.
	for _, d := range u.PromptTokensDetails {
		switch d.Modality {
		case modalityAudio:
			usage.InputAudioTokens = frames.AddTokens(usage.InputAudioTokens, d.TokenCount)
		case modalityText:
			usage.InputTextTokens = frames.AddTokens(usage.InputTextTokens, d.TokenCount)
		}
	}
	for _, d := range u.ResponseTokensDetails {
		switch d.Modality {
		case modalityAudio:
			usage.OutputAudioTokens = frames.AddTokens(usage.OutputAudioTokens, d.TokenCount)
		case modalityText:
			usage.OutputTextTokens = frames.AddTokens(usage.OutputTextTokens, d.TokenCount)
		}
	}
	for _, d := range u.CacheTokensDetails {
		if d.Modality == modalityAudio {
			usage.CacheReadAudioTokens = frames.AddTokens(usage.CacheReadAudioTokens, d.TokenCount)
		}
	}
	return usage
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
func (s *Service) readLoop(conn *wsutil.Conn, connCtx context.Context) {
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
	if msg.UsageMetadata != nil {
		// The accounting arrives with the turn it belongs to, so the span
		// covering that turn is opened here and the usage recorded against it.
		spanCtx, end := s.traceResponse(ctx, msg)
		if s.UsageMetricsEnabled() {
			_ = s.PushTokenUsage(spanCtx, msg.UsageMetadata.tokenUsage())
		}
		end()
	}
	if msg.ToolCall != nil {
		s.runToolCalls(ctx, msg.ToolCall)
	}
	sc := msg.ServerContent
	if sc == nil {
		return
	}
	if sc.Interrupted {
		// The bot was cut off, so it gives the floor up and the pipeline is
		// interrupted. No user-speaking frame goes with it: this API reports an
		// interruption but never a turn starting or ending, so a start invented
		// here would have no stop to match it, and everything keyed off those
		// frames (turn tracking, the idle watch, mute strategies) would be left
		// believing the user is still speaking. A pipeline that needs turn
		// boundaries runs its own detection alongside this service.
		s.setSpeaking(ctx, false)
		// Broadcast, not pushed on: the aggregators on either side of this
		// service both act on an interruption, and the one keeping the user's
		// turn sits upstream of it.
		_ = s.BroadcastInterruption(ctx)
	}
	if sc.InputTranscription != nil && sc.InputTranscription.Text != "" {
		// The user's words go upstream, where the user aggregator is: everything
		// downstream of this service is the reply to them.
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(sc.InputTranscription.Text, "", ""), processor.Upstream)
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

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// ServiceMetadataFrame describes this service to the pipeline as a realtime one.
// The user aggregator reads that and stops holding turns open for transcripts,
// which this service is not answering from, and writes the user's message when
// the answer starts rather than when the turn ends, because the transcript for
// what the user said arrives late.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	return f
}
