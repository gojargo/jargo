package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/adapter"
	realtimeadapter "github.com/gojargo/jargo/adapter/realtime"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
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

	// live is what the service holds of its configuration while the pipeline
	// runs, which the session is configured from.
	live *Settings
	// reasoningWarned keeps the report of reasoning stripped for a model that
	// cannot take it to one.
	reasoningWarned atomic.Bool
	// sessionReady says the server has applied the session's configuration.
	// Guarded by mu.
	sessionReady bool
	// runWhenReady says a response was asked for before the session was ready,
	// and is created once it is. Guarded by mu.
	runWhenReady bool
	// needsConversationSetup says the session has not been seeded with the
	// conversation yet; the first response does it. Guarded by mu.
	needsConversationSetup bool
	// addedManually are the items this service put into the conversation,
	// whose echo from the server is not the model's doing. Guarded by mu.
	addedManually map[string]bool
	// assistantResponding says an assistant message is being produced, which an
	// interruption closes. Guarded by mu.
	assistantResponding bool
	// retrievals are the callers waiting on a conversation item, by item id.
	// Guarded by mu.
	retrievals map[string][]chan retrieval

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
	// openCalls are the calls the session is waiting on, by call id: only these
	// can be answered with a function_call_output. Cleared on disconnect, since
	// a new session knows none of the old ids. Guarded by mu.
	openCalls map[string]bool
	// resultsAwaitingResponse says something was delivered that no response
	// has taken in yet. A response starting clears it; a request to respond
	// while it stands creates one. Guarded by mu.
	resultsAwaitingResponse bool
	// responseCreateOwed says a response was asked for while one was already
	// running, and is owed once that one is done. Guarded by mu.
	responseCreateOwed bool
	// responseActive says a response is running. Guarded by mu.
	responseActive bool

	// audioResp is the assistant audio item currently being played, which an
	// interruption truncates on the server so the conversation the model keeps
	// holds what the caller actually heard. Nil when nothing is playing.
	// Guarded by mu.
	audioResp *audioResponse

	connector Connector
	// write, when set, stands in for the socket: tests capture what the service
	// sends through it.
	write func(v any) error

	// inRS converts input audio to the session's fixed 24 kHz, and inRate and
	// inChannels are the format it was built for: it is rebuilt when the input
	// format changes. Guarded by inMu.
	inMu       sync.Mutex
	inRS       *resample.Resampler
	inRate     int
	inChannels int
}

// audioResponse tracks the assistant audio item the session is playing: which
// item it is, when it started, and how much audio has arrived. An interruption
// reads it to say how much of the item the caller heard.
type audioResponse struct {
	itemID       string
	contentIndex int
	responseID   string
	started      time.Time
	total        int
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
	s := &Service{
		cfg:                    cfg,
		live:                   newSettings(cfg),
		connector:              conn,
		pendingCalls:           map[string]string{},
		addedManually:          map[string]bool{},
		retrievals:             map[string][]chan retrieval{},
		needsConversationSetup: true,
		sentResults:            map[string]bool{},
		openCalls:              map[string]bool{},
	}
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New(name, s, llm.WithContinuousGeneration())
	s.SetModel(cfg.Model)
	s.SetSystemInstruction(cfg.Instructions)
	s.Events().Register(EventConversationItemCreated, false)
	s.Events().Register(EventConversationItemUpdated, false)
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
//
//nolint:cyclop // one case per frame the session answers
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if in, ok := f.(frames.InputAudioFrame); ok && dir == processor.Downstream {
		audio := in.AudioData()
		// The model consumes the audio, so it does not travel on. Only the
		// processor bookkeeping runs: the LLM base would forward it.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sendAudio(ctx, *audio)
		return nil
	}
	if _, ok := f.(*frames.InterruptionFrame); ok {
		// Before the interruption travels on, so what closes the response in
		// progress reaches the processors after this one ahead of it.
		s.handleInterruption(ctx)
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.BotStoppedSpeakingFrame:
		// The audio has played out, so there is no longer an item an
		// interruption would cut back.
		s.mu.Lock()
		s.audioResp = nil
		s.mu.Unlock()
		return nil
	case *frames.LLMContextFrame:
		// The conversation the session is part of: the toolset it advertises, and
		// the tool results it has gained since it was last reported.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context, dir)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMMessagesAppendFrame:
		s.handleMessagesAppend(ctx, fr)
		return nil
	case *frames.LLMSetToolsFrame:
		// The toolset changed mid-conversation. A text LLM would pick this up on
		// its next run; this model is generating continuously, so the session is
		// updated now, with the toolset the conversation holds. The handlers the
		// new tools carry are registered here for the same reason: the base does
		// that when a conversation arrives, and this service does not get one per
		// turn.
		s.SyncToolHandlers(ctx, frames.ToolsSchema{Standard: fr.Tools})
		s.sendSessionUpdate(ctx)
		return nil
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
	s.closeResampler()
	return s.Base.Cleanup(ctx)
}

// Setup sets the service up and opens the session, ahead of the pipeline
// starting. A session that cannot be opened is reported as an error.
func (s *Service) Setup(ctx context.Context, setup processor.Setup) error {
	if err := s.Base.Setup(ctx, setup); err != nil {
		return err
	}
	if err := s.connect(ctx); err != nil {
		s.PushError(ctx, "openai realtime: error connecting", err, false)
	}
	return nil
}

// connect dials the Realtime WebSocket and starts the read loop. The session is
// configured once the server reports it created.
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

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// sendSessionUpdate configures the session from the properties the service
// holds, on a copy so the stored ones are left as they are.
//
// With a conversation, the adapter settles the instruction and the tools: the
// instruction the service holds wins over the conversation's prompt, and the
// conversation's tools win over the configured ones, the built-in tools riding
// along either way. Without one, the session carries the configured tools and
// the built-in ones.
func (s *Service) sendSessionUpdate(ctx context.Context) {
	sp := s.sessionProperties().Copy()
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo != nil {
		params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{
			SystemInstruction: s.SystemInstruction(),
			ServiceTools:      sp.Tools,
		})
		if err != nil {
			s.PushError(ctx, "openai realtime: converting the conversation", err, false)
			return
		}
		if params.SystemInstruction != "" {
			sp.Instructions = params.SystemInstruction
		}
		sp.WireTools = params.Tools
	} else {
		sp.WireTools = s.adapter.ToProviderToolsFormat(s.adapter.WithBuiltins(sp.Tools))
	}
	s.sendEvent(ctx, events.NewSessionUpdateEvent(s.stripUnsupportedReasoning(sp)))
}

// PushFrame pushes a frame, delivering function call outcomes to the session on
// the way.
//
// Function call results and cancellations are broadcast by the base; the
// downstream copy is observed here and sent to the session as it is produced,
// intermediate results included. The frames travel on to the assistant
// aggregator, which records them in the conversation and decides when the
// model should answer.
func (s *Service) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if dir == processor.Downstream {
		switch fr := f.(type) {
		case *frames.FunctionCallResultFrame:
			s.handleFunctionCallResult(ctx, fr)
		case *frames.FunctionCallCancelFrame:
			s.handleFunctionCallCancel(ctx, fr)
		}
	}
	return s.Base.PushFrame(ctx, f, dir)
}

// handleContext takes the conversation the pipeline reported.
//
// The model generates continuously and never re-reads it, so a conversation
// arriving means three things: the toolset it advertises may have changed, it
// may hold a tool result that reached it some other way than from this
// service, and, traveling upstream, it is the assistant aggregator asking the
// model to answer what was delivered since its last response.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext, dir processor.Direction) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	s.mu.Unlock()

	if first {
		// The results already in the first conversation were produced before
		// this session existed, so they are recorded as known rather than sent:
		// the model never asked for them and has nothing to do with them.
		s.processCompletedCalls(ctx, convo, false)
		// The model answers at the first opportunity.
		s.createResponse(ctx)
		return
	}

	// Results this service pushed are already sent; this catches any written
	// into the conversation some other way.
	s.processCompletedCalls(ctx, convo, true)

	s.mu.Lock()
	respond := dir == processor.Upstream && s.resultsAwaitingResponse
	s.mu.Unlock()
	if respond {
		s.createResponse(ctx)
	}
}

// handleMessagesAppend puts appended messages into the conversation as items,
// for the model to take in.
//
// Whether the model answers them is the aggregator's call: an append that asks
// to run comes back to this service as the conversation the assistant
// aggregator pushes upstream, which creates the response (see handleContext).
// Before the session's conversation is set up, nothing is sent: the setup sends
// the conversation, which the aggregator records the messages in.
func (s *Service) handleMessagesAppend(ctx context.Context, f *frames.LLMMessagesAppendFrame) {
	s.mu.Lock()
	ready := s.sessionReady && !s.needsConversationSetup
	s.mu.Unlock()
	if !ready {
		return
	}
	for _, m := range f.Messages {
		text := m.StandardText()
		if text == "" {
			continue
		}
		role := "system"
		if m.Role == frames.RoleAssistant {
			role = roleAssistant
		}
		s.sendMessageItem(ctx, text, role)
	}
	s.mu.Lock()
	s.resultsAwaitingResponse = true
	s.mu.Unlock()
}

// processCompletedCalls sends any tool result the conversation holds that has
// not reached the session yet, when send is set.
//
// Results this service produced are already sent, as they are produced (see
// PushFrame), so what this finds is a result written into the conversation
// some other way, plus, with send unset, the results a conversation brings
// with it at the start of a session.
func (s *Service) processCompletedCalls(ctx context.Context, convo *frames.LLMContext, send bool) {
	for _, m := range convo.Messages() {
		if async, ok := frames.ParseAsyncToolMessage(m); ok {
			// The started marker opens a call the model already knows it is
			// waiting on, and an intermediate result was sent as a message when
			// it was produced; the call stays open for the final one.
			if async.Kind != frames.AsyncToolFinal || s.resultSeen(async.ToolCallID) {
				continue
			}
			// Delivered through the tool-result channel, as a synchronous
			// result is, just later.
			s.recordResult(async.ToolCallID)
			if send {
				s.awaitResponse()
				s.sendToolResult(ctx, async.ToolCallID, async.Result)
			}
			continue
		}
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
			s.recordResult(r.ID)
			if send {
				s.awaitResponse()
				s.sendToolResult(ctx, r.ID, r.Content)
			}
		}
	}
}

// awaitResponse records that something was delivered that no response has
// taken in yet.
func (s *Service) awaitResponse() {
	s.mu.Lock()
	s.resultsAwaitingResponse = true
	s.mu.Unlock()
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
	s.mu.Lock()
	delete(s.openCalls, toolCallID)
	s.mu.Unlock()
	item := events.NewConversationItem(events.ItemFunctionCallOutput)
	item.CallID, item.Output = toolCallID, result
	s.sendEvent(ctx, events.NewConversationItemCreateEvent(item))
}

// sendMessageItem puts text into the conversation for the model to take in,
// without answering it.
func (s *Service) sendMessageItem(ctx context.Context, text, role string) {
	item := events.NewConversationItem(events.ItemMessage)
	item.Role = role
	item.Content = []events.ItemContent{{Type: "input_text", Text: text}}
	s.sendEvent(ctx, events.NewConversationItemCreateEvent(item))
}

// handleFunctionCallResult sends one function call result to the session as it
// is produced.
//
// An intermediate result cannot go through the tool-result channel, which
// takes one output per call, so it goes in as a message carrying the same
// envelope a text LLM reads from the conversation. A final result answers the
// call, unless a reconnect has left the session without the id to answer, in
// which case it goes in as a message too.
func (s *Service) handleFunctionCallResult(ctx context.Context, f *frames.FunctionCallResultFrame) {
	result := f.Result
	if result == "" {
		result = frames.ToolResultCompleted
	}
	s.mu.Lock()
	open := s.openCalls[f.ToolCallID]
	s.mu.Unlock()

	switch {
	case !f.Properties.Final():
		s.sendMessageItem(ctx, frames.NewAsyncToolIntermediateMessage(f.ToolCallID, result).Text, "system")
	case open:
		s.sendToolResult(ctx, f.ToolCallID, result)
	default:
		s.sendMessageItem(ctx, frames.NewAsyncToolFinalMessage(f.ToolCallID, result).Text, "system")
	}

	s.mu.Lock()
	s.sentResults[f.ToolCallID] = true
	s.resultsAwaitingResponse = true
	s.mu.Unlock()
}

// handleFunctionCallCancel settles a canceled call, so the session is not left
// waiting on it.
func (s *Service) handleFunctionCallCancel(ctx context.Context, f *frames.FunctionCallCancelFrame) {
	s.mu.Lock()
	open := s.openCalls[f.ToolCallID]
	s.mu.Unlock()
	if !open {
		return
	}
	s.sendToolResult(ctx, f.ToolCallID, frames.ToolResultCanceled)
	s.recordResult(f.ToolCallID)
}

// createResponse asks the model to respond.
//
// Before the session is ready, the response waits for it. The session refuses a
// second response while one is running, so one asked for then is held until
// that one is done rather than lost to the error. The first response seeds the
// session with the conversation, and configures it again now that the
// conversation's instruction and tools are known.
func (s *Service) createResponse(ctx context.Context) {
	s.mu.Lock()
	if !s.sessionReady {
		s.runWhenReady = true
		s.mu.Unlock()
		return
	}
	if s.responseActive {
		s.responseCreateOwed = true
		s.mu.Unlock()
		slog.DebugContext(ctx, "a response is already running, holding the next one until it is done",
			"service", s.Name())
		return
	}
	convo, setup := s.convo, s.needsConversationSetup
	s.mu.Unlock()
	if convo == nil {
		return
	}

	if setup {
		params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{})
		if err != nil {
			s.PushError(ctx, "openai realtime: converting the conversation", err, false)
			return
		}
		slog.DebugContext(ctx, "setting the realtime conversation up", "service", s.Name(),
			"items", len(params.Messages))
		for _, item := range params.Messages {
			s.mu.Lock()
			s.addedManually[item.ID] = true
			s.mu.Unlock()
			s.sendEvent(ctx, events.NewConversationItemCreateEvent(item))
		}
		s.sendSessionUpdate(ctx)
		s.mu.Lock()
		s.needsConversationSetup = false
		s.mu.Unlock()
	}

	// Active from the moment the request is sent, not from response.created: a
	// second request in between would be refused.
	s.mu.Lock()
	s.responseActive = true
	s.mu.Unlock()
	slog.DebugContext(ctx, "creating a response", "service", s.Name())
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	s.StartProcessingMetrics()
	s.StartTTFBMetrics()
	s.sendEvent(ctx, events.NewResponseCreateEvent(&events.ResponseProperties{
		OutputModalities: s.enabledModalities(),
	}))
}

// enabledModalities is what a response is made of: audio when the session
// responds with it, text otherwise. The API takes one modality per response.
func (s *Service) enabledModalities() []events.Modality {
	modalities := s.sessionProperties().OutputModalities
	if len(modalities) == 0 || slices.Contains(modalities, events.ModalityAudio) {
		return []events.Modality{events.ModalityAudio}
	}
	return []events.Modality{events.ModalityText}
}

// ResetConversation starts the conversation over: the session is closed and
// opened again, and seeded afresh from the conversation on the next response.
// It must not be called from the read loop.
func (s *Service) ResetConversation(ctx context.Context) error {
	slog.DebugContext(ctx, "resetting the realtime conversation", "service", s.Name())
	s.disconnect()
	s.mu.Lock()
	s.needsConversationSetup = true
	convo := s.convo
	s.mu.Unlock()
	if convo != nil {
		s.processCompletedCalls(ctx, convo, false)
	}
	return s.connect(ctx)
}

// retrieval is the outcome of retrieving a conversation item.
type retrieval struct {
	item *events.ConversationItem
	err  error
}

// errRetrieve reports an item the server could not retrieve.
var errRetrieve = errors.New("openairealtime: retrieving a conversation item")

// RetrieveConversationItem asks the server for a conversation item and waits
// for it. Callers asking for the same item share one request.
func (s *Service) RetrieveConversationItem(ctx context.Context, itemID string) (*events.ConversationItem, error) {
	ch := make(chan retrieval, 1)
	s.mu.Lock()
	inFlight := len(s.retrievals[itemID]) > 0
	s.retrievals[itemID] = append(s.retrievals[itemID], ch)
	s.mu.Unlock()
	if !inFlight {
		ev := events.NewConversationItemRetrieveEvent(itemID)
		// The id names the item, so a failed retrieval can be traced back to it.
		ev.EventID = "rci_" + itemID
		s.sendEvent(ctx, ev)
	}
	select {
	case r := <-ch:
		return r.item, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// resolveRetrieval hands a retrieved item, or the error retrieving it, to the
// callers waiting on it.
func (s *Service) resolveRetrieval(itemID string, r retrieval) {
	s.mu.Lock()
	waiting := s.retrievals[itemID]
	delete(s.retrievals, itemID)
	s.mu.Unlock()
	for _, ch := range waiting {
		ch <- r
	}
}

// sendAudio appends a chunk of input PCM to the model's input buffer. The
// session's input format is PCM at a fixed 24 kHz, so audio at any other rate
// is resampled first.
func (s *Service) sendAudio(ctx context.Context, audio frames.AudioRawData) {
	pcm := s.toSessionRate(audio)
	if len(pcm) == 0 {
		return
	}
	s.sendEvent(ctx, events.NewInputAudioBufferAppendEvent(base64.StdEncoding.EncodeToString(pcm)))
}

// toSessionRate converts input audio to the session's rate.
func (s *Service) toSessionRate(audio frames.AudioRawData) []byte {
	if audio.SampleRate == sampleRate || audio.SampleRate <= 0 {
		return audio.Audio
	}
	channels := max(audio.NumChannels, 1)
	s.inMu.Lock()
	defer s.inMu.Unlock()
	if s.inRS == nil || s.inRate != audio.SampleRate || s.inChannels != channels {
		if s.inRS != nil {
			s.inRS.Close()
		}
		rs, err := resample.New(audio.SampleRate, sampleRate, channels)
		if err != nil {
			s.inRS = nil
			return nil
		}
		s.inRS, s.inRate, s.inChannels = rs, audio.SampleRate, channels
	}
	return s.inRS.Process(audio.Audio)
}

// closeResampler releases the input resampler.
func (s *Service) closeResampler() {
	s.inMu.Lock()
	defer s.inMu.Unlock()
	if s.inRS != nil {
		s.inRS.Close()
		s.inRS = nil
	}
}

// send marshals v and writes it as a text frame, serializing concurrent writes.
func (s *Service) send(v any) error {
	if s.write != nil {
		return s.write(v)
	}
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

// sendEvent sends a client event. A send failing while the session is open is
// rare between servers and hard to recover from cleanly, so it is reported as
// an error rather than retried; one failing because the session is closed is
// expected and ignored.
func (s *Service) sendEvent(ctx context.Context, ev any) {
	err := s.send(ev)
	if err == nil || errors.Is(err, errNotConnected) {
		return
	}
	s.mu.Lock()
	open := s.conn != nil
	s.mu.Unlock()
	if open {
		s.PushError(ctx, "openai realtime: error sending client event", err, false)
	}
}

// disconnect cancels the session context, closes the socket, and waits for the
// read loop to exit. It is safe to call more than once.
func (s *Service) disconnect() {
	s.StopAllMetrics(context.Background())
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.audioResp = nil
	// A new session knows none of the old call ids, and no response is running
	// in it.
	s.openCalls = map[string]bool{}
	s.responseActive = false
	s.responseCreateOwed = false
	s.sessionReady = false
	s.sentResults = map[string]bool{}
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
	s.wg.Wait()
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func tokenUsage(u *events.Usage) frames.LLMTokenUsage {
	out := frames.LLMTokenUsage{
		PromptTokens:     valueOr(u.InputTokens),
		CompletionTokens: valueOr(u.OutputTokens),
		TotalTokens:      valueOr(u.TotalTokens),
	}
	if d := u.InputTokenDetails; d != nil {
		out.CacheReadTokens = d.CachedTokens
		out.InputAudioTokens = d.AudioTokens
		out.InputTextTokens = d.TextTokens
		if c := d.CachedTokensDetails; c != nil {
			out.CacheReadAudioTokens = c.AudioTokens
		}
	}
	if d := u.OutputTokenDetails; d != nil {
		out.OutputAudioTokens = d.AudioTokens
		out.OutputTextTokens = d.TextTokens
	}
	return out
}

// valueOr is the count p points to, or zero when the server reported none.
func valueOr(p *int64) int64 {
	if p == nil {
		return 0
	}
	return *p
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
		ev, err := events.ParseServerEvent(data)
		if err != nil {
			s.PushError(connCtx, "openai realtime: reading a server event", err, false)
			return
		}
		if ev == nil {
			continue
		}
		if stop := s.handleEvent(connCtx, ev); stop {
			return
		}
	}
}

// handleEvent maps a server event onto pipeline frames, reporting whether the
// session has failed in a way that ends the read loop.
func (s *Service) handleEvent(ctx context.Context, ev *events.ServerEvent) bool {
	switch {
	case s.handleTurnEvent(ctx, ev):
		return false
	case s.handleResponseEvent(ctx, ev):
		return false
	case s.handleTranscriptEvent(ctx, ev):
		return false
	case ev.Type == "session.created":
		// The session exists, so it is configured now.
		s.traceSetup(ctx, s.sessionProperties())
		s.sendSessionUpdate(ctx)
	case ev.Type == "session.updated":
		s.mu.Lock()
		s.sessionReady = true
		run := s.runWhenReady
		s.runWhenReady = false
		s.mu.Unlock()
		if run {
			s.createResponse(ctx)
		}
	case ev.Type == "conversation.item.retrieved":
		if ev.Item != nil {
			s.resolveRetrieval(ev.Item.ID, retrieval{item: ev.Item})
		}
	case ev.Type == "conversation.item.added":
		s.handleItemAdded(ctx, ev.Item)
	case ev.Type == "conversation.item.done":
		if ev.Item != nil {
			s.Events().Call(ctx, EventConversationItemUpdated, s, ConversationItemEvent{ItemID: ev.Item.ID, Item: ev.Item})
		}
	case ev.Type == "response.function_call_arguments.done":
		s.runToolCall(ctx, ev)
	case ev.Type == "error":
		return s.handleServerError(ctx, ev)
	}
	return false
}

// handleTurnEvent takes the boundaries the session's own VAD reports, saying
// whether it took the event.
//
// A boundary is proposed rather than announced: the external turn strategies the
// service recommends decide the turn from it and broadcast the interruption,
// which comes back here as the frame that truncates what the model believes it
// said.
func (s *Service) handleTurnEvent(ctx context.Context, ev *events.ServerEvent) bool {
	switch ev.Type {
	case "input_audio_buffer.speech_started":
		// The model has stopped speaking as far as the caller is concerned, so
		// its record of the turn is cut back to what was heard, and a response
		// owed behind the one it was giving is moot.
		s.mu.Lock()
		s.responseCreateOwed = false
		s.mu.Unlock()
		s.truncateAudioResponse(ctx)
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewProposedUserStartedSpeakingFrame() })
	case "input_audio_buffer.speech_stopped":
		s.StartTTFBMetrics()
		s.StartProcessingMetrics()
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() })
	default:
		return false
	}
	return true
}

// handleResponseEvent takes the events that make up the model's answer: its
// audio, the transcript of that audio, the same answer in text from a session
// answering in that modality, and the end of the response. It reports whether it
// took the event.
func (s *Service) handleResponseEvent(ctx context.Context, ev *events.ServerEvent) bool {
	switch ev.Type {
	case "response.created":
		// The response takes in everything delivered before it.
		s.mu.Lock()
		s.responseActive = true
		s.resultsAwaitingResponse = false
		s.mu.Unlock()
	case "response.output_audio.delta":
		s.handleAudioDelta(ctx, ev)
	case "response.output_audio_transcript.delta":
		// The transcript of the spoken answer: kept out of the conversation as
		// LLM text, which RTVI reports, and recorded by the assistant aggregator
		// as the words spoken.
		if ev.Delta != "" {
			llmText := frames.NewLLMTextFrame(ev.Delta)
			llmText.AppendToContext = false
			_ = s.PushFrame(ctx, llmText, processor.Downstream)
			ttsText := frames.NewTTSTextFrame(ev.Delta, frames.TextTypeSentence)
			ttsText.IncludesInterFrameSpaces = true
			_ = s.PushFrame(ctx, ttsText, processor.Downstream)
		}
	case "response.output_text.delta":
		// The answer itself, in the text modality.
		if ev.Delta != "" {
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(ev.Delta), processor.Downstream)
		}
	case "response.done":
		s.handleResponseDone(ctx, ev)
	default:
		return false
	}
	return true
}

// handleAudioDelta plays a chunk of the model's audio. The first chunk of a
// response opens the spoken answer; a chunk of the next output item of the same
// response continues it, the item being tracked anew so an interruption cuts
// back the item the caller was hearing.
func (s *Service) handleAudioDelta(ctx context.Context, ev *events.ServerEvent) {
	s.ReportTTFBMetrics(ctx)
	s.mu.Lock()
	started := false
	switch {
	case s.audioResp == nil:
		s.audioResp = &audioResponse{
			itemID: ev.ItemID, contentIndex: ev.ContentIndex, responseID: ev.ResponseID, started: time.Now(),
		}
		started = true
	case s.audioResp.itemID != ev.ItemID:
		slog.DebugContext(ctx, "advancing to the next audio output item", "service", s.Name(),
			"prev_item_id", s.audioResp.itemID, "item_id", ev.ItemID, "response_id", ev.ResponseID)
		*s.audioResp = audioResponse{
			itemID: ev.ItemID, contentIndex: ev.ContentIndex, responseID: ev.ResponseID, started: time.Now(),
		}
	}
	s.mu.Unlock()
	if started {
		_ = s.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream)
	}
	pcm, err := base64.StdEncoding.DecodeString(ev.Delta)
	if err != nil {
		return
	}
	s.mu.Lock()
	if s.audioResp != nil {
		s.audioResp.total += len(pcm)
	}
	s.mu.Unlock()
	_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, sampleRate, 1), processor.Downstream)
}

// handleResponseDone closes the response: its usage is reported, a spoken
// answer is closed with a TTS stop and the response with its end, a response
// owed behind it is created, and one that failed is reported as an error.
func (s *Service) handleResponseDone(ctx context.Context, ev *events.ServerEvent) {
	s.reportUsage(ctx, ev)
	s.StopProcessingMetrics(ctx)
	s.mu.Lock()
	spoken := s.audioResp != nil
	s.assistantResponding = false
	s.responseActive = false
	// A response asked for while this one was running is owed now, unless this
	// one took in what it was for.
	owed := s.responseCreateOwed && s.resultsAwaitingResponse
	s.responseCreateOwed = false
	s.mu.Unlock()
	// Closed once for the whole response, however many output items it held.
	if spoken {
		_ = s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
	}
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	if owed {
		s.createResponse(ctx)
	}
	if ev.Response == nil {
		return
	}
	if ev.Response.Status == "failed" {
		message := "Response failed"
		if e, ok := ev.Response.StatusDetails["error"].(map[string]any); ok {
			if m, ok := e["message"].(string); ok && m != "" {
				message = m
			}
		}
		s.PushError(ctx, message, fmt.Errorf("%w: %s", errServer, message), false)
		return
	}
	for i := range ev.Response.Output {
		item := &ev.Response.Output[i]
		s.Events().Call(ctx, EventConversationItemUpdated, s, ConversationItemEvent{ItemID: item.ID, Item: item})
	}
}

// handleInterruption puts right what the interruption makes moot: the
// session's record of the response being played is cut back to what the caller
// heard, the response's timing ends, a response owed behind it is dropped (the
// user's turn makes one of its own), and a response in progress is closed.
func (s *Service) handleInterruption(ctx context.Context) {
	s.truncateAudioResponse(ctx)
	s.StopAllMetrics(ctx)
	s.mu.Lock()
	s.responseCreateOwed = false
	responding := s.assistantResponding
	s.mu.Unlock()
	if responding {
		_ = s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
		if slices.Contains(s.outputModalities(), events.ModalityAudio) {
			_ = s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
		}
	}
}

// outputModalities are what the session responds with: audio and text unless
// the session properties say otherwise.
func (s *Service) outputModalities() []events.Modality {
	if m := s.sessionProperties().OutputModalities; len(m) > 0 {
		return m
	}
	return []events.Modality{events.ModalityAudio, events.ModalityText}
}

// handleTranscriptEvent takes what the session heard the user say, reporting
// whether it took the event.
//
// A transcript travels upstream, where the user aggregator is: it is the user's
// message, and everything downstream of this service is the reply to it.
func (s *Service) handleTranscriptEvent(ctx context.Context, ev *events.ServerEvent) bool {
	switch ev.Type {
	case "conversation.item.input_audio_transcription.delta":
		_ = s.PushFrame(ctx, frames.NewInterimTranscriptionFrame(ev.Delta, "", frames.NowTimestamp()),
			processor.Upstream)
	case "conversation.item.input_audio_transcription.completed":
		s.Events().Call(ctx, EventConversationItemUpdated, s, ConversationItemEvent{ItemID: ev.ItemID})
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(ev.Transcript, "", frames.NowTimestamp()),
			processor.Upstream)
	default:
		return false
	}
	return true
}

// handleServerError reports an error event, saying whether it ends the session.
//
// Canceling a response the session had already finished is a race the
// interruption path runs into by design: it says the model had stopped, which is
// what was wanted. So is asking for a response while one is running. Both are
// logged and the session carries on; anything else is reported and ends the read
// loop, since the session cannot be relied on afterwards.
func (s *Service) handleServerError(ctx context.Context, ev *events.ServerEvent) bool {
	if ev.Error == nil {
		return false
	}
	if ev.Error.Code == "item_retrieve_invalid_item_id" {
		// A retrieval failed: the event id names the item.
		_, itemID, _ := strings.Cut(ev.Error.EventID, "_")
		s.resolveRetrieval(itemID, retrieval{err: fmt.Errorf("%w: %s", errRetrieve, ev.Error.Message)})
		return false
	}
	switch ev.Error.Code {
	case "response_cancel_not_active", "conversation_already_has_active_response":
		slog.DebugContext(ctx, "the realtime session reported a benign error",
			"service", s.Name(), "code", ev.Error.Code, "message", ev.Error.Message)
		return false
	}
	s.PushError(ctx, "openai realtime error: "+ev.Error.Message,
		fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	return true
}

// truncateAudioResponse cuts the item being played back to what the caller
// actually heard, so the conversation the model keeps matches it. Without this
// the model believes it said a whole answer the user cut off after a word.
//
// The cut is the shorter of the time since playback started and the audio that
// arrived, since either can be the smaller: a response can arrive faster than it
// plays, and playback can outlast a response that stopped arriving.
func (s *Service) truncateAudioResponse(ctx context.Context) {
	s.mu.Lock()
	resp := s.audioResp
	s.audioResp = nil
	s.mu.Unlock()
	if resp == nil {
		return
	}
	// 16-bit mono PCM, so two bytes to the sample.
	audioMS := int64(resp.total) * 1000 / int64(2*sampleRate)
	elapsedMS := time.Since(resp.started).Milliseconds()
	endMS := max(0, min(elapsedMS, audioMS))
	slog.DebugContext(ctx, "truncating the interrupted response", "service", s.Name(),
		"item_id", resp.itemID, "audio_ms", audioMS, "elapsed_ms", elapsedMS, "end_ms", endMS)
	s.sendEvent(ctx, events.NewConversationItemTruncateEvent(resp.itemID, resp.contentIndex, int(endMS)))
}

// handleItemAdded takes an item added to the session's conversation: a tool
// call the model announced is tracked, and the echo of an item this service put
// in is recognized as such.
func (s *Service) handleItemAdded(ctx context.Context, item *events.ConversationItem) {
	if item == nil {
		return
	}
	s.trackToolCall(item)
	s.Events().Call(ctx, EventConversationItemCreated, s, ConversationItemEvent{ItemID: item.ID, Item: item})
	s.mu.Lock()
	manual := s.addedManually[item.ID]
	delete(s.addedManually, item.ID)
	assistant := !manual && item.Role == roleAssistant
	if assistant {
		s.assistantResponding = true
	}
	s.mu.Unlock()
	if assistant {
		_ = s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	}
}

// trackToolCall records a tool call the model has announced. The call is named
// here and its arguments finish arriving later, so the name has to be kept until
// they do.
func (s *Service) trackToolCall(item *events.ConversationItem) {
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
func (s *Service) runToolCall(ctx context.Context, ev *events.ServerEvent) {
	s.mu.Lock()
	name, tracked := s.pendingCalls[ev.CallID]
	// Taken out first, so a repeated event cannot run the same call twice.
	delete(s.pendingCalls, ev.CallID)
	if tracked {
		// The session is waiting on this call now, so its id can be answered
		// with a function_call_output.
		s.openCalls[ev.CallID] = true
	}
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
func (s *Service) reportUsage(ctx context.Context, ev *events.ServerEvent) {
	spanCtx, end := s.traceResponse(ctx, ev.Response)
	defer end()
	if ev.Response == nil || ev.Response.Usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	_ = s.PushTokenUsage(spanCtx, tokenUsage(ev.Response.Usage))
}

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// ServiceMetadataFrame describes this service to the pipeline as a realtime one.
// The user aggregator reads that and stops holding turns open for transcripts,
// which this service is not answering from, and writes the user's message when
// the answer starts rather than when the turn ends, because the transcript for
// what the user said arrives late.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	// The session detects the turns and proposes each boundary; the strategies
	// resolve a proposal into the turn frames and the interruption. Without them
	// a proposal reaches nothing and no turn is ever opened or closed.
	f.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	return f
}

// The events the service raises about the session's conversation items, each
// with a ConversationItemEvent.
//
//	events.On(svc.Events(), realtime.EventConversationItemUpdated,
//	    func(ctx context.Context, e realtime.ConversationItemEvent) { … })
const (
	// EventConversationItemCreated fires when an item is added to the
	// conversation, whoever added it.
	EventConversationItemCreated = "on_conversation_item_created"
	// EventConversationItemUpdated fires when an item is complete, and when the
	// transcription of the user's audio item is (with no item).
	EventConversationItemUpdated = "on_conversation_item_updated"
)

// ConversationItemEvent names a conversation item, and carries it when known.
type ConversationItemEvent struct {
	ItemID string
	Item   *events.ConversationItem
}
