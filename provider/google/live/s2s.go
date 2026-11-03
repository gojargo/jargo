package live

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/adapter"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/adapter/geminilive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/wsutil"
	textutil "github.com/gojargo/jargo/utils/text"
	"github.com/google/uuid"
)

// Service is the Gemini Live speech-to-speech processor.
type Service struct {
	// adapter converts the conversation into what the session takes: the
	// system instruction, the history, and the tools it carries.
	adapter geminilive.Adapter
	*llm.Base
	cfg Config
	// live is what the service holds of its configuration while the pipeline
	// runs, which the setup message is built from.
	live *Settings

	mu        sync.Mutex
	conn      *wsutil.Conn
	connCtx   context.Context
	cancel    context.CancelFunc
	writeMu   sync.Mutex
	wg        sync.WaitGroup
	ready     atomic.Bool
	connector Connector

	// convo is the conversation this session is part of, as the pipeline last
	// reported it. The model generates continuously and so never re-reads it,
	// but a tool call is answered into it and its results are read back out of
	// it here. Guarded by mu.
	convo *frames.LLMContext
	// toolNames maps a call id to the function it called. A result has to name
	// the function as well as the call, and by the time it comes back only the
	// id is at hand. Guarded by mu.
	toolNames map[string]string
	// sentResults names the tool calls whose result has already gone to the
	// model, so a conversation reported again does not send one twice. Guarded
	// by mu.
	sentResults map[string]bool
	// syncToolWarning keeps the report that a synchronous tool cannot block on
	// this model to one, however many tools the session advertises.
	syncToolWarning sync.Once

	// turnHeld records a completed generation held back because the model
	// reported that it was still working. Guarded by mu.
	turnHeld bool
	// heldTurnWatchdog ends a held turn if the session goes quiet without ever
	// reporting that it has finished. Guarded by mu.
	heldTurnWatchdog *time.Timer
	// contentAwaitingResponse says content was appended to the session that no
	// turn has answered yet; the aggregator's next upstream conversation
	// completes the turn. Guarded by mu.
	contentAwaitingResponse bool

	// readyForInput says the session has been seeded with the conversation, or
	// needed no seeding, and takes realtime input. Audio before it is dropped.
	readyForInput atomic.Bool
	// runWhenReady says the conversation arrived before the session was ready,
	// so seeding it waits for the session. Guarded by mu.
	runWhenReady bool
	// settlingFirstContext says the first conversation is being taken in, which
	// seeds the session itself once it is done. A session becoming ready
	// meanwhile leaves the seeding to it. Guarded by mu.
	settlingFirstContext bool
	// needsInitialTurnComplete says a 2.5 session was seeded without being asked
	// to answer, so the user's first turn has to complete one: otherwise the
	// model ignores what it was seeded with. Guarded by mu.
	needsInitialTurnComplete bool

	// botResponding says the model is answering: its response has been opened
	// downstream and not yet closed. Guarded by mu.
	botResponding bool
	// botText is the text the model has answered with so far, in the text
	// modality; a response with none was spoken. Guarded by mu.
	botText string
	// userTranscript holds the user's words until they make up a sentence, or
	// go quiet for transcriptFlushAfter. Guarded by mu.
	userTranscript string
	// transcriptFlush flushes userTranscript once the words stop. Guarded by mu.
	transcriptFlush *time.Timer
	// warnedMimeType keeps the report of audio without a sample rate to one.
	warnedMimeType sync.Once

	// vadDisabled says server-side voice activity detection is off, so the
	// pipeline's own turn detection drives the session: the user's turns are
	// marked with activity signals, and audio is sent only inside one.
	vadDisabled bool
	// audioPaused stops user audio reaching the session.
	audioPaused atomic.Bool
	// userIsSpeaking says the user's turn is in progress. Guarded by mu.
	userIsSpeaking bool
	// preroll holds the most recent user audio outside a turn, while server-side
	// detection is off, to replay once a turn starts, so the onset that started
	// it is not clipped. Guarded by mu.
	preroll     []byte
	prerollRate int
	// prerollSecs is how much audio preroll holds, and prerollPinned says
	// Config.UserAudioPrerollSecs set it, so the pipeline's detection does not
	// resize it. Guarded by mu.
	prerollSecs   float64
	prerollPinned bool

	// connMu serializes opening and closing the session, which the frame
	// goroutine and a reconnection after a failed read can both do.
	connMu sync.Mutex
	// stopped says the pipeline is ending, so a failed session is not opened
	// again.
	stopped atomic.Bool
	// resumptionHandle is the latest handle the server gave for resuming the
	// session, which a session opened again resumes from. Guarded by mu.
	resumptionHandle string
	// consecutiveFailures counts the sessions lost in a row; connectedAt is when
	// the current one opened, and a session that stays up long enough resets
	// the count. Guarded by mu.
	consecutiveFailures int
	connectedAt         time.Time
}

// A session lost maxConsecutiveFailures times in a row is not opened again; one
// that stays up for stableConnection resets the count.
const (
	maxConsecutiveFailures = 3
	stableConnection       = 10 * time.Second
)

// The pre-roll windows, in seconds: the margin added to the start delay of the
// pipeline's voice activity detection, and the window used until that is known.
const (
	prerollMarginSecs  = 0.1
	defaultPrerollSecs = 0.5
)

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

// HistoryConfigGate is an optional interface a Connector implements when the
// deployment it dials does not take the history configuration in the setup
// message. The service then leaves it out.
type HistoryConfigGate interface {
	SupportsHistoryConfig() bool
}

// NonBlockingToolsGate is an optional interface a Connector implements when the
// deployment it dials does not follow the model on non-blocking tool calls. The
// service then asks the connector, rather than the model, whether to declare
// asynchronous tools NON_BLOCKING, attach scheduling hints to their results, and
// take intermediate results.
type NonBlockingToolsGate interface {
	SupportsNonBlockingTools(model string) bool
}

// apiKeyConnector is the standard Gemini Live addressing: the api key travels in
// the x-goog-api-key header and the model is named relative to the API.
type apiKeyConnector struct {
	baseURL string
	apiKey  string
}

func (c apiKeyConnector) Endpoint(context.Context) (string, http.Header, error) {
	return c.baseURL, http.Header{"X-Goog-Api-Key": {c.apiKey}}, nil
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
	warnTextModality(cfg.Modalities)
	prerollSecs := defaultPrerollSecs
	if cfg.UserAudioPrerollSecs != nil {
		prerollSecs = *cfg.UserAudioPrerollSecs
	}
	s := &Service{
		vadDisabled:   cfg.VAD != nil && cfg.VAD.Disabled != nil && *cfg.VAD.Disabled,
		prerollSecs:   prerollSecs,
		prerollPinned: cfg.UserAudioPrerollSecs != nil,
		cfg:           cfg,
		live:          newSettings(cfg),
		connector:     conn,
		toolNames:     map[string]string{},
		sentResults:   map[string]bool{},
	}
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New(name, s, llm.WithContinuousGeneration())
	s.SetModel(cfg.Model)
	s.audioPaused.Store(cfg.StartAudioPaused)
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
	if in, ok := f.(frames.InputAudioFrame); ok && dir == processor.Downstream {
		audio := in.AudioData()
		// The model consumes the audio, so it does not travel on. Only the
		// processor bookkeeping runs: the LLM base would forward it.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sendAudio(ctx, audio.Audio, audio.SampleRate, audio.NumChannels)
		return nil
	}
	if _, ok := f.(*frames.InterruptionFrame); ok {
		// Before the interruption travels on, so whatever closes the response
		// in progress reaches the processors after this one ahead of it.
		s.handleInterruption(ctx)
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
			s.handleContext(ctx, fr.Context, dir)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMMessagesAppendFrame:
		s.handleMessagesAppend(ctx, fr)
		return nil
	case *frames.EndFrame, *frames.CancelFrame:
		s.stop()
		return nil
	default:
		s.handleUserInput(ctx, f)
		return nil
	}
}

// handleUserInput takes what the pipeline reports about the user: their turn
// starting and stopping, how its own voice activity detection is tuned, and
// text they typed.
func (s *Service) handleUserInput(ctx context.Context, f frames.Frame) {
	switch fr := f.(type) {
	case *frames.UserStartedSpeakingFrame:
		s.handleUserStartedSpeaking(ctx)
	case *frames.UserStoppedSpeakingFrame:
		s.handleUserStoppedSpeaking(ctx)
	case *frames.SpeechControlParamsFrame:
		s.handleSpeechControlParams(fr)
	case *frames.InputTextRawFrame:
		s.sendUserText(ctx, fr.Text)
	}
}

// Cleanup tears down the session and stops the read loop.
func (s *Service) Cleanup(ctx context.Context) error {
	s.stop()
	return s.Base.Cleanup(ctx)
}

// stop closes the session for good: a session lost from now on is not opened
// again.
func (s *Service) stop() {
	s.stopped.Store(true)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	s.disconnect()
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
	setup, err := s.setup()
	if err != nil {
		cancel()
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return err
	}
	if err := s.send(setup); err != nil {
		cancel()
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return err
	}

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// generationConfig is the part of the setup the model generates by: what it
// responds with, its voice and language, and the sampling and reasoning
// settings.
func (s *Service) generationConfig() map[string]any {
	live := s.live
	gen := map[string]any{
		"responseModalities": []string{string(live.Modalities.Or(ModalityAudio))},
		"speechConfig": map[string]any{
			"voiceConfig": map[string]any{
				"prebuiltVoiceConfig": map[string]any{"voiceName": live.Voice.Or("")},
			},
			"languageCode": string(live.Language.Or(defaultLanguage)),
		},
		"mediaResolution": string(live.MediaResolution.Or(MediaResolutionUnspecified)),
	}
	// The sampling settings go out only when set, leaving the model's own.
	setGiven(gen, "frequencyPenalty", live.FrequencyPenalty)
	setGiven(gen, "maxOutputTokens", live.MaxTokens)
	setGiven(gen, "presencePenalty", live.PresencePenalty)
	setGiven(gen, "seed", live.Seed)
	setGiven(gen, "temperature", live.Temperature)
	setGiven(gen, "topK", live.TopK)
	setGiven(gen, "topP", live.TopP)
	if thinking := gemini.ThinkingParams(resolvedThinking(s.model(), live.Thinking.Or(nil))); thinking != nil {
		gen["thinkingConfig"] = thinking
	}
	if live.EnableAffectiveDialog.Or(false) {
		gen["enableAffectiveDialog"] = true
	}
	return gen
}

// setup is the initial session-configuration message, built from the settings
// the service holds.
func (s *Service) setup() (map[string]any, error) {
	live := s.live
	gen := s.generationConfig()

	// Resumption is always on, so the server hands out handles; a session
	// opened again resumes from the last one.
	resumption := map[string]any{}
	s.mu.Lock()
	if s.resumptionHandle != "" {
		resumption["handle"] = s.resumptionHandle
	}
	s.mu.Unlock()
	setup := map[string]any{
		"model":                    s.connector.ModelPath(s.model()),
		"generationConfig":         gen,
		"inputAudioTranscription":  map[string]any{},
		"outputAudioTranscription": map[string]any{},
		"sessionResumption":        resumption,
	}
	// The conversation the session is seeded with goes in as client content,
	// which the API has to be told to expect.
	if gate, ok := s.connector.(HistoryConfigGate); !ok || gate.SupportsHistoryConfig() {
		setup["historyConfig"] = map[string]any{"initialHistoryInClientContent": true}
	}
	if cwc := live.ContextWindowCompression.Or(nil); cwc != nil && cwc.Enabled {
		// Compression always slides the window.
		compression := map[string]any{"slidingWindow": map[string]any{}}
		if cwc.TriggerTokens != nil {
			compression["triggerTokens"] = *cwc.TriggerTokens
		}
		setup["contextWindowCompression"] = compression
	}
	if p := live.Proactivity.Or(nil); p != nil {
		proactivity := map[string]any{}
		if p.ProactiveAudio != nil {
			proactivity["proactiveAudio"] = *p.ProactiveAudio
		}
		setup["proactivity"] = proactivity
	}
	if input := realtimeInputConfig(live.VAD.Or(nil), live.TurnCoverage.Or("")); input != nil {
		setup["realtimeInputConfig"] = input
	}

	// The system instruction and the tools come from the conversation when
	// there is one, which the adapter settles against the configured ones: the
	// instruction configured on the service wins over the conversation's own
	// prompt, and the conversation's tools win over the configured ones. The
	// built-in tools ride along either way.
	system, tools, err := s.sessionInstructionAndTools()
	if err != nil {
		return nil, err
	}
	if system != "" {
		setup["systemInstruction"] = map[string]any{
			keyParts: []any{map[string]any{keyText: system}},
		}
	}
	if len(tools) > 0 {
		if s.supportsNonBlockingTools() {
			s.tagToolBehaviors(tools)
		}
		setup["tools"] = tools
	}
	return map[string]any{"setup": setup}, nil
}

// setGiven sets key to the value o holds, if it holds one.
func setGiven[T any](m map[string]any, key string, o settings.Opt[T]) {
	if v, ok := o.Value(); ok {
		m[key] = v
	}
}

// realtimeInputConfig is how the server detects activity and what a turn
// covers, or nil when nothing about either is set. Only the detection
// parameters that are set are sent.
func realtimeInputConfig(vad *VADParams, coverage TurnCoverage) map[string]any {
	config := map[string]any{}
	if detection := activityDetection(vad); len(detection) > 0 {
		config["automaticActivityDetection"] = detection
	}
	if coverage != "" {
		config["turnCoverage"] = string(coverage)
	}
	if len(config) == 0 {
		return nil
	}
	return config
}

// activityDetection is the server's activity detection as vad sets it: only the
// parameters that are set, so the server's own default holds for the rest.
func activityDetection(vad *VADParams) map[string]any {
	if vad == nil {
		return nil
	}
	detection := map[string]any{}
	if vad.Disabled != nil {
		detection["disabled"] = *vad.Disabled
	}
	if vad.StartSensitivity != "" {
		detection["startOfSpeechSensitivity"] = string(vad.StartSensitivity)
	}
	if vad.EndSensitivity != "" {
		detection["endOfSpeechSensitivity"] = string(vad.EndSensitivity)
	}
	if vad.PrefixPaddingMs != nil {
		detection["prefixPaddingMs"] = *vad.PrefixPaddingMs
	}
	if vad.SilenceDurationMs != nil {
		detection["silenceDurationMs"] = *vad.SilenceDurationMs
	}
	return detection
}

// sessionInstructionAndTools settles the system instruction and the tools a
// session opens with. Without a conversation yet, they are the configured ones,
// with the built-in tools added.
func (s *Service) sessionInstructionAndTools() (string, []map[string]any, error) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		return s.cfg.Instructions, s.adapter.ToProviderToolsFormat(s.adapter.WithBuiltins(s.cfg.Tools)), nil
	}
	params, err := s.sessionParams(convo)
	if err != nil {
		return "", nil, err
	}
	return params.SystemInstruction, params.Tools, nil
}

// sessionParams converts the conversation against what the service was
// configured with.
func (s *Service) sessionParams(convo *frames.LLMContext) (geminiadapter.Params, error) {
	return s.adapter.LLMInvocationParams(convo, adapter.Options{
		SystemInstruction: s.cfg.Instructions,
		ServiceTools:      s.cfg.Tools,
	})
}

// ServiceTools returns the tools configured on the service, which the base
// registers the handlers of while the conversation advertises no tools.
func (s *Service) ServiceTools() frames.ToolsSchema { return s.cfg.Tools }

// LLMAdapter returns the adapter this service converts through, so the base can
// give it the tools the service implements itself.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// handleContext takes the conversation the pipeline reported.
//
// The first one settles the session. The session was opened with the
// configured system instruction and tools, and the Live API takes both in the
// setup message and nowhere else, so a conversation that changes the effective
// instruction, or brings tools of its own, is applied by opening the session
// again with it.
//
// A later one is the conversation moving on. The model keeps its own history,
// so what it brings is the results of the tool calls the model asked for, and,
// when it comes from upstream, the aggregator asking the model to answer content
// appended since it last spoke.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext, dir processor.Direction) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	s.mu.Unlock()

	if !first {
		s.mu.Lock()
		complete := dir == processor.Upstream && s.contentAwaitingResponse
		s.mu.Unlock()
		if complete {
			s.completeTurn(ctx)
		}
		s.processCompletedCalls(ctx, convo, true)
		return
	}

	s.mu.Lock()
	s.settlingFirstContext = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.settlingFirstContext = false
		s.mu.Unlock()
	}()

	// Whether the conversation changes the effective instruction is the
	// adapter's to settle, and it still prefers the configured one when there is
	// one.
	params, err := s.sessionParams(convo)
	if err != nil {
		s.PushError(ctx, "gemini live: converting the conversation", err, false)
		return
	}
	instructionChanged := params.SystemInstruction != s.cfg.Instructions
	// Tools of its own warrant reopening without being compared with the
	// configured ones.
	hasTools := !convo.ToolsSchema().Empty()
	if hasTools && !s.cfg.Tools.Empty() {
		slog.WarnContext(ctx, "tools were configured on the service and the conversation brings its own; "+
			"using the conversation's", "service", s.Name())
	}
	if instructionChanged || hasTools {
		s.reconnect(ctx)
	}

	// The results already in the first conversation were produced before this
	// session existed, so they are recorded as known rather than sent: the model
	// never asked for them and has nothing to do with them.
	s.processCompletedCalls(ctx, convo, false)

	// A conversation with nothing in it gives the model nothing to answer, so
	// the configured instruction is put in for it to answer, when it is to
	// answer at all.
	if len(params.Contents) == 0 && s.inferenceOnContextInitialization() {
		if s.cfg.Instructions != "" {
			slog.DebugContext(ctx, "no messages in the initial conversation; seeding it with the "+
				"system instruction to have the bot answer", "service", s.Name())
			convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: s.cfg.Instructions})
		} else {
			slog.WarnContext(ctx, "no messages in the initial conversation and no system instruction; "+
				"the bot cannot be made to speak first", "service", s.Name())
		}
	}
	s.createInitialResponse(ctx, false)
}

// inferenceOnContextInitialization says whether the model answers the
// conversation the session is seeded with.
func (s *Service) inferenceOnContextInitialization() bool {
	return s.cfg.InferenceOnContextInitialization == nil || *s.cfg.InferenceOnContextInitialization
}

// handleSessionReady takes the server's acknowledgement of a session's setup.
// A conversation that arrived first is seeded now. A session resumed from a
// handle is restored by the server and takes input at once. On a session opened
// again without one after the conversation was taken in, the conversation is
// seeded again, so the new session has what the old one had before it takes
// input. Otherwise the conversation has yet to arrive, and seeding waits for it.
func (s *Service) handleSessionReady(ctx context.Context) {
	s.mu.Lock()
	s.ready.Store(true)
	s.connectedAt = time.Now()
	run := s.runWhenReady
	s.runWhenReady = false
	resumed := !run && s.resumptionHandle != ""
	reseed := !run && !resumed && s.convo != nil && !s.settlingFirstContext
	s.mu.Unlock()
	switch {
	case run:
		s.createInitialResponse(ctx, false)
	case resumed:
		// The server restores the session it resumes, so it takes input at once.
		s.readyForInput.Store(true)
	case reseed:
		s.createInitialResponse(ctx, true)
	}
}

// createInitialResponse seeds the session with the conversation and, when it
// is to answer, asks the model to.
//
// On the session the conversation first arrived on, the model answers when
// InferenceOnContextInitialization says so. A 2.5 model seeded without being
// asked to answer ignores what it was seeded with once the user speaks, unless
// the user's first turn completes one, which is arranged for here.
//
// On a session opened again, a 3.x model needs nothing: the user's next turn is
// answered with the history in mind. A 2.5 model is asked to answer at once,
// recapping the conversation, rather than seeming for a turn to have forgotten
// it.
//
// A 2.5 model also requires the seed to end on a user turn, so a blank one is
// added when it does not.
func (s *Service) createInitialResponse(ctx context.Context, forReconnect bool) {
	s.mu.Lock()
	if !s.ready.Load() {
		s.runWhenReady = true
		s.mu.Unlock()
		return
	}
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		return
	}

	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		s.PushError(ctx, "gemini live: converting the conversation", err, false)
		return
	}
	if len(params.Contents) == 0 {
		// Nothing to seed the session with, so it takes input right away.
		s.readyForInput.Store(true)
		return
	}

	gemini3 := isGemini3(s.model())
	trigger := s.inferenceOnContextInitialization()
	if forReconnect {
		trigger = !gemini3
	}

	seed := params.Contents
	if !gemini3 {
		if role, _ := seed[len(seed)-1]["role"].(string); role != roleUser {
			seed = append(seed, map[string]any{"role": roleUser, keyParts: []any{map[string]any{keyText: " "}}})
		}
	}

	slog.DebugContext(ctx, "seeding the live session with the conversation",
		"service", s.Name(), "turns", len(seed), "answer", trigger)
	s.StartTTFBMetrics()
	if err := s.send(map[string]any{
		keyClientContent: map[string]any{keyTurns: seed, keyTurnComplete: trigger},
	}); err != nil {
		s.handleSendError(ctx, err)
	} else if gemini3 && trigger {
		// 3.x completes the turn but runs no inference without a realtime input.
		if err := s.send(map[string]any{keyRealtimeInput: map[string]any{keyText: " "}}); err != nil {
			s.handleSendError(ctx, err)
		}
	}

	if !trigger && !gemini3 {
		s.mu.Lock()
		s.needsInitialTurnComplete = true
		s.mu.Unlock()
	}
	s.readyForInput.Store(true)
}

// handleUserStoppedSpeaking starts the wait for the model's answer, and on a
// 2.5 session seeded without being asked to answer, completes the turn so the
// model answers with what it was seeded with in mind.
func (s *Service) handleUserStoppedSpeaking(ctx context.Context) {
	s.mu.Lock()
	s.userIsSpeaking = false
	s.preroll = nil
	s.mu.Unlock()
	s.StartTTFBMetrics()
	if !s.ready.Load() {
		return
	}
	if s.vadDisabled && s.readyForInput.Load() {
		s.sendRealtimeInput(ctx, map[string]any{"activityEnd": map[string]any{}})
	}
	s.mu.Lock()
	needed := s.needsInitialTurnComplete
	s.needsInitialTurnComplete = false
	s.mu.Unlock()
	if needed {
		if err := s.send(map[string]any{keyClientContent: map[string]any{keyTurnComplete: true}}); err != nil {
			slog.ErrorContext(ctx, "completing the first turn failed", "service", s.Name(), "err", err)
		}
	}
}

// PushFrame pushes a frame, sending function call results to the session on
// the way.
//
// Results are broadcast by the base; the downstream copy is observed here and
// sent as it is produced, so a tool that reports progress reaches the model
// while it is still working.
func (s *Service) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if r, ok := f.(*frames.FunctionCallResultFrame); ok && dir == processor.Downstream {
		s.handleFunctionCallResult(ctx, r)
	}
	return s.Base.PushFrame(ctx, f, dir)
}

// AcceptsIntermediateFunctionCallResults reports whether the model takes a
// call's intermediate results. They ride the tool-response channel as a
// response that keeps the call open, which only a non-blocking call allows, so
// this follows the same model gating as the rest of the asynchronous tool
// support.
func (s *Service) AcceptsIntermediateFunctionCallResults() bool {
	return s.supportsNonBlockingTools()
}

// unnamedToolResult names a result whose call was never seen, as the
// conversation names one.
const unnamedToolResult = "tool_call_result"

// handleFunctionCallResult sends one result to the session, and says whether
// the model should answer it.
func (s *Service) handleFunctionCallResult(ctx context.Context, f *frames.FunctionCallResultFrame) {
	final := f.Properties.Final()
	s.mu.Lock()
	name, known := s.toolNames[f.ToolCallID]
	s.mu.Unlock()
	if !known {
		name = unnamedToolResult
	}
	if !final && !s.AcceptsIntermediateFunctionCallResults() {
		slog.WarnContext(ctx, "the model takes one result per function call, dropping an intermediate result",
			"service", s.Name(), "model", s.model(), "function", f.ToolName)
		return
	}
	spoken := true
	switch {
	case f.Properties != nil && f.Properties.RunLLM != nil:
		spoken = *f.Properties.RunLLM
	case f.RunLLM != nil:
		spoken = *f.RunLLM
	}
	result := f.Result
	if result == "" {
		result = frames.ToolResultCompleted
	}
	s.sendToolResultAs(ctx, f.ToolCallID, name, result, final, spoken)
	if final {
		s.recordResult(f.ToolCallID)
	}
}

// handleMessagesAppend puts appended messages into the session, for the model
// to take in.
//
// They go in as client content that does not complete a turn, so the model
// reads them without answering. Whether it answers is the aggregator's call: an
// append that asks to run comes back to this service as the conversation the
// assistant aggregator pushes upstream, which completes the turn (see
// handleContext). Without a conversation, the append starts one on its own,
// which is how a pipeline with no aggregators kicks one off.
func (s *Service) handleMessagesAppend(ctx context.Context, f *frames.LLMMessagesAppendFrame) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		s.createSingleResponse(ctx, f.Messages)
		return
	}
	if !s.ready.Load() {
		return
	}
	var turns []any
	for _, m := range f.Messages {
		if text := m.StandardText(); text != "" {
			turns = append(turns, map[string]any{"role": roleUser, keyParts: []any{map[string]any{keyText: text}}})
		}
	}
	if len(turns) == 0 {
		return
	}
	if err := s.send(map[string]any{
		keyClientContent: map[string]any{keyTurns: turns, keyTurnComplete: false},
	}); err != nil {
		s.handleSendError(ctx, err)
		return
	}
	s.mu.Lock()
	s.contentAwaitingResponse = true
	s.mu.Unlock()
}

// completeTurn asks the model to answer the content it has been given.
func (s *Service) completeTurn(ctx context.Context) {
	if !s.ready.Load() {
		return
	}
	s.mu.Lock()
	s.contentAwaitingResponse = false
	s.mu.Unlock()
	slog.DebugContext(ctx, "completing the turn for appended content", "service", s.Name())
	s.sendTurnComplete(ctx, map[string]any{keyTurnComplete: true})
}

// createSingleResponse has the model answer messages appended with no
// conversation to append them to, the case of a pipeline with no aggregators
// that starts the conversation this way.
func (s *Service) createSingleResponse(ctx context.Context, messages []frames.Message) {
	if !s.ready.Load() {
		return
	}
	// A throwaway conversation, only to convert the messages the way the
	// session takes them. A leading system message is its prompt, as it would
	// be of any conversation built from a message list.
	system := ""
	if len(messages) > 0 && messages[0].Role == frames.RoleSystem && !messages[0].IsLLMSpecific() {
		system, messages = messages[0].Text, messages[1:]
	}
	convo := frames.NewLLMContext(system)
	convo.SetMessages(messages)
	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{})
	if err != nil || len(params.Contents) == 0 {
		return
	}
	slog.DebugContext(ctx, "creating a response", "service", s.Name())
	s.StartTTFBMetrics()
	s.sendTurnComplete(ctx, map[string]any{keyTurns: params.Contents, keyTurnComplete: true})
}

// sendTurnComplete sends client content that completes the turn. The 3.x
// models want that, but also run no inference without a realtime input, so one
// follows.
func (s *Service) sendTurnComplete(ctx context.Context, content map[string]any) {
	if err := s.send(map[string]any{keyClientContent: content}); err != nil {
		s.handleSendError(ctx, err)
		return
	}
	if isGemini3(s.model()) {
		if err := s.send(map[string]any{keyRealtimeInput: map[string]any{keyText: " "}}); err != nil {
			s.handleSendError(ctx, err)
		}
	}
}

// reconnect opens the session again, so its setup carries what the conversation
// settled, and resumes it from the last handle the server gave.
func (s *Service) reconnect(ctx context.Context) {
	s.connMu.Lock()
	defer s.connMu.Unlock()
	if s.stopped.Load() {
		return
	}
	slog.InfoContext(ctx, "reopening the live session", "service", s.Name())
	s.disconnect()
	if err := s.connect(ctx); err != nil {
		s.PushError(ctx, "gemini live reconnect failed", err, true)
	}
}

// handleConnectionError opens a lost session again, unless sessions have been
// lost maxConsecutiveFailures times in a row, which is reported as an error
// instead.
func (s *Service) handleConnectionError(ctx context.Context, lost *wsutil.Conn, err error) {
	s.mu.Lock()
	s.consecutiveFailures++
	failures := s.consecutiveFailures
	current := s.conn == lost
	s.mu.Unlock()
	if !current || s.stopped.Load() {
		// Already replaced, or the pipeline is ending.
		return
	}
	slog.WarnContext(ctx, "the live session was lost", "service", s.Name(),
		"failure", failures, "max", maxConsecutiveFailures, "err", err)
	if failures >= maxConsecutiveFailures {
		s.PushError(ctx, fmt.Sprintf("max consecutive failures (%d) reached, treating as fatal error",
			maxConsecutiveFailures), err, false)
		return
	}
	slog.InfoContext(ctx, "reconnecting", "service", s.Name(),
		"attempt", failures, "max", maxConsecutiveFailures)
	s.reconnect(ctx)
}

// resetFailuresOnceStable forgets the sessions lost before, once the current
// one has stayed up for stableConnection.
func (s *Service) resetFailuresOnceStable() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.consecutiveFailures > 0 && !s.connectedAt.IsZero() && time.Since(s.connectedAt) >= stableConnection {
		slog.Info("the live session is stable; resetting the failure count",
			"service", s.Name(), "after", stableConnection, "failures", s.consecutiveFailures)
		s.consecutiveFailures = 0
	}
}

// handleSendError reports a failed send. A send failing is rare between
// servers and hard to recover from cleanly, so it is reported as an error
// rather than retried. One failing because the session was closing, or was not
// ready, is expected and ignored.
func (s *Service) handleSendError(ctx context.Context, err error) {
	if s.stopped.Load() || !s.ready.Load() {
		return
	}
	s.PushError(ctx, "gemini live: send error", err, false)
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
		if async, ok := frames.ParseAsyncToolMessage(m); ok {
			// The started marker opens a call the model already knows it is
			// waiting on, and an intermediate result was sent as it was
			// produced, keeping the call open for the result that settles it.
			if async.Kind != frames.AsyncToolFinal || s.resultSeen(async.ToolCallID) {
				continue
			}
			s.recordResult(async.ToolCallID)
			if send {
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
				s.sendToolResult(ctx, r.ID, r.Content)
			}
		}
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

// sendToolResult hands one call's final result to the session, for the model
// to answer. The response names the function as well as the call, so the name
// the call went out under is looked back up here.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	s.mu.Lock()
	name := s.toolNames[toolCallID]
	s.mu.Unlock()
	s.sendToolResultAs(ctx, toolCallID, name, result, true, true)
}

// sendToolResultAs hands one call's result to the session. final says whether
// the result completes the call: an intermediate one keeps it open, which only
// a non-blocking call allows. spoken says whether the model should address the
// result; it is told to stay silent otherwise, so the result is context it can
// draw on if the conversation turns that way.
func (s *Service) sendToolResultAs(ctx context.Context, toolCallID, name, result string, final, spoken bool) {
	slog.DebugContext(ctx, "sending a tool result to the live session",
		"service", s.Name(), "tool_call_id", toolCallID, "function", name)
	response := map[string]any{
		"id":       toolCallID,
		"name":     name,
		"response": geminiadapter.FunctionResponseDict(result),
	}
	// Pair the non-blocking declaration on an asynchronous tool with a hint on
	// its result. WHEN_IDLE has the model finish whatever it is saying before it
	// turns to the result, so a result landing late does not cut it off
	// mid-sentence; SILENT takes the result in without answering it. It means
	// nothing for a call the model waited for, so it is gated the way the
	// declaration was.
	if s.supportsNonBlockingTools() && s.FunctionIsAsync(name) {
		scheduling := "WHEN_IDLE"
		if !spoken {
			scheduling = "SILENT"
		}
		// The guides document scheduling inside the response, the API
		// reference as a field of its own; both are set so whichever the
		// server reads is there. willContinue only exists as a field: true
		// keeps the call open for more responses, and false closes it, which
		// the last response has to say or the call stays open for good.
		response["response"] = withScheduling(response["response"], scheduling)
		response["scheduling"] = scheduling
		response["willContinue"] = !final
	}
	if err := s.send(map[string]any{
		"toolResponse": map[string]any{"functionResponses": []any{response}},
	}); err != nil {
		s.handleSendError(ctx, err)
	}
}

// withScheduling adds the scheduling hint to a function response object.
func withScheduling(response any, scheduling string) any {
	var obj map[string]any
	switch r := response.(type) {
	case map[string]any:
		obj = make(map[string]any, len(r)+1)
		maps.Copy(obj, r)
	case json.RawMessage:
		dec := json.NewDecoder(bytes.NewReader(r))
		dec.UseNumber()
		if dec.Decode(&obj) != nil {
			return response
		}
	default:
		return response
	}
	obj["scheduling"] = scheduling
	return obj
}

// tagToolBehaviors says, on each declaration, whether the model should wait for
// that call.
//
// A tool registered to outlive the reply that asked for it is declared
// non-blocking, so the model is not stalled while it runs. A synchronous tool
// should block, since the model then finishes its turn only once the result has
// landed, which is what keeps it from saying it will go and look something up.
// Where not waiting is the model's own default, a synchronous tool is therefore
// declared blocking explicitly. The thinking models take only non-blocking
// declarations, so there a synchronous tool cannot block, which is said once.
//
// See https://ai.google.dev/gemini-api/docs/live-api/tools#async-function-calling.
func (s *Service) tagToolBehaviors(tools []map[string]any) {
	declareBlocking := toolsDefaultToNonBlocking(s.model()) && supportsBlockingTools(s.model())
	for _, tool := range tools {
		decls, ok := tool["functionDeclarations"].([]map[string]any)
		if !ok {
			continue
		}
		for _, decl := range decls {
			name, _ := decl["name"].(string)
			if name == "" {
				continue
			}
			switch {
			case s.FunctionIsAsync(name):
				decl["behavior"] = "NON_BLOCKING"
			case declareBlocking:
				decl["behavior"] = "BLOCKING"
			case toolsDefaultToNonBlocking(s.model()):
				s.warnSyncToolCannotBlock(name)
			}
		}
	}
}

// warnSyncToolCannotBlock reports, once for the session, that a synchronous tool
// will not pause the conversation on a model that runs every call without
// waiting.
func (s *Service) warnSyncToolCannotBlock(name string) {
	s.syncToolWarning.Do(func() {
		slog.Warn("this model runs every function call without waiting for it, so a synchronous "+
			"tool does not pause the conversation and the model may keep talking before its result arrives",
			"service", s.Name(), "model", s.model(), "function", name)
	})
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

// sendAudio streams the user's audio to the session, once it is seeded and
// unless audio is paused. With server-side detection on, it streams all of it,
// since the detection needs the uninterrupted stream. With it off, it streams
// only inside the user's turn, the session discarding anything outside one, and
// keeps the most recent audio outside a turn as the pre-roll to replay when the
// next one starts.
func (s *Service) sendAudio(ctx context.Context, pcm []byte, sampleRate, channels int) {
	if len(pcm) == 0 || s.audioPaused.Load() || !s.ready.Load() || !s.readyForInput.Load() {
		return
	}
	s.mu.Lock()
	inTurn := s.userIsSpeaking
	if s.vadDisabled && !inTurn {
		if channels < 1 {
			channels = 1
		}
		keep := int(float64(sampleRate*channels*2) * s.prerollSecs)
		s.preroll = append(s.preroll, pcm...)
		if len(s.preroll) > keep {
			s.preroll = s.preroll[len(s.preroll)-keep:]
		}
		s.prerollRate = sampleRate
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	s.sendAudioBlob(ctx, pcm, sampleRate)
}

// sendAudioBlob sends one chunk of PCM as realtime input.
func (s *Service) sendAudioBlob(ctx context.Context, pcm []byte, sampleRate int) {
	s.sendRealtimeInput(ctx, map[string]any{
		"audio": map[string]any{
			"data":     base64.StdEncoding.EncodeToString(pcm),
			"mimeType": fmt.Sprintf("audio/pcm;rate=%d", sampleRate),
		},
	})
}

// sendRealtimeInput sends one realtime input message.
func (s *Service) sendRealtimeInput(ctx context.Context, input map[string]any) {
	if err := s.send(map[string]any{keyRealtimeInput: input}); err != nil {
		s.handleSendError(ctx, err)
	}
}

// handleUserStartedSpeaking marks the user's turn started. With server-side
// detection off, the session is told so, and the pre-roll is replayed inside
// the turn: the speech that started it arrived, and was kept, before the turn
// was detected.
func (s *Service) handleUserStartedSpeaking(ctx context.Context) {
	s.mu.Lock()
	s.userIsSpeaking = true
	s.mu.Unlock()
	if !s.vadDisabled || !s.ready.Load() || !s.readyForInput.Load() {
		return
	}
	s.sendRealtimeInput(ctx, map[string]any{"activityStart": map[string]any{}})
	s.flushPreroll(ctx)
}

// flushPreroll sends the pre-roll, then clears it.
func (s *Service) flushPreroll(ctx context.Context) {
	s.mu.Lock()
	pcm, rate := s.preroll, s.prerollRate
	s.preroll = nil
	s.mu.Unlock()
	if len(pcm) == 0 || rate == 0 {
		return
	}
	s.sendAudioBlob(ctx, pcm, rate)
}

// handleSpeechControlParams sizes the pre-roll from the start delay of the
// pipeline's voice activity detection, which is the speech its turn detection
// takes in before confirming a turn, plus a margin. A pre-roll pinned in the
// Config is left as it is.
func (s *Service) handleSpeechControlParams(f *frames.SpeechControlParamsFrame) {
	if f.VADParams == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.prerollPinned {
		return
	}
	s.prerollSecs = f.VADParams.StartSecs + prerollMarginSecs
}

// sendUserText sends the user's text as realtime input, in step with their
// audio. The API infers the end of the turn from the user's activity.
func (s *Service) sendUserText(ctx context.Context, text string) {
	if !s.ready.Load() || !s.readyForInput.Load() {
		return
	}
	s.sendRealtimeInput(ctx, map[string]any{keyText: text})
}

// SetAudioInputPaused stops, or resumes, user audio reaching the session.
func (s *Service) SetAudioInputPaused(paused bool) { s.audioPaused.Store(paused) }

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
	// Nothing is going to report this session finished now, so a turn held open
	// waiting for it is dropped rather than left with a watch running on it.
	s.discardHeldTurn()
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.ready.Store(false)
	s.stopTranscriptFlushLocked()
	// What was sent to the closed session is forgotten with it.
	s.sentResults = map[string]bool{}
	s.toolNames = map[string]string{}
	s.mu.Unlock()
	s.readyForInput.Store(false)
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
	// SessionResumptionUpdate hands out a handle to resume the session from.
	SessionResumptionUpdate *struct {
		NewHandle string `json:"newHandle"` //nolint:tagliatelle // Gemini wire field
		Resumable bool   `json:"resumable"`
	} `json:"sessionResumptionUpdate"` //nolint:tagliatelle // Gemini wire field
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
	TurnComplete        bool         `json:"turnComplete"` //nolint:tagliatelle // Gemini wire field
	// InteractionStatus says whether the session is still working. The thinking
	// models report it, and nothing else does.
	InteractionStatus string `json:"interactionStatus"` //nolint:tagliatelle // Gemini wire field
}

// interactionIdle reads whether the session has finished working, and whether it
// said anything about it at all. A model that does not report the status, and a
// value this does not know, express no opinion.
//
// The idle state was renamed, so both names answer for it.
func (sc *serverContent) interactionIdle() (idle, known bool) {
	if sc == nil {
		return false, false
	}
	switch sc.InteractionStatus {
	case "IDLE", "REQUIRES_ACTION":
		return true, true
	case "IN_PROGRESS":
		return false, true
	default:
		return false, false
	}
}

type part struct {
	Text string `json:"text"`
	// Thought marks text that is the model's reasoning rather than its answer.
	Thought    bool `json:"thought"`
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
			if connCtx.Err() == nil && !s.stopped.Load() {
				// Not closed by this service, so the session was lost. Opening it
				// again closes this one, which waits for this loop, so it is done
				// once this loop has ended, and outlives this session's context,
				// which closing it cancels.
				go s.handleConnectionError(context.WithoutCancel(connCtx), conn, err)
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
		s.handleSessionReady(ctx)
	}
	s.resetFailuresOnceStable()
	idle, statusKnown := msg.ServerContent.interactionIdle()
	// Any server message while a turn is held is the session proving it is still
	// there, so the watch on it measures silence and nothing else.
	s.restartHeldTurnWatchdog(ctx)
	if msg.ServerContent != nil {
		s.handleServerContent(ctx, msg, idle, statusKnown)
	}
	if msg.ToolCall != nil {
		s.runToolCalls(ctx, msg.ToolCall)
	}
	if u := msg.SessionResumptionUpdate; u != nil && u.Resumable && u.NewHandle != "" {
		s.mu.Lock()
		s.resumptionHandle = u.NewHandle
		s.mu.Unlock()
	}
	// A held turn is closed only once any tool call bundled with the message is
	// on its way, so the turn ends with the message fully taken in.
	if statusKnown && idle {
		s.completeHeldTurn(ctx)
	}
}

// handleServerContent takes the content of one message. Its fields are not
// exclusive: a 3.x model can bundle content and the end of its turn in one, so
// the content is taken before the turn is closed.
//
// idle and statusKnown are what the model said about whether it has finished
// working, which decides whether a completed turn ends the reply.
func (s *Service) handleServerContent(ctx context.Context, msg serverMessage, idle, statusKnown bool) {
	sc := msg.ServerContent
	if sc.Interrupted {
		// The bot was cut off, so the pipeline is interrupted. No user-speaking
		// frame goes with it: this API reports an interruption but never a turn
		// starting or ending, so a start invented here would have no stop to
		// match it. A pipeline that needs turn boundaries runs its own detection
		// alongside this service.
		slog.DebugContext(ctx, "the model reports an interruption", "service", s.Name())
		// The broadcast does not come back through this service, so it handles
		// the interruption itself, first, as it would one arriving: whatever
		// closes the response in progress goes on ahead of the interruption.
		s.handleInterruption(ctx)
		// Broadcast, not pushed on: the aggregators on either side of this
		// service both act on an interruption, and the one keeping the user's
		// turn sits upstream of it.
		_ = s.BroadcastInterruption(ctx)
	}
	if sc.ModelTurn != nil {
		s.handleModelTurn(ctx, sc.ModelTurn.Parts)
	}
	if sc.InputTranscription != nil {
		s.handleInputTranscription(ctx, sc.InputTranscription.Text)
	}
	if sc.OutputTranscription != nil {
		s.handleOutputTranscription(ctx, sc.OutputTranscription.Text)
	}
	if !sc.TurnComplete {
		return
	}
	if msg.UsageMetadata == nil {
		slog.WarnContext(ctx, "the turn completed without usage metadata", "service", s.Name())
	}
	if statusKnown && !idle {
		// The model is reasoning in the background, so this closes one chunk of
		// output and not the reply.
		s.holdTurn(ctx)
	} else {
		// This one ends the reply, so it stands in for anything still held.
		s.discardHeldTurn()
		s.handleTurnComplete(ctx)
	}
	if msg.UsageMetadata != nil {
		s.handleUsage(ctx, msg)
	}
}

// handleUsage reports the accounting that arrives with the turn it belongs to,
// against a span covering that turn.
func (s *Service) handleUsage(ctx context.Context, msg serverMessage) {
	spanCtx, end := s.traceResponse(ctx, msg)
	if s.UsageMetricsEnabled() {
		_ = s.PushTokenUsage(spanCtx, msg.UsageMetadata.tokenUsage())
	}
	end()
}

// handleInterruption drops what the interruption makes moot: a held turn, a
// reply owed for appended content, which the user's turn makes one of its own,
// and the response in progress. No response end goes with it: the
// interruption already tells the assistant aggregator the response is over.
func (s *Service) handleInterruption(ctx context.Context) {
	s.discardHeldTurn()
	s.mu.Lock()
	s.contentAwaitingResponse = false
	stopped := s.setBotRespondingLocked(false)
	s.mu.Unlock()
	if stopped && s.live.Modalities.Or(ModalityAudio) == ModalityAudio {
		_ = s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
	}
}

// setBotRespondingLocked records whether the model is answering and reports
// whether that changed. Call it with mu held.
func (s *Service) setBotRespondingLocked(responding bool) bool {
	if s.botResponding == responding {
		return false
	}
	s.botResponding = responding
	return true
}

// startResponse opens the model's response downstream if it is not open yet:
// with a TTSStartedFrame when it is spoken, so what follows reaches the
// assistant aggregator as the bot's speech.
func (s *Service) startResponse(ctx context.Context, spoken bool) {
	s.mu.Lock()
	started := s.setBotRespondingLocked(true)
	s.mu.Unlock()
	if !started {
		return
	}
	if spoken {
		_ = s.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream)
	}
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
}

// handleModelTurn takes the first part of what the model produced: its text, in
// the text modality, as the response or as a thought, and its audio.
func (s *Service) handleModelTurn(ctx context.Context, parts []part) {
	if len(parts) == 0 {
		return
	}
	p := parts[0]
	s.StopTTFBMetrics()

	if p.Text != "" {
		s.startResponse(ctx, false)
		if p.Thought {
			// Thoughts arrive whole rather than in chunks, so each is bracketed.
			_ = s.PushFrame(ctx, frames.NewLLMThoughtStartFrame(), processor.Downstream)
			_ = s.PushFrame(ctx, frames.NewLLMThoughtTextFrame(p.Text), processor.Downstream)
			_ = s.PushFrame(ctx, frames.NewLLMThoughtEndFrame(), processor.Downstream)
		} else {
			s.mu.Lock()
			s.botText += p.Text
			s.mu.Unlock()
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(p.Text), processor.Downstream)
		}
	}

	if p.InlineData == nil {
		return
	}
	switch p.InlineData.MimeType {
	case fmt.Sprintf("audio/pcm;rate=%d", outputSampleRate):
	case "audio/pcm":
		s.warnedMimeType.Do(func() {
			slog.WarnContext(ctx, "audio carries no sample rate, assuming the output rate",
				"service", s.Name(), "mime_type", p.InlineData.MimeType, "rate", outputSampleRate)
		})
	default:
		slog.WarnContext(ctx, "unrecognized server content format",
			"service", s.Name(), "mime_type", p.InlineData.MimeType)
		return
	}
	pcm, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
	if err != nil || len(pcm) == 0 {
		return
	}
	s.startResponse(ctx, true)
	_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, outputSampleRate, 1), processor.Downstream)
}

// handleTurnComplete closes the model's response: a spoken one with a
// TTSStoppedFrame as well.
func (s *Service) handleTurnComplete(ctx context.Context) {
	s.mu.Lock()
	text := s.botText
	s.botText = ""
	stopped := s.setBotRespondingLocked(false)
	s.mu.Unlock()
	if !stopped {
		return
	}
	if text == "" {
		_ = s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
	}
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// transcriptFlushAfter is how long the user's words may go without completing a
// sentence before they are sent on anyway.
const transcriptFlushAfter = 500 * time.Millisecond

// handleInputTranscription assembles the user's words into sentences. They
// arrive as single words or short phrases, so they are held until a sentence
// ends, and sent on unfinished if no more arrive for transcriptFlushAfter.
func (s *Service) handleInputTranscription(ctx context.Context, text string) {
	if text == "" {
		return
	}
	s.mu.Lock()
	s.stopTranscriptFlushLocked()
	// A sentence does not start with a space.
	if s.userTranscript == "" {
		text = strings.TrimLeft(text, " ")
	}
	s.userTranscript += text
	var sentences []string
	for {
		end := textutil.MatchEndOfSentence(s.userTranscript, "")
		if end == 0 {
			break
		}
		sentences = append(sentences, s.userTranscript[:end])
		s.userTranscript = s.userTranscript[end:]
	}
	if s.userTranscript != "" {
		s.transcriptFlush = time.AfterFunc(transcriptFlushAfter, func() { s.flushUserTranscript(ctx) })
	}
	s.mu.Unlock()
	for _, text := range sentences {
		s.pushUserTranscript(ctx, text)
	}
}

// flushUserTranscript sends on the user's words held without completing a
// sentence, once they have stopped coming.
func (s *Service) flushUserTranscript(ctx context.Context) {
	s.mu.Lock()
	text := s.userTranscript
	s.userTranscript = ""
	s.transcriptFlush = nil
	s.mu.Unlock()
	if text != "" {
		s.pushUserTranscript(ctx, text)
	}
}

// stopTranscriptFlushLocked stops the pending flush. Call it with mu held.
func (s *Service) stopTranscriptFlushLocked() {
	if s.transcriptFlush != nil {
		s.transcriptFlush.Stop()
		s.transcriptFlush = nil
	}
}

// pushUserTranscript sends the user's words upstream, where the user aggregator
// is: everything downstream of this service is the reply to them.
func (s *Service) pushUserTranscript(ctx context.Context, text string) {
	_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(text, "", frames.NowTimestamp()), processor.Upstream)
}

// handleOutputTranscription takes the words of the model's spoken answer. On
// Vertex AI they arrive before the audio they transcribe, so they may open the
// response. They go out as an LLMTextFrame, which RTVI reports as the bot's
// text but which is kept out of the conversation, and as a TTSTextFrame, which
// is what the assistant aggregator records the spoken answer from.
func (s *Service) handleOutputTranscription(ctx context.Context, text string) {
	if text == "" {
		return
	}
	s.startResponse(ctx, true)
	llmText := frames.NewLLMTextFrame(text)
	llmText.AppendToContext = false
	_ = s.PushFrame(ctx, llmText, processor.Downstream)
	ttsText := frames.NewTTSTextFrame(text, frames.TextTypeSentence)
	ttsText.IncludesInterFrameSpaces = true
	_ = s.PushFrame(ctx, ttsText, processor.Downstream)
}

// heldTurnTimeout is how long the session may go quiet while the model still
// reports that it is working before the bot's turn is closed anyway. Every
// server message restarts it, so it measures silence rather than the whole hold:
// a long reply still streaming keeps it at bay, and a session that stops
// speaking without ever reporting itself finished does not leave everything
// downstream waiting on a turn that never ends.
const heldTurnTimeout = 30 * time.Second

// holdTurn keeps the bot's turn open past a completed generation, because the
// model is still reasoning in the background and has more to say.
func (s *Service) holdTurn(ctx context.Context) {
	s.mu.Lock()
	s.turnHeld = true
	s.mu.Unlock()
	s.restartHeldTurnWatchdog(ctx)
	slog.DebugContext(ctx, "holding the bot's turn open while the model is still working",
		"service", s.Name())
}

// restartHeldTurnWatchdog starts the watch again, for a turn being held. It is
// called on every server message, since one arriving is the session proving it
// is still there.
func (s *Service) restartHeldTurnWatchdog(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.turnHeld {
		return
	}
	s.stopHeldTurnWatchdogLocked()
	s.heldTurnWatchdog = time.AfterFunc(heldTurnTimeout, func() {
		slog.WarnContext(ctx, "the session went quiet with the model still reporting that it is "+
			"working, so the bot's turn is being ended", "service", s.Name(), "after", heldTurnTimeout)
		s.completeHeldTurn(ctx)
	})
}

// stopHeldTurnWatchdogLocked stops the watch. Call it with mu held.
func (s *Service) stopHeldTurnWatchdogLocked() {
	if s.heldTurnWatchdog != nil {
		s.heldTurnWatchdog.Stop()
		s.heldTurnWatchdog = nil
	}
}

// completeHeldTurn closes a turn that was held while the model worked. Taking
// the hold and clearing it happen together, so a status arriving as the watchdog
// fires closes the turn once between them.
func (s *Service) completeHeldTurn(ctx context.Context) {
	s.mu.Lock()
	held := s.turnHeld
	s.turnHeld = false
	s.stopHeldTurnWatchdogLocked()
	s.mu.Unlock()
	if !held {
		return
	}
	s.handleTurnComplete(ctx)
}

// discardHeldTurn drops a held turn without closing it, for an interruption,
// which has ended the turn already, and for a generation that supersedes it.
func (s *Service) discardHeldTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnHeld = false
	s.stopHeldTurnWatchdogLocked()
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
