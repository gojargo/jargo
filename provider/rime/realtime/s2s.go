package realtime

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/rime/internal/rimepcm"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// errNotGenerator is returned by the generation entry point this service does
// not use: it generates continuously rather than answering a conversation.
//
//nolint:gochecknoglobals // sentinel error
var errNotGenerator = errors.New("rime realtime: the model generates continuously")

// operation is one piece of work for the session, done in the order it was
// asked for.
type operation struct {
	// messages are appended to the history.
	messages []frames.Message
	// tools is the conversation's toolset, when the operation carries one.
	tools *frames.ToolsSchema
	// toolChoice is the tool selection, when the operation carries one.
	toolChoice *frames.ToolChoice
	// delta is a settings update to apply.
	delta         *frames.LLMUpdateSettingsFrame
	runLLM        bool
	audioResponse bool
	// tool is a call whose result is written.
	tool       *tool
	toolOutput string
	toolFinal  bool
	// toolMessage is the async-tool message that settles the call after
	// progress was written.
	toolMessage string
	// answerable says the write is content a reply can answer: appended
	// messages or a tool update.
	answerable bool
	// answersContent says the reply answers earlier content, and is skipped if
	// none was written since the last reply.
	answersContent bool
}

// Service is the Rime Prism speech-to-speech processor: one Prism conversation
// driven by the standard conversation aggregators and transports.
type Service struct {
	*llm.Base
	cfg     Config
	adapter prismAdapter
	// live is the settings store updates are merged into.
	live *Settings
	// cfgErr is why the configuration cannot be used, reported when the
	// conversation would open.
	cfgErr       error
	playbackMode string
	timeouts     Timeouts
	// shutdownTimeout bounds the drain of a graceful stop.
	shutdownTimeout time.Duration

	// lifeCtx is the lifetime of the work the service owns, canceled when it
	// closes.
	lifeCtx context.Context
	cancel  context.CancelFunc
	tasks   sync.WaitGroup
	// controls carries the operations to the loop that does them.
	controls        chan operation
	controlsPending atomic.Int64
	// ready is closed once the opening has finished, one way or the other.
	ready     chan struct{}
	readyOnce sync.Once
	closeOnce sync.Once
	closing   atomic.Bool
	// replyLock serializes the replies asked for.
	replyLock lock
	// inputResampler converts audio at a rate Prism does not take. It is the
	// audio loop's alone.
	inputResampler     *resample.Resampler
	convertedInputRate int

	// audioMu guards the audio waiting to be sent.
	audioMu           sync.Mutex
	audioQueue        []frames.AudioRawData
	audioWake         chan struct{}
	pendingAudioBytes int
	audioPending      atomic.Int64

	// mu guards everything below.
	mu           sync.Mutex
	conversation *conversation
	playback     *discardPlayback
	toolChoice   frames.ToolChoice
	// replyOwed says something was written that the assistant aggregator will
	// ask a reply for.
	replyOwed         bool
	owedAudioResponse bool
	writesUnwritten   int
	contentWritten    bool
	// owedCalls are the calls whose final results were written since the last
	// reply.
	owedCalls        []*tool
	earlyAppends     [][]frames.Message
	interruptEpoch   int
	replyEpoch       int
	responseSeen     *signal
	requestPending   bool
	admittedResponse *response
	requests         int
	failure          error
	session          *session
	opening          bool
	convo            *frames.LLMContext
	// prismInstructions and prismTools are what Prism last accepted.
	prismInstructions  string
	prismTools         []toolDefinition
	history            *ledger
	transcripts        map[string]string
	replyAudioResponse bool
	// replyReports says the reply asked for reports a late tool result, and
	// survives interruptions.
	replyReports bool
	// playing are the audio outputs in the order the transport plays them; the
	// first is playing.
	playing []*output
	// skipTTS is the output configuration in effect.
	skipTTS bool
	// serviceTools are the tools the settings hold, for the base to register the
	// handlers of.
	serviceTools frames.ToolsSchema
}

var (
	_ llm.SettingsHolder         = (*Service)(nil)
	_ llm.SettingsUpdateDeferrer = (*Service)(nil)
	_ llm.ServiceToolsHolder     = (*Service)(nil)
	_ llm.AdapterHolder          = (*Service)(nil)
)

// New builds a Prism service. A configuration Validate refuses is reported as
// an error when the conversation would open, and leaves the service unusable.
func New(cfg Config) *Service {
	if cfg.Playback == "" {
		cfg.Playback = PlaybackEstimated
	}
	s := &Service{
		cfg:                cfg,
		playbackMode:       cfg.Playback,
		timeouts:           cfg.Timeouts.withDefaults(),
		shutdownTimeout:    defaultShutdownTimeout,
		controls:           make(chan operation, maxPendingOperations),
		ready:              make(chan struct{}),
		replyLock:          newLock(),
		audioWake:          make(chan struct{}, 1),
		conversation:       newConversation(),
		toolChoice:         frames.ToolChoiceAuto,
		owedAudioResponse:  true,
		responseSeen:       newSignal(),
		transcripts:        map[string]string{},
		replyAudioResponse: true,
	}
	s.cfgErr = cfg.Validate()
	live, err := newSettings(cfg)
	if err != nil {
		live = defaultSettings()
	}
	s.live = live
	s.serviceTools, _ = live.Tools.Value()
	s.lifeCtx, s.cancel = context.WithCancel(context.Background())
	s.Base = llm.New("RimeRealtime", s, llm.WithContinuousGeneration())
	s.SetModel(modelPrism)
	s.SetSystemInstruction(live.SystemInstruction.Or(""))
	s.Events().Register(EventSessionUpdated, false)
	s.Events().Register(EventRimeEvent, false)
	return s
}

// Generate is never called: this service generates continuously from the audio
// it is sent, and says so with llm.WithContinuousGeneration. It exists because
// the base identifies the service it belongs to by the generator it was built
// with.
func (s *Service) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return errNotGenerator
}

// Settings returns the store a settings update is merged into.
func (s *Service) Settings() any { return s.live }

// ServiceTools returns the tools the settings hold, which the base registers
// the handlers of while the conversation advertises none.
func (s *Service) ServiceTools() frames.ToolsSchema {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.serviceTools
}

// LLMAdapter returns the adapter the service converts through, so the base can
// add the tools the service implements itself to it.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// SessionInfo returns the session information Prism acknowledged, the
// provider's request id included. It reports false before the conversation
// opens.
func (s *Service) SessionInfo() (SessionInfo, bool) {
	s.mu.Lock()
	sess := s.session
	s.mu.Unlock()
	if sess == nil {
		return SessionInfo{}, false
	}
	info, err := sess.sessionInfo()
	return info, err == nil
}

// AcceptsIntermediateFunctionCallResults reports that intermediate results
// reach Prism, as history messages beside the open call. Prism keeps one
// result per call, so only the final result answers it.
func (s *Service) AcceptsIntermediateFunctionCallResults() bool { return true }

// CanGenerateMetrics reports that the service times the turn to its first
// output and counts its tokens.
func (s *Service) CanGenerateMetrics() bool { return true }

// ServiceMetadataFrame describes the service to the pipeline as a realtime one,
// and recommends the external turn strategies that resolve the turn boundaries
// Prism proposes.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	f.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	return f
}

// usable reports whether the service can still be given work.
func (s *Service) usable() bool { return !s.closing.Load() && s.Usable() }

// start starts the loops that do the session's work, apart from one another so
// audio and control never hold each other up.
func (s *Service) start() {
	s.tasks.Go(s.controlLoop)
	s.tasks.Go(s.audioLoop)
}

// enqueue queues an operation behind the earlier ones.
func (s *Service) enqueue(op operation) {
	if !s.usable() {
		return
	}
	s.controlsPending.Add(1)
	select {
	case s.controls <- op:
	default:
		s.controlsPending.Add(-1)
		s.fail(&failureError{message: "Prism pending operation limit exceeded"})
	}
}

// ProcessFrame queues the session's work without holding up audio and control
// frames.
//
// Every frame but an image is forwarded by the LLM base, which handles it
// first, so the work done here is the session's side of it.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch f.(type) {
	case *frames.InputImageRawFrame:
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.PushError(ctx, "Prism does not support images or video", nil, false)
		return nil
	case *frames.EndFrame:
		// Before the frame travels on, so what the drain produces precedes it.
		s.stop()
	case *frames.CancelFrame:
		s.close()
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := s.handleFrame(f, dir); err != nil {
		s.operationError(err)
	}
	return nil
}

// handleFrame does the session's side of a frame.
func (s *Service) handleFrame(f frames.Frame, dir processor.Direction) error {
	switch fr := f.(type) {
	case *frames.StartFrame:
		s.start()
	case *frames.LLMContextFrame:
		if fr.Context != nil {
			return s.handleContext(fr.Context, dir)
		}
	case *frames.LLMMessagesAppendFrame:
		s.handleMessagesAppend(fr)
	case *frames.LLMSetToolChoiceFrame:
		choice := fr.ToolChoice
		s.enqueue(operation{toolChoice: &choice})
	case *frames.LLMSetToolsFrame:
		s.enqueue(operation{delta: toolsDelta(frames.ToolsSchema{Standard: fr.Tools})})
	case *frames.LLMConfigureOutputFrame:
		s.mu.Lock()
		s.skipTTS = fr.SkipTTS
		s.mu.Unlock()
	case *frames.BotStoppedSpeakingFrame:
		s.handleBotStoppedSpeaking()
	case *frames.InterruptionFrame:
		s.handleInterruption()
	case frames.InputAudioFrame:
		s.handleInputAudio(*fr.AudioData())
	}
	return nil
}

// toolsDelta is the settings update that replaces the tools.
func toolsDelta(tools frames.ToolsSchema) *frames.LLMUpdateSettingsFrame {
	return frames.NewLLMUpdateSettingsFrame(&Settings{Tools: settings.Set(tools)})
}

// DeferSettingsUpdate queues a settings update behind the earlier operations.
// It is applied once they have been written, and nothing has changed until
// then.
func (s *Service) DeferSettingsUpdate(_ context.Context, f *frames.LLMUpdateSettingsFrame) {
	s.enqueue(operation{delta: f})
}

func (s *Service) handleMessagesAppend(f *frames.LLMMessagesAppendFrame) {
	s.mu.Lock()
	if s.convo == nil {
		// The assistant aggregator adds these to the conversation, which may
		// happen after the opening conversation frame is read.
		s.earlyAppends = append(s.earlyAppends, slices.Clone(f.Messages))
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	s.enqueue(operation{messages: slices.Clone(f.Messages), answerable: true})
	s.mu.Lock()
	s.writesUnwritten++
	s.replyOwed = true
	// The output settings apply to the reply as of the append.
	s.owedAudioResponse = !s.skipTTS
	s.mu.Unlock()
}

func (s *Service) handleInputAudio(audio frames.AudioRawData) {
	if !s.usable() {
		return
	}
	s.audioMu.Lock()
	if s.pendingAudioBytes+len(audio.Audio) > maxInputAudioBytes {
		s.audioMu.Unlock()
		s.fail(&failureError{message: "Prism microphone buffer limit exceeded"})
		return
	}
	s.pendingAudioBytes += len(audio.Audio)
	s.audioQueue = append(s.audioQueue, audio)
	s.audioPending.Add(1)
	s.audioMu.Unlock()
	select {
	case s.audioWake <- struct{}{}:
	default:
	}
}

// handleBotStoppedSpeaking takes a stop of the bot's speech. Stops name no
// output but arrive in playback order, one to an output. A stop while the
// oldest output still streams is a pause in it.
func (s *Service) handleBotStoppedSpeaking() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.playing) > 0 && s.playing[0].closed {
		s.playing = s.playing[1:]
		if len(s.playing) > 0 {
			s.playing[0].startedAt = time.Now()
		}
	}
}

func (s *Service) handleInterruption() {
	s.mu.Lock()
	s.interruptEpoch++
	// The assistant aggregator drops a held reply request on interruption.
	s.replyOwed = false
	s.contentWritten = false
	s.settleOwedCallsLocked()
	var playing *output
	if len(s.playing) > 0 {
		playing = s.playing[0]
	}
	s.playing = nil
	s.mu.Unlock()
	s.spawnRequest(func(ctx context.Context) error { return s.interrupt(ctx, playing) })
}

// bindContext ties the service to the one conversation it runs, and opens the
// Prism conversation the first time.
func (s *Service) bindContext(convo *frames.LLMContext) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.convo != nil && s.convo != convo {
		return false, settingsError("A Prism conversation requires one shared LLMContext")
	}
	initial := s.convo == nil
	s.convo = convo
	if !s.opening {
		s.opening = true
		s.tasks.Go(s.open)
	}
	return initial, nil
}

// missingAppends are the early appends the conversation does not hold yet.
func missingAppends(present []frames.Message, early [][]frames.Message) []frames.Message {
	remaining := slices.Clone(present)
	var missing []frames.Message
	for _, batch := range early {
		for _, message := range batch {
			index := slices.IndexFunc(remaining, func(m frames.Message) bool {
				return reflect.DeepEqual(m, message)
			})
			if index < 0 {
				missing = append(missing, message)
				continue
			}
			remaining = slices.Delete(remaining, index, index+1)
		}
	}
	return missing
}

// handleContext takes a conversation frame. The first one opens the Prism
// conversation; a later one traveling upstream is the assistant aggregator
// asking for a reply.
func (s *Service) handleContext(convo *frames.LLMContext, dir processor.Direction) error {
	initial, err := s.bindContext(convo)
	if err != nil {
		return err
	}
	tools, choice := convo.ToolsSchema(), convo.ToolChoice()
	op := operation{tools: &tools, toolChoice: &choice}
	s.mu.Lock()
	if initial {
		// The opening conversation becomes the history, followed by the earlier
		// appends the assistant aggregator has not added yet.
		messages := convo.Messages()
		if system := convo.System(); system != "" {
			op.messages = append(op.messages, frames.Message{Role: frames.RoleSystem, Text: system})
		}
		op.messages = append(op.messages, messages...)
		op.messages = append(op.messages, missingAppends(messages, s.earlyAppends)...)
		s.earlyAppends = nil
		op.runLLM = true
		op.audioResponse = !s.skipTTS
	} else {
		// Appended messages and tool results are already in Prism. A
		// conversation frame traveling upstream comes from the assistant
		// aggregator, asking for a reply to them.
		op.runLLM = dir == processor.Upstream && s.replyOwed
		op.audioResponse = s.owedAudioResponse
		if op.runLLM {
			s.replyOwed = false
		}
		op.answersContent = op.runLLM
	}
	s.mu.Unlock()
	s.enqueue(op)
	return nil
}

// sdkTools is the tools the settings advertise as the session takes them, with
// their fillers.
func (s *Service) sdkTools(live *Settings) ([]toolDefinition, error) {
	schema, _ := live.Tools.Value()
	tools, err := s.adapter.tools(schema)
	if err != nil {
		return nil, err
	}
	fillers, _ := live.ToolFillers.Value()
	for i := range tools {
		if filler, ok := fillers[tools[i].name]; ok {
			if err := filler.validate(); err != nil {
				return nil, settingsError("%s", err.Error())
			}
			tools[i].filler = &filler
		}
	}
	return tools, nil
}

// snapshot copies the settings store.
func (s *Service) snapshot() *Settings {
	var live Settings
	s.ReadSettings(func() { live = *s.live })
	return &live
}

// prismView is the instructions and the tools Prism should hold. The
// instructions are the whole system instruction the base composes, and the
// tools include the ones the service implements itself, such as cancel tools.
func (s *Service) prismView() (string, []toolDefinition, error) {
	tools, err := s.sdkTools(s.snapshot())
	return s.SystemInstruction(), tools, err
}

// openingTurnDetection is the turn settings the conversation opens with: the
// ones configured, with the interruption rules the settings hold.
func (s *Service) openingTurnDetection(live *Settings) (*TurnDetection, error) {
	rules, err := interruptionRulesOf(live, interruptionRules)
	if err != nil {
		return nil, err
	}
	if rules.InterruptMinWords == nil && rules.InterruptPhrases == nil && rules.AcknowledgmentPhrases == nil {
		return s.cfg.TurnDetection, nil
	}
	detection := TurnDetection{}
	if s.cfg.TurnDetection != nil {
		detection = *s.cfg.TurnDetection
	}
	if rules.InterruptMinWords != nil {
		detection.InterruptMinWords = rules.InterruptMinWords
	}
	if rules.InterruptPhrases != nil {
		detection.InterruptPhrases = rules.InterruptPhrases
	}
	if rules.AcknowledgmentPhrases != nil {
		detection.AcknowledgmentPhrases = rules.AcknowledgmentPhrases
	}
	return &detection, nil
}

// connectOptions is what the conversation opens with.
func (s *Service) connectOptions() (connectOptions, error) {
	live := s.snapshot()
	schema, _ := live.Tools.Value()
	s.SyncToolHandlers(s.lifeCtx, schema)
	instructions, tools, err := s.prismView()
	if err != nil {
		return connectOptions{}, err
	}
	detection, err := s.openingTurnDetection(live)
	if err != nil {
		return connectOptions{}, err
	}
	options := connectOptions{
		apiKey:            s.cfg.APIKey,
		endpoint:          s.cfg.Endpoint,
		voice:             live.Voice.Or(""),
		instructions:      instructions,
		tools:             tools,
		interruptOnSpeech: live.InterruptOnSpeech.Or(true),
		turnDetection:     detection,
		speed:             live.Speed.Ptr(),
		timeouts:          s.timeouts,
		requestID:         s.cfg.RequestID,
		truncation:        live.Truncation.Ptr(),
	}
	if labels, ok := live.ResponseMetadata.Value(); ok && labels != nil {
		options.responseMetadata = labels
	}
	return options, nil
}

// open opens the Prism conversation.
func (s *Service) open() {
	defer s.readyOnce.Do(func() { close(s.ready) })
	if s.cfgErr != nil {
		s.fail(s.cfgErr)
		return
	}
	options, err := s.connectOptions()
	if err != nil {
		s.fail(err)
		return
	}
	sess, err := connectWithRetry(s.lifeCtx, s.cfg.Startup, options)
	if err != nil {
		if s.lifeCtx.Err() == nil {
			s.fail(err)
		}
		return
	}
	s.mu.Lock()
	s.session = sess
	s.prismInstructions, s.prismTools = options.instructions, options.tools
	s.history = newLedger(sess)
	if s.playbackMode == PlaybackDiscard {
		s.playback = newDiscardPlayback(sess)
	}
	s.mu.Unlock()
	s.tasks.Go(func() { s.eventLoop(sess) })
	if info, err := sess.sessionInfo(); err == nil {
		s.Events().Call(s.lifeCtx, EventSessionUpdated, s, info)
	}
}

// sess is the open session.
func (s *Service) sess() (*session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session == nil {
		return nil, &failureError{message: "Prism session is not open"}
	}
	return s.session, nil
}

// controlLoop does the operations queued, one at a time and in order.
func (s *Service) controlLoop() {
	for {
		select {
		case op := <-s.controls:
			s.runOperation(op)
			s.controlsPending.Add(-1)
		case <-s.lifeCtx.Done():
			return
		}
	}
}

// runOperation does one operation, once the conversation has opened.
func (s *Service) runOperation(op operation) {
	unwritten := op.answerable
	written := func() {
		if unwritten {
			unwritten = false
			s.mu.Lock()
			s.writesUnwritten--
			s.mu.Unlock()
		}
	}
	defer written()
	select {
	case <-s.ready:
	case <-s.lifeCtx.Done():
		return
	}
	if !s.usable() {
		return
	}
	if err := s.operate(op, written); err != nil {
		s.operationError(err)
	}
}

// operate carries an operation out: the tool selection, the write, the
// settings and the reply, in that order.
func (s *Service) operate(op operation, written func()) error {
	if op.toolChoice != nil {
		choice := *op.toolChoice
		if choice != frames.ToolChoiceAuto && choice != frames.ToolChoiceNone && choice != "" {
			return settingsError("Prism supports automatic or disabled tool selection")
		}
		if choice == "" {
			choice = frames.ToolChoiceAuto
		}
		s.mu.Lock()
		s.toolChoice = choice
		s.mu.Unlock()
	}
	if len(op.messages) > 0 || op.tool != nil {
		err := s.write(op)
		written()
		if err != nil {
			return err
		}
		if op.answerable {
			s.mu.Lock()
			s.contentWritten = true
			s.mu.Unlock()
		}
	}
	if op.tools != nil && !op.tools.Empty() {
		held, _ := s.snapshot().Tools.Value()
		if !sameTools(*op.tools, held) {
			if err := s.settingsOperation(toolsDelta(*op.tools)); err != nil {
				return err
			}
		}
	}
	if op.delta != nil {
		if err := s.settingsOperation(op.delta); err != nil {
			return err
		}
	}
	if op.runLLM {
		s.startReply(op)
	}
	return nil
}

// write writes an operation's content to the history.
func (s *Service) write(op operation) error {
	if op.tool != nil {
		return s.writeToolUpdate(op.tool, op.toolOutput, op.toolFinal, op.toolMessage)
	}
	return s.appendHistory(func() ([]historyItem, error) { return historyItems(op.messages) })
}

// appendHistory writes items to the end of the history.
func (s *Service) appendHistory(items func() ([]historyItem, error)) error {
	s.mu.Lock()
	history := s.history
	s.mu.Unlock()
	if history == nil {
		return &failureError{message: "Prism history is not initialized"}
	}
	return history.append(s.lifeCtx, items)
}

// startReply asks for the reply an operation runs, unless it answers content
// and none was written since the last reply.
func (s *Service) startReply(op operation) {
	s.mu.Lock()
	if op.answersContent && !s.contentWritten {
		s.mu.Unlock()
		return
	}
	var parent *ResponseRef
	var call *ToolCallRef
	if op.answersContent {
		s.contentWritten = false
		var covered bool
		parent, call, covered = s.toolReplyTargetLocked()
		if covered {
			s.mu.Unlock()
			return
		}
	}
	disableTools := s.toolChoice == frames.ToolChoiceNone
	s.mu.Unlock()
	s.spawnRequest(func(ctx context.Context) error {
		return s.reply(ctx, op.audioResponse, call, nil, disableTools, parent)
	})
}

// toolReplyTargetLocked chooses how a reply answers the tool results written
// since the last one. It returns the tool round to continue and the call to
// report, or reports that a continuation already asked for covers them.
func (s *Service) toolReplyTargetLocked() (parent *ResponseRef, call *ToolCallRef, covered bool) {
	calls := slices.Clone(s.owedCalls)
	if len(calls) == 0 {
		return nil, nil, false
	}
	latest := calls[len(calls)-1].event.Call
	value, ok := s.conversation.responses[latest.ResponseID]
	if ok {
		// The reply holds the reference while it reports the call.
		value.operations++
	}
	s.settleOwedCallsLocked()
	if !ok || value.stopped {
		return nil, &latest, false
	}
	sameRound := !slices.ContainsFunc(calls, func(t *tool) bool {
		return t.event.Call.ResponseID != latest.ResponseID
	})
	if value.continuing && sameRound {
		// Prism's continuation waits for every result in the round.
		value.operations--
		s.conversation.release(value)
		return nil, nil, true
	}
	openCalls := false
	for _, t := range s.conversation.calls {
		if !t.submitted && t.event.Call.ResponseID == latest.ResponseID {
			openCalls = true
		}
	}
	if value.continued || openCalls {
		return nil, &latest, false
	}
	value.continued, value.continuing = true, true
	ref := value.ref
	return &ref, &latest, false
}

// settleOwedCallsLocked lets go of the references held for results that need
// no reply of their own.
func (s *Service) settleOwedCallsLocked() {
	calls := s.owedCalls
	s.owedCalls = nil
	for _, t := range calls {
		if value, ok := s.conversation.responses[t.event.Call.ResponseID]; ok {
			value.operations--
			s.conversation.release(value)
		}
	}
}

// spawnRequest runs a request to Prism in the background, bounded in number.
func (s *Service) spawnRequest(work func(ctx context.Context) error) {
	if !s.usable() {
		return
	}
	s.mu.Lock()
	if s.requests >= maxPendingOperations {
		s.mu.Unlock()
		s.tasks.Go(func() { s.fail(&failureError{message: "Prism request limit exceeded"}) })
		return
	}
	s.requests++
	s.mu.Unlock()
	s.tasks.Go(func() {
		defer func() {
			s.mu.Lock()
			s.requests--
			s.mu.Unlock()
		}()
		if err := work(s.lifeCtx); err != nil && !errors.Is(err, context.Canceled) {
			s.operationError(err)
		}
	})
}

// reply asks Prism for a reply, and lets go of the response it answers once
// the request has settled.
func (s *Service) reply(
	ctx context.Context, audioResponse bool, call *ToolCallRef, instruction *string, disableTools bool,
	parent *ResponseRef,
) error {
	defer func() {
		held := ""
		switch {
		case parent != nil:
			held = parent.ResponseID
		case call != nil:
			held = call.ResponseID
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if value, ok := s.conversation.responses[held]; ok {
			value.continuing = false
			value.operations--
			s.conversation.release(value)
		}
	}()
	return s.lockedReply(ctx, audioResponse, call, instruction, disableTools, parent, false)
}

// interrupted reports whether the user interrupted since epoch.
func (s *Service) interrupted(epoch int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return epoch != s.interruptEpoch
}

// lockedReply asks for one reply at a time. A reply the user interrupted
// before it was asked for is dropped, unless it reports a late tool result.
func (s *Service) lockedReply(
	ctx context.Context, audioResponse bool, call *ToolCallRef, instruction *string, disableTools bool,
	parent *ResponseRef, report bool,
) error {
	s.mu.Lock()
	epoch := s.interruptEpoch
	s.mu.Unlock()
	if err := s.replyLock.acquire(ctx); err != nil {
		return err
	}
	defer s.replyLock.release()
	if s.interrupted(epoch) && !report {
		return nil
	}
	if err := s.settlePlayback(ctx); err != nil {
		return err
	}
	if s.interrupted(epoch) && !report {
		return nil
	}
	s.mu.Lock()
	s.replyEpoch, s.replyReports, s.replyAudioResponse = epoch, report, audioResponse
	s.mu.Unlock()
	s.StartTTFBMetrics()
	if !disableTools {
		_, err := s.requestReply(ctx, call, instruction, parent)
		return err
	}
	return s.replyWithoutTools(ctx, call, instruction, parent)
}

// replyWithoutTools asks for a reply with the tools withdrawn for its length,
// and puts them back.
func (s *Service) replyWithoutTools(
	ctx context.Context, call *ToolCallRef, instruction *string, parent *ResponseRef,
) (err error) {
	sess, err := s.sess()
	if err != nil {
		return err
	}
	if err := s.conversation.settingsLock.acquire(ctx); err != nil {
		return err
	}
	defer s.conversation.settingsLock.release()
	s.mu.Lock()
	s.conversation.suppressingTools = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		failed := s.failure != nil
		s.mu.Unlock()
		if !s.closing.Load() && !failed {
			s.restoreTools(sess)
		}
		s.mu.Lock()
		s.conversation.suppressingTools = false
		s.mu.Unlock()
	}()
	none := []toolDefinition{}
	info, err := sess.update(ctx, sessionPatch{tools: &none})
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.conversation.suppress(info)
	s.mu.Unlock()
	_, err = s.requestReply(ctx, call, instruction, parent)
	return err
}

// restoreTools puts the tools back after a reply that ran without them. A
// session left without its tools cannot go on.
func (s *Service) restoreTools(sess *session) {
	tools, err := s.sdkTools(s.snapshot())
	if err == nil {
		_, err = sess.update(context.WithoutCancel(s.lifeCtx), sessionPatch{tools: &tools})
	}
	if err != nil {
		s.fail(err)
		return
	}
	s.mu.Lock()
	s.prismTools = tools
	s.mu.Unlock()
}

// requestReply asks for a reply and lets go of the response Prism admitted for
// the request once it has settled.
func (s *Service) requestReply(
	ctx context.Context, call *ToolCallRef, instruction *string, parent *ResponseRef,
) (ResponseRef, error) {
	s.mu.Lock()
	s.requestPending = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.requestPending = false
		if s.admittedResponse != nil {
			s.admittedResponse.operations--
			s.conversation.release(s.admittedResponse)
			s.admittedResponse = nil
		}
	}()
	return s.createReply(ctx, call, instruction, parent)
}

// continuationUnavailable reports whether a continuation failed in a way a
// requested reply can stand in for.
func continuationUnavailable(err error) bool {
	if failure, ok := errors.AsType[*Error](err); ok {
		return failure.Kind == KindInput
	}
	if refusal, ok := errors.AsType[*FaultError](err); ok {
		return refusal.Fault.Scope == scopeEvent && refusal.Fault.Code == "tool_continuation_unavailable"
	}
	return false
}

// createReply continues the tool round when there is one to continue, and
// otherwise asks for a reply, reporting the call's result when one is named.
func (s *Service) createReply(
	ctx context.Context, call *ToolCallRef, instruction *string, parent *ResponseRef,
) (ResponseRef, error) {
	sess, err := s.sess()
	if err != nil {
		return ResponseRef{}, err
	}
	if parent != nil {
		ref, err := sess.continueReply(ctx, *parent)
		if err == nil {
			err = s.createdReply(ctx, sess, ref)
		}
		if err == nil {
			return ref, nil
		}
		if !continuationUnavailable(err) {
			return ResponseRef{}, err
		}
	}
	if call != nil && instruction == nil {
		text := toolReportInstruction
		instruction = &text
	}
	ref, err := sess.requestReply(ctx, instruction, call)
	if err != nil {
		return ResponseRef{}, err
	}
	return ref, s.createdReply(ctx, sess, ref)
}

// createdReply waits for the reply's start to be handled, and cancels a reply
// that was stopped before it started.
func (s *Service) createdReply(ctx context.Context, sess *session, ref ResponseRef) error {
	for {
		s.mu.Lock()
		value, ok := s.conversation.responses[ref.ResponseID]
		if ok {
			stopped := value.stopped
			s.mu.Unlock()
			if stopped {
				return sess.cancel(ctx, ref)
			}
			return nil
		}
		s.responseSeen.reset()
		seen := s.responseSeen.ch
		s.mu.Unlock()
		select {
		case <-seen:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// settingsOperation applies a settings update, one at a time.
func (s *Service) settingsOperation(delta *frames.LLMUpdateSettingsFrame) error {
	if err := s.conversation.settingsLock.acquire(s.lifeCtx); err != nil {
		return err
	}
	defer s.conversation.settingsLock.release()
	return s.applySettings(delta)
}

// requiredSettings are the settings an update cannot clear, which Prism would
// read as keeping the value.
func requiredSettings() []string {
	return slices.Concat([]string{settingSystemInstruction}, sessionFields, interruptionRules)
}

// proposal validates an update against the settings it would produce, and
// returns it as a delta of the service's own kind with what it gives.
func (s *Service) proposal(f *frames.LLMUpdateSettingsFrame) (*Settings, map[string]any, error) {
	live := s.snapshot()
	given, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, live)
	if err != nil || !ok {
		return nil, nil, err
	}
	supplied, err := settings.Given(given)
	if err != nil {
		return nil, nil, err
	}
	delta := &Settings{}
	if err := settings.FromMap(delta, supplied); err != nil {
		return nil, nil, err
	}
	if _, err := settings.Apply(live, delta); err != nil {
		return nil, nil, err
	}
	if err := validateSettings(live); err != nil {
		return nil, nil, err
	}
	// Refuses invalid tools or fillers.
	if _, err := s.sdkTools(live); err != nil {
		return nil, nil, err
	}
	for _, name := range requiredSettings() {
		if _, given := supplied[name]; given && !hasValue(live, name) {
			return nil, nil, settingsError("Prism cannot reset %s to None", name)
		}
	}
	return delta, supplied, nil
}

// applySettings applies a settings update, keeping the local settings matched
// to Prism's.
//
// The base applies the update and recomposes the system instruction and the
// tools, and Prism is sent whatever then differs from what it holds. Prism can
// refuse an update, so a refusal restores the previous settings.
func (s *Service) applySettings(f *frames.LLMUpdateSettingsFrame) error {
	delta, supplied, err := s.proposal(f)
	if err != nil || delta == nil {
		return err
	}
	sess, err := s.sess()
	if err != nil {
		return err
	}
	// The labels are sent apart from the session update, below.
	sessionDelta := *delta
	sessionDelta.ResponseMetadata = settings.Opt[map[string]string]{}
	previous := s.SettingsState()
	changed, instructions, tools, updated, err := s.updateSession(sess, &sessionDelta)
	if err != nil {
		s.RestoreSettingsState(previous)
		s.syncTools(changed.Has(settingTools))
		return err
	}
	if updated {
		s.mu.Lock()
		s.prismInstructions, s.prismTools = instructions, tools
		s.mu.Unlock()
		s.sessionUpdated(sess)
	}
	labels, given := supplied[settingResponseMetadata]
	if !given {
		return nil
	}
	metadata, _ := labels.(map[string]string)
	if err := sess.setResponseMetadata(s.lifeCtx, metadata); err != nil {
		return err
	}
	labelsOnly := &Settings{ResponseMetadata: delta.ResponseMetadata}
	if _, err := s.ApplySettingsUpdate(s.lifeCtx, frames.NewLLMUpdateSettingsFrame(labelsOnly)); err != nil {
		return err
	}
	s.sessionUpdated(sess)
	return nil
}

func (s *Service) sessionUpdated(sess *session) {
	if info, err := sess.sessionInfo(); err == nil {
		s.Events().Call(s.lifeCtx, EventSessionUpdated, s, info)
	}
}

// syncTools brings the handlers, and the conversation when the tools changed,
// into line with the tools the settings hold.
func (s *Service) syncTools(toolsChanged bool) {
	schema, _ := s.snapshot().Tools.Value()
	s.mu.Lock()
	s.serviceTools = schema
	convo := s.convo
	s.mu.Unlock()
	// The base syncs the handlers to the conversation's tools on every
	// conversation frame, so the conversation changes with the handlers.
	if convo != nil && toolsChanged {
		convo.SetToolsSchema(schema)
	}
	s.SyncToolHandlers(s.lifeCtx, schema)
}

// updateSession applies a settings update locally and sends Prism what then
// differs from what it holds. It reports what changed, what Prism should now
// hold, and whether anything was sent.
func (s *Service) updateSession(sess *session, delta *Settings) (
	changed settings.Changed, instructions string, tools []toolDefinition, updated bool, err error,
) {
	changed, err = s.ApplySettingsUpdate(s.lifeCtx, frames.NewLLMUpdateSettingsFrame(delta))
	if err != nil {
		return changed, "", nil, false, err
	}
	s.syncTools(changed.Has(settingTools))
	instructions, tools, err = s.prismView()
	if err != nil {
		return changed, "", nil, false, err
	}
	live := s.snapshot()
	patch := sessionPatch{}
	if changed.Has(settingVoice) {
		patch.voice = live.Voice.Ptr()
	}
	if changed.Has(settingInterruptOnSpeech) {
		patch.interruptOnSpeech = live.InterruptOnSpeech.Ptr()
	}
	if changed.Has(settingTruncation) {
		patch.truncation = live.Truncation.Ptr()
	}
	if changed.Has(settingSpeed) {
		patch.speed = live.Speed.Ptr()
	}
	s.mu.Lock()
	if instructions != s.prismInstructions {
		patch.instructions = &instructions
	}
	if !reflect.DeepEqual(tools, s.prismTools) {
		patch.tools = &tools
	}
	s.mu.Unlock()
	if names := slices.DeleteFunc(slices.Clone(interruptionRules), func(name string) bool {
		return !changed.Has(name)
	}); len(names) > 0 {
		rules, err := interruptionRulesOf(live, names)
		if err != nil {
			return changed, "", nil, false, err
		}
		patch.turnDetection = &rules
	}
	if patch.empty() {
		return changed, instructions, tools, false, nil
	}
	_, err = sess.update(s.lifeCtx, patch)
	return changed, instructions, tools, err == nil, err
}

// audioLoop sends the audio queued, in order.
func (s *Service) audioLoop() {
	for {
		s.audioMu.Lock()
		if len(s.audioQueue) == 0 {
			s.audioMu.Unlock()
			select {
			case <-s.audioWake:
				continue
			case <-s.lifeCtx.Done():
				return
			}
		}
		audio := s.audioQueue[0]
		s.audioQueue = s.audioQueue[1:]
		s.audioMu.Unlock()
		s.sendQueuedAudio(audio)
		s.audioMu.Lock()
		s.pendingAudioBytes -= len(audio.Audio)
		s.audioMu.Unlock()
		s.audioPending.Add(-1)
	}
}

func (s *Service) sendQueuedAudio(audio frames.AudioRawData) {
	select {
	case <-s.ready:
	case <-s.lifeCtx.Done():
		return
	}
	if !s.usable() {
		return
	}
	if err := s.sendAudio(audio); err != nil {
		s.fail(err)
	}
}

// downmix averages the channels of interleaved PCM into one.
func downmix(pcm []byte, channels int) []byte {
	out := make([]byte, 0, len(pcm)/channels)
	for i := 0; i+channels*2 <= len(pcm); i += channels * 2 {
		sum := 0
		for c := range channels {
			sum += int(int16(uint16(pcm[i+2*c]) | uint16(pcm[i+2*c+1])<<8))
		}
		mean := int16(sum / channels)
		out = append(out, byte(mean), byte(mean>>8))
	}
	return out
}

// standardRate reports whether Prism takes audio at rate as it is.
func standardRate(rate int) bool {
	return slices.Contains([]int{8000, 16000, 24000, 48000}, rate)
}

// sendAudio sends one frame of input audio. Audio with more than two channels
// is mixed down, and audio at a rate Prism does not take is resampled to
// 16 kHz mono.
func (s *Service) sendAudio(audio frames.AudioRawData) error {
	sess, err := s.sess()
	if err != nil {
		return err
	}
	pcm, rate, channels := audio.Audio, audio.SampleRate, audio.NumChannels
	if channels < 1 || channels > 32 || rate < 8000 || rate > 192000 || len(pcm)%(channels*2) != 0 {
		return settingsError("Prism input requires complete PCM16 frames, 1 to 32 channels, and 8 to 192 kHz")
	}
	if channels > 2 || (!standardRate(rate) && channels > 1) {
		pcm, channels = downmix(pcm, channels), 1
	}
	if s.convertedInputRate != 0 && rate != s.convertedInputRate {
		if err := s.flushResampler(sess); err != nil {
			return err
		}
	}
	if !standardRate(rate) {
		if s.inputResampler == nil {
			resampler, err := resample.New(rate, rimepcm.WireRate, 1)
			if err != nil {
				return err
			}
			s.inputResampler = resampler
		}
		s.convertedInputRate = rate
		pcm, rate = s.inputResampler.Process(pcm), rimepcm.WireRate
	}
	size := maxAudioChunk - maxAudioChunk%(channels*2)
	for offset := 0; offset < len(pcm); offset += size {
		chunk := audioChunk{
			data:   pcm[offset:min(offset+size, len(pcm))],
			format: rimepcm.Format{SampleRate: rate, Channels: channels},
		}
		if err := sess.sendAudio(chunk); err != nil {
			return err
		}
	}
	return nil
}

// flushResampler sends the audio the resampler still holds, and lets it go.
func (s *Service) flushResampler(sess *session) error {
	resampler := s.inputResampler
	s.inputResampler, s.convertedInputRate = nil, 0
	if resampler == nil {
		return nil
	}
	if tail := resampler.Flush(); len(tail) > 0 {
		return sess.sendAudio(audioChunk{data: tail, format: defaultFormat()})
	}
	return nil
}

// eventLoop handles the session's events until it ends.
func (s *Service) eventLoop(sess *session) {
	var err error
	for err == nil {
		var event *Event
		event, err = sess.next(s.lifeCtx)
		if err != nil {
			break
		}
		s.mu.Lock()
		failed := s.failure != nil
		s.mu.Unlock()
		if s.closing.Load() || failed {
			return
		}
		err = s.handleEvent(sess, event)
	}
	if s.closing.Load() || s.lifeCtx.Err() != nil {
		return
	}
	if errors.Is(err, io.EOF) {
		err = &failureError{message: "Prism event stream ended"}
	}
	s.mu.Lock()
	history := s.history
	s.mu.Unlock()
	if history != nil && history.lock.locked() {
		err = &HistoryUpdateError{Status: historyUncertain, Err: err}
	}
	s.fail(err)
}

func (s *Service) push(f frames.Frame) {
	_ = s.PushFrame(s.lifeCtx, f, processor.Downstream)
}

// handleEvent maps one session event onto pipeline frames.
func (s *Service) handleEvent(sess *session, event *Event) error {
	s.Events().Call(s.lifeCtx, EventRimeEvent, s, event)
	switch payload := event.Payload.(type) {
	case SpeechStarted:
		return s.Broadcast(s.lifeCtx, func() frames.Frame { return frames.NewProposedUserStartedSpeakingFrame() })
	case SpeechStopped:
		s.StartTTFBMetrics()
		return s.Broadcast(s.lifeCtx, func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() })
	case TranscriptDelta:
		return s.handleTranscript(payload.ItemID, payload.Delta, payload.Transcript, false)
	case TranscriptFinal:
		return s.handleTranscript(payload.ItemID, payload.Text, nil, true)
	case TranscriptFailed:
		s.mu.Lock()
		delete(s.transcripts, payload.ItemID)
		s.mu.Unlock()
		failure := &FaultError{Fault: payload.Error}
		s.PushError(s.lifeCtx, failure.Error(), failure, false)
	case ResponseStarted:
		return s.handleResponseStarted(sess, payload)
	case ResponseAbandoned:
		s.mu.Lock()
		value, ok := s.conversation.responses[payload.Response.ResponseID]
		if ok {
			value.stopped = true
		}
		s.mu.Unlock()
		if !ok {
			return unknownResponse()
		}
		return s.BroadcastInterruption(s.lifeCtx)
	case MessageStarted:
		return s.handleMessageStarted(payload)
	case TextDelta, TextDone, AudioDelta, AudioDone:
		return s.handleOutput(event.Payload)
	case ToolCall:
		return s.handleToolCall(payload)
	case ResponseEnded:
		return s.handleResponseEnded(payload)
	case FaultEvent:
		s.handleFault(payload.Error)
	}
	return nil
}

func unknownResponse() error {
	return &failureError{message: "Prism event refers to an unknown response"}
}

// handleTranscript pushes what Prism heard the user say, in both directions.
func (s *Service) handleTranscript(itemID, text string, full *string, final bool) error {
	s.mu.Lock()
	switch {
	case final:
	case full != nil:
		// A full transcript replaces the partial text, revisions included.
		text = *full
	default:
		text = s.transcripts[itemID] + text
	}
	if len(text) > maxRetainedText || len(s.transcripts) >= retentionLimit {
		s.mu.Unlock()
		return &failureError{message: "Prism transcript retention limit exceeded"}
	}
	if final {
		delete(s.transcripts, itemID)
	} else {
		s.transcripts[itemID] = text
	}
	s.mu.Unlock()
	timestamp := frames.NowTimestamp()
	return s.Broadcast(s.lifeCtx, func() frames.Frame {
		if final {
			return frames.NewTranscriptionFrame(text, "", timestamp)
		}
		return frames.NewInterimTranscriptionFrame(text, "", timestamp)
	})
}

// supersedeOpenCallsLocked closes the round of every call still running that
// another reply made.
func (s *Service) supersedeOpenCallsLocked(responseID string) {
	for _, t := range s.conversation.calls {
		if !t.final && t.event.Call.ResponseID != responseID {
			t.superseded = true
		}
	}
}

// startResponseLocked takes a reply that started into the conversation.
func (s *Service) startResponseLocked(sess *session, payload ResponseStarted) (*response, error) {
	value, err := s.conversation.start(sess, payload)
	if err != nil {
		return nil, err
	}
	s.supersedeOpenCallsLocked(payload.Response.ResponseID)
	// A reply that starts while content is being written may not take it in,
	// so the reply stays owed.
	if s.writesUnwritten == 0 {
		s.replyOwed, s.contentWritten = false, false
		s.settleOwedCallsLocked()
	}
	if s.requestPending && (payload.Cause == causeProactive || payload.Cause == causeTools) {
		if s.admittedResponse != nil {
			return nil, &failureError{message: "Prism created multiple responses for one request"}
		}
		value.operations++
		s.admittedResponse = value
	}
	if s.playback != nil {
		if err := s.playback.start(payload.Response); err != nil {
			return nil, err
		}
	}
	if payload.Cause == causeProactive {
		value.audioResponse = s.replyAudioResponse
		value.stopped = !s.replyReports && s.replyEpoch != s.interruptEpoch
	}
	if payload.Parent != nil {
		parent, ok := s.conversation.responses[payload.Parent.ResponseID]
		value.stopped = ok && parent.stopped
		if ok {
			value.audioResponse = parent.audioResponse
		}
	}
	if s.playbackMode == PlaybackDiscard {
		value.audioResponse = false
	}
	s.responseSeen.fire()
	return value, nil
}

func (s *Service) handleResponseStarted(sess *session, payload ResponseStarted) error {
	s.mu.Lock()
	value, err := s.startResponseLocked(sess, payload)
	if err != nil {
		s.mu.Unlock()
		return err
	}
	stopped, ref, inputIDs := value.stopped, value.ref, value.inputIDs
	s.mu.Unlock()

	started := frames.NewLLMFullResponseStartFrame()
	info, _ := sess.sessionInfo()
	var revision any
	if payload.SettingsRevision != nil {
		revision = *payload.SettingsRevision
	}
	started.Metadata()["rime"] = map[string]any{
		"provider_request_id": info.RequestID,
		"response_id":         payload.Response.ResponseID,
		"input_item_ids":      inputIDs,
		"settings_revision":   revision,
		"response_metadata":   payload.Metadata,
		"instruction_item_id": payload.InstructionItemID,
	}
	s.push(started)
	if stopped {
		s.spawnRequest(func(ctx context.Context) error { return sess.cancel(ctx, ref) })
	}
	return nil
}

func (s *Service) handleMessageStarted(payload MessageStarted) error {
	s.mu.Lock()
	value, ok := s.conversation.responses[payload.Output.Response.ResponseID]
	if !ok {
		s.mu.Unlock()
		return unknownResponse()
	}
	if len(value.outputs) >= retentionLimit {
		s.mu.Unlock()
		return &failureError{message: "Prism output retention limit exceeded"}
	}
	out := newOutput(payload.Output)
	value.outputs[payload.Output.ItemID] = out
	value.outputOrder = append(value.outputOrder, payload.Output.ItemID)
	stopped := value.stopped
	s.mu.Unlock()
	if !stopped {
		started := frames.NewTTSStartedFrame()
		started.ContextID = out.contextID
		s.push(started)
	}
	return nil
}

// outputOf is the output an output event is about, and the reply it belongs
// to.
func (s *Service) outputLocked(ref OutputRef) (*response, *output, error) {
	value, ok := s.conversation.responses[ref.Response.ResponseID]
	if !ok {
		return nil, nil, unknownResponse()
	}
	out, ok := value.outputs[ref.ItemID]
	if !ok {
		return nil, nil, &failureError{message: "Prism event refers to an unknown output"}
	}
	return value, out, nil
}

func outputRef(payload EventPayload) OutputRef {
	switch p := payload.(type) {
	case TextDelta:
		return p.Output
	case TextDone:
		return p.Output
	case AudioDelta:
		return p.Output
	case AudioDone:
		return p.Output
	default:
		return OutputRef{}
	}
}

// handleOutput takes the text and the audio of a message.
func (s *Service) handleOutput(payload EventPayload) error {
	s.mu.Lock()
	value, out, err := s.outputLocked(outputRef(payload))
	if err != nil || value.stopped || out.closed {
		s.mu.Unlock()
		return err
	}
	switch p := payload.(type) {
	case TextDelta:
		out.text += p.Delta
		tooLong := len(out.text) > maxRetainedText
		s.mu.Unlock()
		s.StopTTFBMetrics(s.lifeCtx)
		if tooLong {
			return &failureError{message: "Prism output text retention limit exceeded"}
		}
		text := frames.NewLLMTextFrame(p.Delta)
		text.AppendToContext = false
		s.push(text)
	case TextDone:
		out.text = p.Text
		s.mu.Unlock()
		s.push(frames.NewTTSTextFrame(p.Text, frames.TextTypeSentence))
	case AudioDelta:
		s.handleAudioLocked(value, out, p)
	default:
		s.mu.Unlock()
		s.endOutput(out)
	}
	return nil
}

// handleAudioLocked pushes a chunk of a message's audio, and counts it toward
// the estimate of how far the message has played. It lets go of the lock.
func (s *Service) handleAudioLocked(value *response, out *output, payload AudioDelta) {
	if !value.audioResponse {
		s.mu.Unlock()
		return
	}
	if len(s.playing) == 0 || s.playing[len(s.playing)-1] != out {
		s.playing = append(s.playing, out)
	}
	out.audioMS += float64(len(payload.Audio)) * 1000 / float64(payload.SampleRate*payload.Channels*2)
	if out.startedAt.IsZero() {
		out.startedAt = time.Now()
	}
	contextID := out.contextID
	s.mu.Unlock()
	audio := frames.NewTTSAudioRawFrame(payload.Audio, payload.SampleRate, payload.Channels)
	audio.ContextID = contextID
	s.StopTTFBMetrics(s.lifeCtx)
	s.push(audio)
}

// endOutput closes a message's speech, once.
func (s *Service) endOutput(out *output) {
	s.mu.Lock()
	closed := out.closed
	out.closed = true
	s.mu.Unlock()
	if !closed {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = out.contextID
		s.push(stopped)
	}
}

func (s *Service) handleToolCall(payload ToolCall) error {
	s.mu.Lock()
	callID := payload.Call.CallID
	if _, ok := s.conversation.calls[callID]; ok {
		s.mu.Unlock()
		return nil
	}
	if len(s.conversation.calls) >= maxRetainedCalls {
		s.mu.Unlock()
		return &failureError{message: "Prism tool retention limit exceeded"}
	}
	value, ok := s.conversation.responses[payload.Call.ResponseID]
	if !ok || value.stopped {
		s.mu.Unlock()
		if !ok {
			return unknownResponse()
		}
		return nil
	}
	if value.toolsDisabled {
		s.mu.Unlock()
		return &failureError{message: "Prism emitted a tool call during a tool-disabled reply"}
	}
	s.conversation.calls[callID] = &tool{event: payload}
	value.calls[callID] = true
	convo := s.convo
	s.mu.Unlock()
	call := frames.ToolCall{ID: callID, Name: payload.Name, Args: []byte(payload.raw)}
	return s.RunFunctionCalls(s.lifeCtx, convo, []frames.ToolCall{call})
}

func (s *Service) handleResponseEnded(payload ResponseEnded) error {
	s.mu.Lock()
	value, ok := s.conversation.responses[payload.Response.ResponseID]
	if !ok {
		s.mu.Unlock()
		return unknownResponse()
	}
	value.ended = true
	outputs := make([]*output, 0, len(value.outputOrder))
	for _, id := range value.outputOrder {
		outputs = append(outputs, value.outputs[id])
	}
	s.mu.Unlock()
	for _, out := range outputs {
		s.endOutput(out)
	}
	if usage := payload.Usage; usage != nil && s.UsageMetricsEnabled() {
		_ = s.PushTokenUsage(s.lifeCtx, frames.LLMTokenUsage{
			PromptTokens:     int64(usage.InputTokens),
			CompletionTokens: int64(usage.OutputTokens),
			TotalTokens:      int64(usage.TotalTokens),
		})
	}
	s.mu.Lock()
	discarding := s.playback != nil
	if discarding {
		s.playback.end(payload.Response)
	}
	s.conversation.release(value)
	s.mu.Unlock()
	if discarding {
		s.spawnRequest(s.settlePlayback)
	}
	s.push(frames.NewLLMFullResponseEndFrame())
	return nil
}

// handleFault takes a failure Prism reported that answers no request.
func (s *Service) handleFault(fault Fault) {
	if fault.RequestID != "" {
		return
	}
	failure := &FaultError{Fault: fault}
	if fault.Scope == scopeSession {
		s.fail(failure)
		return
	}
	if fault.Code == "tool_result_timeout" && s.supersedeRunningLocked(fault.CallIDs) {
		// The application's tool is still running. Prism records its result
		// when it arrives, and a reply can still report it.
		slog.DebugContext(s.lifeCtx, "prism gave up waiting for a tool that is still running",
			"service", s.Name(), "err", failure)
		return
	}
	s.PushError(s.lifeCtx, failure.Error(), failure, false)
}

// supersedeRunningLocked closes the round of the calls named when one of them
// is still running, and reports whether one is. It takes the lock itself.
func (s *Service) supersedeRunningLocked(callIDs []string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	running := slices.ContainsFunc(callIDs, func(id string) bool {
		t, ok := s.conversation.calls[id]
		return ok && !t.final
	})
	if !running {
		return false
	}
	for _, id := range callIDs {
		if t, ok := s.conversation.calls[id]; ok {
			t.superseded = true
		}
	}
	return true
}

// interrupt stops what Prism is saying, and tells it how much of the output
// playing was heard.
func (s *Service) interrupt(ctx context.Context, playing *output) error {
	s.mu.Lock()
	sess := s.session
	var stopping []*response
	for _, key := range s.conversation.order {
		value := s.conversation.responses[key]
		other := playing == nil || playing.ref.Response.ResponseID != value.ref.ResponseID
		if value.stopped || (value.ended && len(value.calls) == 0 && other) {
			continue
		}
		stopping = append(stopping, value)
	}
	s.mu.Unlock()
	if sess == nil {
		return nil
	}
	for _, value := range stopping {
		s.mu.Lock()
		value.stopped = true
		ended := value.ended
		outputs := make([]*output, 0, len(value.outputOrder))
		for _, id := range value.outputOrder {
			outputs = append(outputs, value.outputs[id])
		}
		s.mu.Unlock()
		if !ended {
			if err := sess.cancel(ctx, value.ref); err != nil {
				return err
			}
		}
		for _, out := range outputs {
			s.endOutput(out)
		}
	}
	// The transports return neither the identity of what they play nor a
	// device clock. One estimate from the time elapsed is reported, for the
	// output playing now only.
	s.mu.Lock()
	report := playing != nil && !playing.startedAt.IsZero()
	var position playbackInterrupted
	if report {
		elapsed := float64(time.Since(playing.startedAt)) / float64(time.Millisecond)
		position = playbackInterrupted{output: playing.ref, audioEndMS: int(min(elapsed, playing.audioMS))}
	}
	s.mu.Unlock()
	if !report {
		return nil
	}
	return sess.reportInterrupted(ctx, position)
}

// settlePlayback sends the receipts of the discarded replies that have ended,
// in the order the replies were created.
func (s *Service) settlePlayback(ctx context.Context) error {
	s.mu.Lock()
	playback := s.playback
	s.mu.Unlock()
	if playback == nil {
		return nil
	}
	if err := playback.lock.acquire(ctx); err != nil {
		return err
	}
	defer playback.lock.release()
	for {
		s.mu.Lock()
		if len(playback.order) == 0 {
			s.mu.Unlock()
			return nil
		}
		key := playback.order[0]
		entry := playback.pending[key]
		if !entry.ended {
			s.mu.Unlock()
			return nil
		}
		s.mu.Unlock()
		zero := 0.0
		err := playback.session.reportFinished(ctx, playbackFinished{response: entry.ref, playedMS: &zero})
		s.mu.Lock()
		if _, ok := playback.pending[key]; ok {
			delete(playback.pending, key)
			playback.order = slices.DeleteFunc(playback.order, func(id string) bool { return id == key })
		}
		s.mu.Unlock()
		entry.hold.release()
		if err != nil {
			return err
		}
	}
}

// PushFrame pushes a frame, writing each tool result to Prism as it is
// produced.
//
// The frames travel on to the assistant aggregator, which records them in the
// conversation and decides when Prism should reply.
func (s *Service) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if dir == processor.Downstream {
		switch fr := f.(type) {
		case *frames.FunctionCallResultFrame:
			s.handleFunctionCallOutcome(fr.ToolCallID, fr, nil)
		case *frames.FunctionCallCancelFrame:
			s.handleFunctionCallOutcome(fr.ToolCallID, nil, fr)
		}
	}
	return s.Base.PushFrame(ctx, f, dir)
}

// handleFunctionCallOutcome queues the write of a tool's result, or of its
// cancellation.
func (s *Service) handleFunctionCallOutcome(
	callID string, result *frames.FunctionCallResultFrame, canceled *frames.FunctionCallCancelFrame,
) {
	s.mu.Lock()
	t, ok := s.conversation.calls[callID]
	if !ok || t.final {
		s.mu.Unlock()
		return
	}
	final := canceled != nil || result.Properties.Final()
	t.final = final
	s.mu.Unlock()
	report := final && s.reportsLateResult(t, result)
	s.mu.Lock()
	t.report = report
	op := operation{
		tool:       t,
		toolOutput: toolOutput(callID, result, final),
		toolFinal:  final,
		answerable: !report,
	}
	if t.progress {
		op.toolMessage = settlingMessage(callID, result)
	}
	t.progress = t.progress || !final
	s.mu.Unlock()
	s.enqueue(op)
	if report {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writesUnwritten++
	s.replyOwed = true
	s.owedAudioResponse = !s.skipTTS
	if value, ok := s.conversation.responses[t.event.Call.ResponseID]; ok {
		s.owedAudioResponse = value.audioResponse
	}
}

// resultText is a tool's result as it is written, the completion marker when
// the tool produced none.
func resultText(result *frames.FunctionCallResultFrame) string {
	if result.Result == "" {
		return frames.ToolResultCompleted
	}
	return result.Result
}

// toolOutput is what is written for a tool's outcome: the result, the
// cancellation marker, or the progress message for an intermediate result.
func toolOutput(callID string, result *frames.FunctionCallResultFrame, final bool) string {
	if result == nil {
		return frames.ToolResultCanceled
	}
	if final {
		return resultText(result)
	}
	return frames.NewAsyncToolIntermediateMessage(callID, resultText(result)).Text
}

// settlingMessage is the async-tool message that states a call's outcome again
// after its progress.
func settlingMessage(callID string, result *frames.FunctionCallResultFrame) string {
	if result == nil {
		return frames.NewAsyncToolCanceledMessage(callID).Text
	}
	return frames.NewAsyncToolFinalMessage(callID, resultText(result)).Text
}

// reportsLateResult reports whether an asynchronous call's final result needs
// a reply of its own.
//
// A round closes when a later reply starts or Prism's deadline for the result
// passes. Prism keeps a result that arrives after that in its history but
// mentions it only when asked, and the assistant aggregator drops the reply it
// holds for the result when the user interrupts.
func (s *Service) reportsLateResult(t *tool, result *frames.FunctionCallResultFrame) bool {
	if result == nil || result.Result == "" {
		return false
	}
	runLLM := result.RunLLM
	if result.Properties != nil && result.Properties.RunLLM != nil {
		runLLM = result.Properties.RunLLM
	}
	if (runLLM != nil && !*runLLM) || !s.FunctionIsAsync(t.event.Name) {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	value, ok := s.conversation.responses[t.event.Call.ResponseID]
	return t.superseded || !ok || value.stopped
}

// writeToolUpdate writes a tool's progress or its final result.
func (s *Service) writeToolUpdate(t *tool, output string, final bool, message string) error {
	developer := func(text string) func() ([]historyItem, error) {
		return func() ([]historyItem, error) {
			return []historyItem{{kind: itemMessage, role: string(frames.RoleSystem), text: text}}, nil
		}
	}
	if !final {
		// Prism keeps one result per call, so progress goes in beside the open
		// call.
		return s.appendHistory(developer(output))
	}
	if err := s.submitToolResult(t, output); err != nil {
		// An unsettled call leaves the conversation in an unknown state.
		s.fail(err)
		return err
	}
	s.mu.Lock()
	t.submitted = true
	call := t.event.Call
	if value, ok := s.conversation.responses[call.ResponseID]; ok {
		delete(value.calls, call.CallID)
		// A reply that reports the call needs the response reference.
		value.operations++
	}
	s.mu.Unlock()
	if message != "" {
		// Prism files the result beside its call, ahead of the progress written
		// since, so the result is stated again after that progress.
		if err := s.appendHistory(developer(message)); err != nil {
			s.operationError(err)
		}
	}
	s.mu.Lock()
	report := t.report
	if !report {
		s.owedCalls = append(s.owedCalls, t)
	}
	s.mu.Unlock()
	if report {
		s.spawnRequest(func(ctx context.Context) error { return s.reportLateResult(ctx, t) })
	}
	return nil
}

// submitToolResult records a tool's result, serialized with the history
// writes.
func (s *Service) submitToolResult(t *tool, output string) error {
	s.mu.Lock()
	history, sess := s.history, s.session
	s.mu.Unlock()
	if history == nil || sess == nil {
		return &failureError{message: "Prism history is not initialized"}
	}
	if err := history.lock.acquire(s.lifeCtx); err != nil {
		return err
	}
	defer history.lock.release()
	return sess.submitToolResult(t.event.Call, output)
}

// retryableReport reports whether a report that failed is worth asking for
// once more: the user's speech, or a reply that stays busy, can push it aside.
func retryableReport(err error) bool {
	if _, ok := errors.AsType[*AdmissionTimeoutError](err); ok {
		return true
	}
	refusal, ok := errors.AsType[*FaultError](err)
	return ok && refusal.Fault.Scope != scopeSession
}

// reportLateResult asks for a reply that reports the result of a call whose
// round had closed. It waits for Prism to be idle again once.
func (s *Service) reportLateResult(ctx context.Context, t *tool) error {
	call := t.event.Call
	s.mu.Lock()
	value, ok := s.conversation.responses[call.ResponseID]
	audioResponse := !s.skipTTS
	if ok {
		audioResponse = value.audioResponse
	}
	s.mu.Unlock()
	defer func() {
		if ok {
			s.mu.Lock()
			value.operations--
			s.conversation.release(value)
			s.mu.Unlock()
		}
	}()
	var err error
	for attempt := range 2 {
		s.mu.Lock()
		disableTools := s.toolChoice == frames.ToolChoiceNone
		s.mu.Unlock()
		err = s.lockedReply(ctx, audioResponse, &call, nil, disableTools, nil, true)
		if err == nil || attempt > 0 || !retryableReport(err) {
			return err
		}
		slog.DebugContext(ctx, "requesting the tool result report again", "service", s.Name(), "err", err)
	}
	return err
}

// operationError reports a failed operation. One the conversation survives is
// reported as an error; any other ends the conversation.
func (s *Service) operationError(err error) {
	recoverable := false
	if update, ok := errors.AsType[*HistoryUpdateError](err); ok {
		recoverable = update.Recoverable()
	} else if _, ok := errors.AsType[*SettingsError](err); ok {
		recoverable = true
	} else if _, ok := errors.AsType[*AdmissionTimeoutError](err); ok {
		recoverable = true
	} else if failure, ok := errors.AsType[*Error](err); ok {
		recoverable = failure.Kind == KindInput
	} else if refusal, ok := errors.AsType[*FaultError](err); ok {
		recoverable = refusal.Fault.Scope == scopeEvent
	} else if errors.Is(err, settings.ErrType) || errors.Is(err, settings.ErrNotSettings) {
		recoverable = true
	}
	if !recoverable {
		s.fail(err)
		return
	}
	s.PushError(s.lifeCtx, err.Error(), err, false)
}

// fail ends the conversation: the bot is interrupted, and the service is left
// unusable.
func (s *Service) fail(err error) {
	s.mu.Lock()
	if s.failure != nil {
		s.mu.Unlock()
		return
	}
	s.failure = err
	s.mu.Unlock()
	_ = s.BroadcastInterruption(s.lifeCtx)
	s.PushError(s.lifeCtx, err.Error(), err, false, processor.ForceTreatAsPermanent())
}

// drained reports whether nothing is left for a graceful stop to wait for.
func (s *Service) drained() bool {
	if s.controlsPending.Load() > 0 || s.audioPending.Load() > 0 {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.requests > 0 {
		return false
	}
	for _, value := range s.conversation.responses {
		if !value.ended {
			return false
		}
	}
	return true
}

// stop drains the requests queued, then closes the session, within the
// shutdown timeout.
func (s *Service) stop() {
	s.mu.Lock()
	opened := s.opening
	s.mu.Unlock()
	deadline := time.Now().Add(s.shutdownTimeout)
	for opened && !s.drained() {
		if time.Now().After(deadline) {
			slog.Warn("prism shutdown drain deadline expired", "service", s.Name())
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	s.close()
}

// close cancels the work the service owns, without waiting for playback to
// complete, and closes the session.
func (s *Service) close() {
	s.closeOnce.Do(func() {
		s.closing.Store(true)
		s.cancel()
		s.mu.Lock()
		sess := s.session
		s.mu.Unlock()
		if sess != nil {
			// Ends what the loops are waiting on the session for.
			sess.close()
		}
		s.tasks.Wait()
		s.mu.Lock()
		if s.playback != nil {
			s.playback.close()
		}
		s.conversation.close()
		s.mu.Unlock()
	})
}

// Cleanup closes the session after a startup that failed or when the pipeline
// is torn down.
func (s *Service) Cleanup(ctx context.Context) error {
	s.close()
	return s.Base.Cleanup(ctx)
}
