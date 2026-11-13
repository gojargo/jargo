package novasonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/adapter"
	novasonicadapter "github.com/gojargo/jargo/adapter/novasonic"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/google/uuid"
)

// The roles and text stages Nova Sonic reports content with.
const (
	roleSystem    = novasonicadapter.RoleSystem
	roleUser      = novasonicadapter.RoleUser
	roleAssistant = novasonicadapter.RoleAssistant

	// stageFinal is text that has been said; stageSpeculative is text the
	// model plans to say, which arrives ahead of its audio.
	stageFinal       = "FINAL"
	stageSpeculative = "SPECULATIVE"

	contentTypeAudio = "AUDIO"
	contentTypeTool  = "TOOL"

	stopReasonInterrupted = "INTERRUPTED"
	stopReasonEndTurn     = "END_TURN"

	// receiveWaitTimeout bounds waiting for a session's receive loop to finish
	// when it is disconnected.
	receiveWaitTimeout = time.Second
)

// errStreamEnded reports the model ending the session's stream while the
// service still wanted it.
//
//nolint:gochecknoglobals // sentinel error
var errStreamEnded = errors.New("AWS Bedrock ended the response stream")

// errArgumentsNotJSON reports a tool call whose arguments are not JSON.
//
//nolint:gochecknoglobals // sentinel error
var errArgumentsNotJSON = errors.New("the arguments of a tool call are not JSON")

// currentContent is the content block being received: the model delivers one
// block at a time, each opened with contentStart and closed with contentEnd.
type currentContent struct {
	typ       string
	role      string
	textStage string
	text      string
}

// Service is the Nova Sonic speech-to-speech service.
type Service struct {
	*llm.Base
	cfg     Config
	live    *Settings
	audio   AudioConfig
	tools   frames.ToolsSchema
	adapter *novasonicadapter.Adapter
	sc      *sessionContinuation

	// newClient builds the client sessions are opened with; a test replaces it.
	newClient func(ctx context.Context, cfg Config) (sessionClient, error)
	// triggerAudio is the recording of the user saying "ready" that starts the
	// first-generation model speaking.
	triggerAudio []byte

	// resampleMu guards the resampler converting the user's audio to the input
	// rate the session declares, and inputResampleFrom, the rate of the audio
	// it converts. The frame path resamples; a trigger starts it afresh.
	resampleMu        sync.Mutex
	inputResampler    *resample.Resampler
	inputResampleFrom int

	mu sync.Mutex
	// The session and its state. Guarded by mu.
	convo                          *frames.LLMContext
	client                         sessionClient
	stream                         sessionStream
	receiveDone                    chan struct{}
	promptName                     string
	audioContentName               string
	content                        *currentContent
	assistantResponding            bool
	readyToSendContext             bool
	triggeringAssistantResponse    bool
	waitingForTriggerTranscription bool
	disconnecting                  bool
	connectedTime                  time.Time
	wantsConnection                bool
	userTextBuffer                 string
	completedToolCalls             map[string]bool
	audioInputStarted              bool
	pendingSpeculativeText         string

	// background tracks the session work run off the frame path: a handover, a
	// reset after an error, closing a session handed over.
	background sync.WaitGroup
}

// New builds a Nova Sonic service.
func New(cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	audio := DefaultAudioConfig()
	if cfg.AudioConfig != nil {
		audio = *cfg.AudioConfig
	}
	s := &Service{
		cfg:                cfg,
		live:               newSettings(cfg),
		audio:              audio,
		tools:              cfg.Tools,
		adapter:            &novasonicadapter.Adapter{},
		newClient:          newBedrockClient,
		triggerAudio:       readyAudio(),
		completedToolCalls: map[string]bool{},
	}
	params := DefaultSessionContinuationParams()
	if cfg.SessionContinuation != nil {
		params = *cfg.SessionContinuation
	}
	s.sc = newSessionContinuation(params, s)
	s.sc.handoff = func() { s.background.Go(func() { s.runHandoff(context.Background()) }) }
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New("NovaSonic", s, llm.WithContinuousGeneration())
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

// CanGenerateMetrics reports that this service reports usage metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// ServiceMetadataFrame describes this service to the pipeline as a realtime
// one. It reports no turns, so it recommends no turn strategies, and warns that
// processors needing turn frames get none without local turn detection.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	s.WarnIfRealtimeServiceEmitsNoTurnFrames(false)
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	return f
}

// AcceptsIntermediateFunctionCallResults reports that a tool's intermediate
// results reach the model: Nova Sonic takes one tool result per call, so an
// intermediate one goes in as text beside the call.
func (s *Service) AcceptsIntermediateFunctionCallResults() bool { return true }

// --- lifecycle ---

// Setup sets the service up and opens the session, ahead of the pipeline
// starting. The session is set up once the first conversation arrives.
func (s *Service) Setup(ctx context.Context, setup processor.Setup) error {
	if err := s.Base.Setup(ctx, setup); err != nil {
		return err
	}
	s.mu.Lock()
	s.wantsConnection = true
	s.mu.Unlock()
	s.startConnecting(ctx)
	return nil
}

// Cleanup stops wanting a session and disconnects.
func (s *Service) Cleanup(ctx context.Context) error {
	err := s.Base.Cleanup(ctx)
	s.teardown(ctx)
	s.background.Wait()
	return err
}

// teardown stops wanting a session and disconnects. It is safe to call more
// than once.
func (s *Service) teardown(ctx context.Context) {
	s.mu.Lock()
	s.wantsConnection = false
	s.mu.Unlock()
	s.disconnect(ctx)
}

// ResetConversation starts the session over with the conversation kept: the
// response in progress is closed, the session is disconnected, and a new one is
// opened and seeded with the conversation.
func (s *Service) ResetConversation(ctx context.Context) {
	slog.DebugContext(ctx, "resetting the conversation", "service", s.Name())
	s.mu.Lock()
	convo := s.convo
	responding := s.assistantResponding
	if convo != nil && responding {
		s.assistantResponding = false
	}
	s.mu.Unlock()
	if convo == nil {
		slog.WarnContext(ctx, "the conversation was reset before an initial conversation was received; "+
			"nothing to reset", "service", s.Name())
		return
	}
	if responding {
		s.reportAssistantResponseEnded(ctx)
	}
	s.disconnect(ctx)
	s.startConnecting(ctx)
	s.handleContext(ctx, convo)
}

// startConnecting opens a session and starts it; it is set up once the first
// conversation is there as well. A failure is reported and the session torn
// down.
func (s *Service) startConnecting(ctx context.Context) {
	slog.InfoContext(ctx, "connecting", "service", s.Name())
	s.mu.Lock()
	if s.client != nil {
		// A client means connected or connecting.
		s.mu.Unlock()
		return
	}
	s.promptName = uuid.NewString()
	s.audioContentName = uuid.NewString()
	s.mu.Unlock()

	if err := s.connect(ctx); err != nil {
		s.PushError(ctx, fmt.Sprintf("Initialization error: %v", err), err, false)
		s.disconnect(ctx)
		return
	}
	s.mu.Lock()
	s.readyToSendContext = true
	s.mu.Unlock()
	s.finishConnectingIfContextAvailable(ctx)
}

// connect builds the client, opens the stream and starts the session.
func (s *Service) connect(ctx context.Context) error {
	client, err := s.createClient(ctx)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.client = client
	s.mu.Unlock()
	stream, err := s.openStream(ctx, client)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.stream = stream
	s.mu.Unlock()
	return sendEvent(ctx, stream, s.sessionStartEvent())
}

// finishConnectingIfContextAvailable sets the session up once both the session
// is started and the first conversation has arrived: the tools, the system
// instruction, the history, the audio block, and the last user message as the
// one to answer. Then it starts receiving.
func (s *Service) finishConnectingIfContextAvailable(ctx context.Context) {
	s.mu.Lock()
	convo, ready := s.convo, s.readyToSendContext
	s.mu.Unlock()
	if convo == nil || !ready {
		return
	}
	slog.InfoContext(ctx, "finishing connecting (setting up the session)", "service", s.Name())

	// The results already in the conversation were produced before this
	// session existed: they are recorded as known rather than sent.
	s.processCompletedFunctionCalls(ctx, false)

	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{
		SystemInstruction: s.live.SystemInstruction.Or(""),
		ServiceTools:      s.tools,
	})
	if err != nil {
		s.PushError(ctx, "nova sonic: converting the conversation", err, false)
		return
	}

	// The tools: the conversation's own when it has any, else the service's,
	// with the built-in tools riding along either way.
	slog.DebugContext(ctx, "using tools", "service", s.Name(), "tools", len(params.Tools))
	s.sendPromptStartEvent(ctx, params.Tools)

	slog.DebugContext(ctx, "using the system instruction", "service", s.Name(),
		"instruction", params.SystemInstruction)
	if params.SystemInstruction != "" {
		s.sendTextEvent(ctx, params.SystemInstruction, roleSystem, false)
	}

	// The history, except a last message from the user, which is sent as the
	// one to answer once the audio block is open.
	var lastUser *novasonicadapter.HistoryMessage
	for i, m := range params.Messages {
		if i == len(params.Messages)-1 && m.Role == roleUser {
			lastUser = &params.Messages[i]
			continue
		}
		s.sendTextEvent(ctx, m.Text, m.Role, false)
	}
	s.sendAudioInputStartEvent(ctx)
	if lastUser != nil {
		s.sendTextEvent(ctx, lastUser.Text, lastUser.Role, true)
	}

	s.mu.Lock()
	stream := s.stream
	done := make(chan struct{})
	s.receiveDone = done
	s.connectedTime = time.Now()
	connected := s.connectedTime
	triggering := s.triggeringAssistantResponse
	s.mu.Unlock()
	go s.receive(context.WithoutCancel(ctx), stream, done)
	slog.InfoContext(ctx, "finished connecting", "service", s.Name())

	s.sc.setConnected(connected)
	// The history the session was seeded with predates it, so no final text
	// will report it: it is seeded into the continuation's history directly.
	for _, m := range params.Messages {
		s.sc.seedHistory(string(m.Role), m.Text)
	}
	s.sc.startMonitor()

	if triggering {
		s.sendAssistantResponseTrigger(ctx)
	}
}

// disconnect ends the session: the prompt and the session are closed, the
// stream is closed, and the session's state starts over. It leaves whether a
// session is wanted alone, so a reset can open the next one. It is safe to call
// more than once.
func (s *Service) disconnect(ctx context.Context) {
	slog.InfoContext(ctx, "disconnecting", "service", s.Name())
	s.mu.Lock()
	s.disconnecting = true
	client, stream, done := s.client, s.stream, s.receiveDone
	s.mu.Unlock()

	if client != nil {
		s.sendSessionEndEvents(ctx)
	}
	s.mu.Lock()
	s.client = nil
	s.convo = nil
	s.stream = nil
	s.mu.Unlock()
	if stream != nil {
		_ = stream.Close()
	}
	if done != nil {
		select {
		case <-done:
		case <-time.After(receiveWaitTimeout):
		}
	}

	s.mu.Lock()
	s.receiveDone = nil
	s.promptName = ""
	s.audioContentName = ""
	s.content = nil
	s.assistantResponding = false
	s.readyToSendContext = false
	s.triggeringAssistantResponse = false
	s.waitingForTriggerTranscription = false
	s.disconnecting = false
	s.connectedTime = time.Time{}
	s.userTextBuffer = ""
	s.completedToolCalls = map[string]bool{}
	s.audioInputStarted = false
	s.pendingSpeculativeText = ""
	s.mu.Unlock()

	s.sc.stopMonitor()
	s.sc.cleanupNextSession()
	s.sc.setDisconnected()
	slog.InfoContext(ctx, "finished disconnecting", "service", s.Name())
}

// --- frames ---

// ProcessFrame takes the conversation, the user's audio and appended messages,
// and ends the session with the pipeline. Every frame travels on.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if in, ok := f.(frames.InputAudioFrame); ok && dir == processor.Downstream {
		// Only the processor bookkeeping runs here: the LLM base would forward
		// the frame ahead of the audio being sent.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		data := in.AudioData()
		s.handleInputAudio(ctx, data.Audio, data.SampleRate)
		return s.PushFrame(ctx, f, dir)
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMContextFrame:
		// The base holds a conversation back from a service generating
		// continuously, so this is where it travels on.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMMessagesAppendFrame:
		s.handleMessagesAppend(ctx, fr)
	case *frames.EndFrame, *frames.CancelFrame:
		s.teardown(ctx)
	}
	return nil
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

// handleContext takes the conversation the pipeline reported. The first one
// lets the session be set up; a later one carries the results of the tool calls
// the model asked for.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext) {
	s.mu.Lock()
	if s.disconnecting {
		s.mu.Unlock()
		return
	}
	first := s.convo == nil
	if first {
		s.convo = convo
	}
	s.mu.Unlock()
	if first {
		s.finishConnectingIfContextAvailable(ctx)
		return
	}
	s.processCompletedFunctionCalls(ctx, true)
}

// handleInputAudio sends the user's audio, at rate, to the session at the input
// rate the session declares, except while the assistant response trigger is
// being sent, and buffers it during a handover.
func (s *Service) handleInputAudio(ctx context.Context, audio []byte, rate int) {
	s.mu.Lock()
	triggering := s.triggeringAssistantResponse
	s.mu.Unlock()
	if triggering {
		return
	}
	audio = s.resampleInputAudio(audio, rate)
	s.sc.onAudioInput(audio)
	// Once a handover is under way the old session hears no more: the audio is
	// buffered above and replayed to the new session.
	if s.sc.inHandoff() {
		return
	}
	s.sendUserAudioEvent(ctx, audio)
}

// resampleInputAudio converts the user's audio, at rate, to the input rate the
// session declares. The audio is resampled as one stream, so a chunk carries on
// from the one before it; audio at another rate than the last chunk's starts a
// new stream. Audio whose rate is not known is sent as it is.
func (s *Service) resampleInputAudio(audio []byte, rate int) []byte {
	want := s.audio.InputSampleRate
	if rate <= 0 || rate == want {
		return audio
	}
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.inputResampler == nil || s.inputResampleFrom != rate {
		r, err := resample.New(rate, want, 1)
		if err != nil {
			slog.Warn("cannot resample the user's audio to the session's input rate, dropping it",
				"service", s.Name(), "from", rate, "to", want, "err", err)
			return nil
		}
		s.inputResampler, s.inputResampleFrom = r, rate
	}
	return s.inputResampler.Process(audio)
}

// resetInputResampler starts the resampled user audio afresh, for audio that
// does not carry on from what was sent before it.
func (s *Service) resetInputResampler() {
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.inputResampler != nil {
		s.inputResampler.Clear()
	}
}

// handleMessagesAppend puts appended messages into the session as text, for
// the model to take in: interactive when the append asks to run, which the
// model answers as it would the user speaking, and plain text otherwise, which
// it takes in without answering.
func (s *Service) handleMessagesAppend(ctx context.Context, f *frames.LLMMessagesAppendFrame) {
	for _, m := range f.Messages {
		if text := m.StandardText(); text != "" {
			s.sendTextEvent(ctx, text, roleUser, f.RunLLM)
		}
	}
}

// handleFunctionCallResult sends one result to the session as it is produced.
//
// A final result goes through the tool-result channel, which Nova Sonic always
// folds into what it says next. An intermediate one goes in as text beside the
// call: interactive when the model should say something about it, plain text
// for it to take in otherwise.
func (s *Service) handleFunctionCallResult(ctx context.Context, f *frames.FunctionCallResultFrame) {
	result := f.Result
	if isEmptyResult(result) {
		result = frames.ToolResultCompleted
	}
	if f.Properties.Final() {
		s.sendToolResult(ctx, f.ToolCallID, result)
		s.mu.Lock()
		s.completedToolCalls[f.ToolCallID] = true
		s.mu.Unlock()
		return
	}
	run := true
	switch {
	case f.Properties != nil && f.Properties.RunLLM != nil:
		run = *f.Properties.RunLLM
	case f.RunLLM != nil:
		run = *f.RunLLM
	}
	s.sendTextEvent(ctx, frames.NewAsyncToolIntermediateMessage(f.ToolCallID, result).Text, roleUser, run)
}

// isEmptyResult reports whether a result carries nothing: no text, or JSON for
// an empty value.
func isEmptyResult(result string) bool {
	switch result {
	case "", "null", "{}", "[]", `""`, "false", "0":
		return true
	}
	return false
}

// processCompletedFunctionCalls finds the tool results the model has not been
// given yet and, when send is set, hands each to the session.
//
// The results are read out of the conversation rather than taken from the
// handler directly, because that is where a call's result lands however it was
// produced.
func (s *Service) processCompletedFunctionCalls(ctx context.Context, send bool) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		return
	}
	for _, m := range convo.Messages() {
		if m.IsLLMSpecific() {
			continue
		}
		if async, ok := frames.ParseAsyncToolMessage(m); ok {
			// The started marker opens a call the model already waits on, and an
			// intermediate result was sent as text when it was produced; only
			// the final result settles the call.
			if async.Kind != frames.AsyncToolFinal || s.completed(async.ToolCallID) {
				continue
			}
			if send {
				s.sendToolResult(ctx, async.ToolCallID, async.Result)
			}
			s.complete(async.ToolCallID)
			continue
		}
		for _, r := range m.ToolResults {
			// A call still running, or canceled, has a placeholder standing in for
			// its result, which is not an answer.
			if r.Content == frames.ToolResultInProgress || r.Content == frames.ToolResultCanceled {
				continue
			}
			if r.ID == "" || s.completed(r.ID) {
				continue
			}
			if send {
				s.sendToolResult(ctx, r.ID, r.Content)
			}
			s.complete(r.ID)
		}
	}
}

// completed reports whether a call's result has been settled.
func (s *Service) completed(toolCallID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.completedToolCalls[toolCallID]
}

// complete records a call's result as settled.
func (s *Service) complete(toolCallID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.completedToolCalls[toolCallID] = true
}

// --- sending on the current session ---

// current is the current session's stream and names.
func (s *Service) current() (sessionStream, string, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stream, s.promptName, s.audioContentName
}

// sendPromptStartEvent opens the prompt on the current session.
func (s *Service) sendPromptStartEvent(ctx context.Context, tools []map[string]any) {
	stream, prompt, _ := s.current()
	if prompt == "" {
		return
	}
	s.reportSendError(ctx, s.sendPromptStart(ctx, tools, prompt, stream))
}

// sendAudioInputStartEvent opens the audio block on the current session.
func (s *Service) sendAudioInputStartEvent(ctx context.Context) {
	stream, prompt, content := s.current()
	if prompt == "" || content == "" {
		return
	}
	s.reportSendError(ctx, s.sendAudioInputStart(ctx, prompt, content, stream))
	s.mu.Lock()
	s.audioInputStarted = true
	s.mu.Unlock()
}

// sendTextEvent sends text on the current session. interactive text is sent
// during the audio, for the model to answer; the rest (the system instruction,
// the history) is sent ahead of it.
func (s *Service) sendTextEvent(ctx context.Context, text string, role novasonicadapter.Role, interactive bool) {
	stream, prompt, _ := s.current()
	if stream == nil || prompt == "" || text == "" {
		return
	}
	s.reportSendError(ctx, s.sendText(ctx, text, string(role), prompt, stream, interactive))
}

// sendUserAudioEvent sends a chunk of the user's audio on the current session,
// once its audio block is open.
func (s *Service) sendUserAudioEvent(ctx context.Context, audio []byte) {
	s.mu.Lock()
	stream, prompt, content, started := s.stream, s.promptName, s.audioContentName, s.audioInputStarted
	s.mu.Unlock()
	if stream == nil || !started || prompt == "" || content == "" {
		return
	}
	s.reportSendError(ctx, s.sendAudio(ctx, audio, prompt, content, stream))
}

// sendSessionEndEvents closes the prompt and the session.
func (s *Service) sendSessionEndEvents(ctx context.Context) {
	stream, prompt, _ := s.current()
	if stream == nil || prompt == "" {
		return
	}
	_ = sendEvent(ctx, stream, promptEnd(prompt))
	_ = sendEvent(ctx, stream, sessionEnd())
}

// sendToolResult hands one call's result to the session, as a block of its own
// naming the call it answers.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	stream, prompt, _ := s.current()
	if stream == nil || prompt == "" {
		return
	}
	slog.DebugContext(ctx, "sending a tool result to the session", "service", s.Name(),
		"tool_call_id", toolCallID)
	for _, ev := range toolResultEvents(prompt, toolCallID, result) {
		if err := sendEvent(ctx, stream, ev); err != nil {
			s.reportSendError(ctx, err)
			return
		}
	}
}

// reportSendError logs a failed send. The session's receive loop reports the
// session failing.
func (s *Service) reportSendError(ctx context.Context, err error) {
	if err != nil {
		slog.ErrorContext(ctx, "sending to the session failed", "service", s.Name(), "err", err)
	}
}

// setupParams is the system instruction and the tools the next session is set
// up with.
func (s *Service) setupParams(ctx context.Context) (string, []map[string]any) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		return "", nil
	}
	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{
		SystemInstruction: s.live.SystemInstruction.Or(""),
		ServiceTools:      s.tools,
	})
	if err != nil {
		slog.ErrorContext(ctx, "converting the conversation for the next session", "service", s.Name(), "err", err)
		return "", nil
	}
	return params.SystemInstruction, params.Tools
}

// runHandoff swaps the current session for the next one, which session
// continuation has opened and set up, and closes the old one in the background.
func (s *Service) runHandoff(ctx context.Context) {
	s.mu.Lock()
	oldStream, oldDone := s.stream, s.receiveDone
	oldPrompt, oldContent := s.promptName, s.audioContentName
	s.mu.Unlock()

	var connected time.Time
	ok := s.sc.executeHandoff(ctx, func(ns *nextSession) {
		done := make(chan struct{})
		s.mu.Lock()
		s.client = ns.client
		s.stream = ns.stream
		s.promptName = ns.promptName
		s.audioContentName = ns.audioContentName
		s.connectedTime = time.Now()
		connected = s.connectedTime
		s.audioInputStarted = true
		s.receiveDone = done
		s.mu.Unlock()
		go s.receive(context.WithoutCancel(ctx), ns.stream, done)
	})
	if !ok {
		return
	}
	s.sc.setConnected(connected)
	slog.InfoContext(ctx, "session continuation: swap complete, closing the old session in the background")
	s.background.Go(func() {
		s.sc.closeOldSession(ctx, oldStream, oldDone, oldPrompt, oldContent)
	})
}

// --- receiving ---

// receive reads one session's events until it ends. It is bound to its stream:
// after a handover the old session's loop stops on its own, so two loops never
// read the same stream. A session failing while it is still wanted is reported
// and the conversation reset.
func (s *Service) receive(ctx context.Context, stream sessionStream, done chan struct{}) {
	defer close(done)
	err := s.receiveEvents(ctx, stream)
	if err == nil {
		return
	}
	s.mu.Lock()
	disconnecting, current, wants := s.disconnecting, s.stream == stream, s.wantsConnection
	s.mu.Unlock()
	if disconnecting {
		return
	}
	if !current {
		slog.DebugContext(ctx, "session continuation: the old session's receive loop ended", "err", err)
		return
	}
	s.PushError(ctx, fmt.Sprintf("Error processing responses: %v", err), err, false)
	if wants {
		// Off this loop: the reset disconnects, which waits for this loop.
		s.background.Go(func() { s.ResetConversation(ctx) })
	}
}

// receiveEvents reads and handles events until the stream ends or fails, or
// another session takes over.
func (s *Service) receiveEvents(ctx context.Context, stream sessionStream) error {
	for {
		s.mu.Lock()
		disconnecting := s.disconnecting
		s.mu.Unlock()
		if stream == nil || disconnecting {
			return nil
		}
		data, err := stream.Recv()
		if err != nil {
			return err
		}
		s.mu.Lock()
		stale := s.stream != stream
		s.mu.Unlock()
		if stale {
			return nil
		}
		if data == nil {
			// The model has no more to send. Our own disconnect and a handover
			// are handled above, so the stream ended while it was still wanted.
			return errStreamEnded
		}
		var envelope struct {
			Event map[string]json.RawMessage `json:"event"`
		}
		if err := json.Unmarshal(data, &envelope); err != nil {
			return err
		}
		if err := s.handleEvent(ctx, envelope.Event); err != nil {
			return err
		}
	}
}

// handleEvent handles one event the model sent. The content of a response is
// delivered in blocks, one at a time: the user's transcript, a tool use, the
// text of what the model plans to say and of what it said, and its audio.
func (s *Service) handleEvent(ctx context.Context, ev map[string]json.RawMessage) error {
	switch {
	case ev["completionStart"] != nil:
		return nil
	case ev["contentStart"] != nil:
		return s.handleContentStart(ctx, ev["contentStart"])
	case ev["textOutput"] != nil:
		return s.handleTextOutput(ev["textOutput"])
	case ev["audioOutput"] != nil:
		return s.handleAudioOutput(ctx, ev["audioOutput"])
	case ev["toolUse"] != nil:
		return s.handleToolUse(ctx, ev["toolUse"])
	case ev["contentEnd"] != nil:
		return s.handleContentEnd(ctx, ev["contentEnd"])
	case ev["completionEnd"] != nil:
		if s.sc.onCompletionEnd() {
			s.background.Go(func() { s.runHandoff(ctx) })
		}
		return nil
	case ev["usageEvent"] != nil:
		return s.handleUsage(ctx, ev["usageEvent"])
	}
	return nil
}

// handleContentStart records the block being received. The assistant's first
// speculative text opens its response and closes the user's turn; its audio
// may set a handover going; the user's content may complete one.
func (s *Service) handleContentStart(ctx context.Context, raw json.RawMessage) error {
	var start struct {
		Type                  string `json:"type"`
		Role                  string `json:"role"`
		AdditionalModelFields string `json:"additionalModelFields"` //nolint:tagliatelle // Nova Sonic wire field
	}
	if err := json.Unmarshal(raw, &start); err != nil {
		return err
	}
	stage := ""
	if start.AdditionalModelFields != "" {
		var fields struct {
			GenerationStage string `json:"generationStage"` //nolint:tagliatelle // Nova Sonic wire field
		}
		if err := json.Unmarshal([]byte(start.AdditionalModelFields), &fields); err != nil {
			return err
		}
		stage = fields.GenerationStage
	}
	s.mu.Lock()
	s.content = &currentContent{typ: start.Type, role: start.Role, textStage: stage}
	opensResponse := start.Role == string(roleAssistant) && start.Type == contentTypeTxt &&
		stage == stageSpeculative && !s.assistantResponding
	if opensResponse {
		s.assistantResponding = true
	}
	s.mu.Unlock()

	switch {
	case opensResponse:
		s.reportUserTranscriptionEnded(ctx)
		s.reportAssistantResponseStarted(ctx)
	case start.Role == string(roleAssistant) && start.Type == contentTypeAudio:
		// The assistant starting to speak is the moment to set a handover going,
		// once the session is past the threshold.
		s.sc.onAssistantAudioStarted(ctx)
	case start.Role == string(roleUser):
		if s.sc.onUserContentStarted() {
			s.background.Go(func() { s.runHandoff(ctx) })
		}
	}
	return nil
}

// handleTextOutput records the text of the block being received. A block
// carries one text.
func (s *Service) handleTextOutput(raw json.RawMessage) error {
	var out struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return err
	}
	s.mu.Lock()
	content := s.content
	if content != nil {
		content.text = out.Content
	}
	s.mu.Unlock()
	if content == nil {
		return nil
	}
	s.sc.onTextOutput(content.role, content.textStage)
	return nil
}

// handleAudioOutput pushes a chunk of the model's audio.
func (s *Service) handleAudioOutput(ctx context.Context, raw json.RawMessage) error {
	s.mu.Lock()
	receiving := s.content != nil
	s.mu.Unlock()
	if !receiving {
		return nil
	}
	var out struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return err
	}
	pcm, err := base64.StdEncoding.DecodeString(out.Content)
	if err != nil {
		return err
	}
	_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, s.audio.OutputSampleRate, s.audio.OutputChannelCount),
		processor.Downstream)
	return nil
}

// unhandledFunctionError reports the model calling a function no handler claims.
type unhandledFunctionError struct{ name string }

// Error implements error.
func (e *unhandledFunctionError) Error() string {
	return fmt.Sprintf("the LLM tried to call a function named %q, "+
		"but there isn't a callback registered for that function", e.name)
}

// handleToolUse runs the call the model asked for. The model calling a tool
// closes the user's turn.
func (s *Service) handleToolUse(ctx context.Context, raw json.RawMessage) error {
	s.mu.Lock()
	receiving, convo := s.content != nil, s.convo
	s.mu.Unlock()
	if !receiving || convo == nil {
		return nil
	}
	s.reportUserTranscriptionEnded(ctx)

	var use struct {
		ToolName  string `json:"toolName"`  //nolint:tagliatelle // Nova Sonic wire field
		ToolUseID string `json:"toolUseId"` //nolint:tagliatelle // Nova Sonic wire field
		Content   string `json:"content"`
	}
	if err := json.Unmarshal(raw, &use); err != nil {
		return err
	}
	if !json.Valid([]byte(use.Content)) {
		return fmt.Errorf("%w: a call to %s, %q", errArgumentsNotJSON, use.ToolName, use.Content)
	}
	if !s.HasFunction(use.ToolName) {
		return &unhandledFunctionError{name: use.ToolName}
	}
	call := frames.ToolCall{ID: use.ToolUseID, Name: use.ToolName, Args: json.RawMessage(use.Content)}
	if err := s.RunFunctionCalls(ctx, convo, []frames.ToolCall{call}); err != nil {
		slog.ErrorContext(ctx, "running a tool call failed", "service", s.Name(),
			"function", use.ToolName, "err", err)
	}
	return nil
}

// handleContentEnd acts on the end of a block. The assistant's speculative text
// is reported as it ends, its final text may complete a handover, and its audio
// ending closes the response. A block cut off by the user is a barge-in. The
// user's final text is their transcript.
func (s *Service) handleContentEnd(ctx context.Context, raw json.RawMessage) error {
	var end struct {
		StopReason string `json:"stopReason"` //nolint:tagliatelle // Nova Sonic wire field
	}
	if err := json.Unmarshal(raw, &end); err != nil {
		return err
	}
	s.mu.Lock()
	content := s.content
	s.content = nil
	s.mu.Unlock()
	if content == nil {
		return nil
	}

	switch {
	case content.role == string(roleAssistant) && content.typ == contentTypeTxt:
		s.handleAssistantTextEnd(ctx, content, end.StopReason)
	case content.role == string(roleAssistant) && content.typ == contentTypeAudio:
		// The text spoken goes out once all its audio has.
		s.reportTTSText(ctx)
		if end.StopReason == stopReasonEndTurn || end.StopReason == stopReasonInterrupted {
			// Either way there is no more audio for this turn.
			if end.StopReason == stopReasonInterrupted {
				// Nova Sonic reports a barge-in no other way: the interruption
				// goes upstream, so the assistant aggregator marks the message
				// interrupted, and downstream, so the output transport stops
				// playing. It goes ahead of the response's end, so the
				// aggregator handles it before the turn closes.
				_ = s.BroadcastInterruption(ctx)
			}
			s.mu.Lock()
			s.assistantResponding = false
			s.mu.Unlock()
			s.reportAssistantResponseEnded(ctx)
		}
	case content.role == string(roleUser) && content.typ == contentTypeTxt && content.textStage == stageFinal:
		s.reportUserTranscriptionTextAdded(content.text)
		s.sc.onContentEndUserFinalText(content.text)
	}
	return nil
}

// handleAssistantTextEnd acts on the end of a block of the assistant's text.
func (s *Service) handleAssistantTextEnd(ctx context.Context, content *currentContent, stopReason string) {
	if stopReason != stopReasonInterrupted {
		if content.textStage == stageSpeculative {
			s.reportLLMText(ctx, content.text)
		}
		if content.textStage == stageFinal && s.sc.onContentEndAssistantFinalText(content.text) {
			s.background.Go(func() { s.runHandoff(ctx) })
		}
		return
	}
	// Final text cut off is the canonical barge-in. The audio block usually
	// closed the response already, but the output transport is still playing
	// what it buffered, so the interruption is broadcast regardless.
	_ = s.BroadcastInterruption(ctx)
	s.mu.Lock()
	responding := s.assistantResponding
	s.assistantResponding = false
	s.mu.Unlock()
	if responding {
		// No audio block will end to close it.
		s.reportAssistantResponseEnded(ctx)
	}
	if s.sc.onContentEndTextInterrupted() {
		s.background.Go(func() { s.runHandoff(ctx) })
	}
}

// handleUsage reports the tokens one usage event accounts for. The delta is
// reported rather than the running total, so usage stays incremental per event
// as the other realtime services report it.
func (s *Service) handleUsage(ctx context.Context, raw json.RawMessage) error {
	var usage struct {
		Details struct {
			Delta usageDelta `json:"delta"`
		} `json:"details"`
	}
	if err := json.Unmarshal(raw, &usage); err != nil {
		return err
	}
	delta := usage.Details.Delta
	// The service splits each direction into speech and text, so the prompt and
	// completion counts are the two added together.
	prompt := delta.Input.SpeechTokens + delta.Input.TextTokens
	completion := delta.Output.SpeechTokens + delta.Output.TextTokens
	if prompt == 0 && completion == 0 {
		return nil
	}
	_ = s.PushTokenUsage(ctx, frames.LLMTokenUsage{
		PromptTokens:      prompt,
		CompletionTokens:  completion,
		TotalTokens:       prompt + completion,
		InputAudioTokens:  new(delta.Input.SpeechTokens),
		OutputAudioTokens: new(delta.Output.SpeechTokens),
		InputTextTokens:   new(delta.Input.TextTokens),
		OutputTextTokens:  new(delta.Output.TextTokens),
	})
	return nil
}

// usageDelta is the token accounting one usage event adds, split by direction
// and, within each, by modality.
type usageDelta struct {
	Input  usageTokens `json:"input"`
	Output usageTokens `json:"output"`
}

// usageTokens is one direction's accounting for a usage event.
type usageTokens struct {
	SpeechTokens int64 `json:"speechTokens"` //nolint:tagliatelle // Nova Sonic wire field
	TextTokens   int64 `json:"textTokens"`   //nolint:tagliatelle // Nova Sonic wire field
}

// --- reporting the assistant's response ---

// reportAssistantResponseStarted opens the response, and the speech that
// stands in for a TTS service's.
func (s *Service) reportAssistantResponseStarted(ctx context.Context) {
	slog.DebugContext(ctx, "assistant response started", "service", s.Name())
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = s.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream)
}

// reportLLMText pushes the assistant's speculative text, which arrives ahead of
// each stretch of audio, for display as it is said. It is kept out of the
// conversation; the TTSTextFrame the conversation records it from goes out
// when the audio has, so it lines up with what was played.
func (s *Service) reportLLMText(ctx context.Context, text string) {
	slog.DebugContext(ctx, "assistant speculative text", "service", s.Name(), "text", text)
	llmText := frames.NewLLMTextFrame(text)
	llmText.AppendToContext = false
	_ = s.PushFrame(ctx, llmText, processor.Downstream)
	aggregated := frames.NewAggregatedTextFrame(text, frames.TextTypeSentence)
	aggregated.AppendToContext = false
	_ = s.PushFrame(ctx, aggregated, processor.Downstream)
	s.mu.Lock()
	s.pendingSpeculativeText = text
	s.mu.Unlock()
}

// reportTTSText pushes the text whose audio has been sent.
func (s *Service) reportTTSText(ctx context.Context) {
	s.mu.Lock()
	text := s.pendingSpeculativeText
	s.pendingSpeculativeText = ""
	s.mu.Unlock()
	if text == "" {
		return
	}
	ttsText := frames.NewTTSTextFrame(text, frames.TextTypeSentence)
	ttsText.IncludesInterFrameSpaces = true
	_ = s.PushFrame(ctx, ttsText, processor.Downstream)
}

// reportAssistantResponseEnded closes the response and its speech.
func (s *Service) reportAssistantResponseEnded(ctx context.Context) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		return
	}
	slog.DebugContext(ctx, "assistant response ended", "service", s.Name())
	_ = s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	_ = s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
}

// --- reporting the user's transcript ---

// reportUserTranscriptionTextAdded buffers the user's words. They arrive in
// large chunks far apart, so they are held here until the turn ends rather than
// left to the user aggregator, which would split them across messages.
func (s *Service) reportUserTranscriptionTextAdded(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.convo == nil {
		return
	}
	if s.userTextBuffer != "" {
		s.userTextBuffer += " " + text
	} else {
		s.userTextBuffer = text
	}
}

// reportUserTranscriptionEnded sends the user's words upstream to the user
// aggregator as one transcript. When they are the transcript of the assistant
// response trigger, they are wrapped in the user's turn starting and stopping,
// so the user aggregator does not take them for an interruption.
func (s *Service) reportUserTranscriptionEnded(ctx context.Context) {
	s.mu.Lock()
	text := s.userTextBuffer
	if s.convo == nil || text == "" {
		// Nothing was transcribed, as when the first turn was text the bot
		// answered.
		s.mu.Unlock()
		return
	}
	wrap := s.waitingForTriggerTranscription && normalizedTrigger(text)
	s.userTextBuffer = ""
	s.waitingForTriggerTranscription = false
	s.mu.Unlock()

	slog.DebugContext(ctx, "user transcription ended", "service", s.Name())
	if wrap {
		slog.DebugContext(ctx, "wrapping the assistant response trigger's transcript in the user's turn",
			"service", s.Name())
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
	}
	_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(text, "", frames.NowTimestamp()), processor.Upstream)
	if wrap {
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	}
}
