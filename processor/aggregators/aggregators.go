// Package aggregators assembles the conversation around an LLM. The user
// aggregator collects transcriptions into a user message and triggers the LLM;
// the assistant aggregator collects the streamed response into an assistant
// message. Both share one LLMContext, so the conversation accrues across turns.
//
// Place the user aggregator before the LLM and the assistant aggregator at the
// end of the pipeline:
//
//	pipeline.New(input, stt, agg.User(), llm, tts, output, agg.Assistant())
//
// By default the user turn ends when the STT service finalizes a transcription.
// With WithTurnTaking, the turn instead ends when a turntaking.Detector reports
// end-of-turn (a UserStoppedSpeakingFrame), gated on having a finalized
// transcript — so a Smart Turn model, not STT endpointing, decides when the bot
// responds. Add the turntaking.Detector right after the input transport:
//
//	pipeline.New(input, detector, stt, agg.User(), llm, tts, output, agg.Assistant())
package aggregators

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
)

// Pair is a user and assistant aggregator sharing one conversation context.
type Pair struct {
	context   *frames.LLMContext
	user      *UserAggregator
	assistant *AssistantAggregator
}

// Option configures an aggregator Pair.
type Option func(*options)

type options struct {
	turns     *turns.Config
	summarize *SummarizeConfig
}

// WithTurns drives the user turn from turn-taking strategies rather than from
// STT finalization: the LLM runs when the strategies say the turn ended.
//
// The strategies run inside the user aggregator, on the same frames and in the
// same order as the aggregation. That is what makes the turn's own transcript
// part of it: a turn ends because a transcript finalized, and were the decision
// made in a processor of its own, the end-of-turn frame would be a system frame
// racing ahead of the transcript that caused it and the user's last words would
// be dropped from the message the model is given.
func WithTurns(cfg turns.Config) Option {
	return func(o *options) { o.turns = &cfg }
}

// New builds a user/assistant aggregator pair around ctx.
func New(ctx *frames.LLMContext, opts ...Option) *Pair {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	return &Pair{
		context:   ctx,
		user:      newUser(ctx, o.turns),
		assistant: newAssistant(ctx, o.summarize),
	}
}

// User returns the user-side aggregator.
func (p *Pair) User() processor.Processor { return p.user }

// Assistant returns the assistant-side aggregator.
func (p *Pair) Assistant() processor.Processor { return p.assistant }

// Context returns the shared conversation context.
func (p *Pair) Context() *frames.LLMContext { return p.context }

// UserAggregator collects transcriptions into a user message and, when the
// user's turn ends, appends it to the context and triggers the LLM with an
// LLMContextFrame.
//
// When turn strategies are configured it also drives them, and the idle
// watchdog, from this same processor: every frame is folded into the
// aggregation first and only then handed to the controllers, so a turn that
// ends on a finalized transcript ends with that transcript already in the
// message.
type UserAggregator struct {
	*processor.Base
	context    *frames.LLMContext
	turnTaking bool

	turn *turns.UserTurnController
	idle *turns.UserIdleController

	muteStrategies []turns.MuteStrategy
	muteMu         sync.Mutex
	muted          bool

	mu           sync.Mutex
	aggregation  string
	turnComplete bool // turn taking: end-of-turn reported
}

func newUser(ctx *frames.LLMContext, cfg *turns.Config) *UserAggregator {
	u := &UserAggregator{context: ctx, turnTaking: cfg != nil}
	u.Base = processor.New("UserContextAggregator", u)
	if cfg == nil {
		return u
	}
	u.muteStrategies = cfg.MuteStrategies
	u.turn = turns.NewUserTurnController(cfg.Strategies, cfg.StopTimeout)
	u.idle = turns.NewUserIdleController(turns.IdleConfig{Timeout: cfg.IdleTimeout, Callback: cfg.OnIdle})
	u.turn.SetHooks(turns.ControllerHooks{
		Started:   u.onTurnStarted,
		Stopped:   u.onTurnStopped,
		Push:      func(ctx context.Context, f frames.Frame, dir processor.Direction) { _ = u.PushFrame(ctx, f, dir) },
		Broadcast: func(ctx context.Context, build func() frames.Frame) { _ = u.Broadcast(ctx, build) },
	})
	return u
}

// Setup wires the controllers.
func (u *UserAggregator) Setup(ctx context.Context, s processor.Setup) error {
	if err := u.Base.Setup(ctx, s); err != nil {
		return err
	}
	if u.turn != nil {
		u.turn.Setup(ctx)
		u.idle.Setup(ctx, u)
	}
	return nil
}

// Cleanup tears the controllers down.
func (u *UserAggregator) Cleanup(ctx context.Context) error {
	if u.turn != nil {
		u.turn.Cleanup()
		u.idle.Cleanup()
	}
	return u.Base.Cleanup(ctx)
}

// ProcessFrame collects transcriptions and triggers the LLM.
//
// The frame is handled first and only then given to the turn and idle
// controllers, so anything the frame contributes to the aggregation is already
// there when a strategy decides the turn ended on it.
func (u *UserAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := u.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	// A muted frame is dropped outright: the controllers must not see input the
	// user is not allowed to give.
	if u.turn != nil && u.suppressed(ctx, f) {
		return nil
	}
	err := u.handleFrame(ctx, f, dir)
	if u.turn != nil {
		u.turn.Process(f)
		u.idle.Process(f)
	}
	return err
}

// handleFrame folds one frame into the aggregation and forwards it.
func (u *UserAggregator) handleFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch fr := f.(type) {
	case *frames.UserStartedSpeakingFrame:
		// A new turn begins; drop any stale aggregation from a prior turn.
		if u.turnTaking {
			u.mu.Lock()
			u.aggregation = ""
			u.turnComplete = false
			u.mu.Unlock()
		}
		return u.PushFrame(ctx, f, dir)

	case *frames.UserStoppedSpeakingFrame:
		if err := u.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		if u.turnTaking {
			u.mu.Lock()
			u.turnComplete = true
			u.mu.Unlock()
			return u.maybeRun(ctx)
		}
		return nil

	case *frames.InterimTranscriptionFrame:
		// Interim (partial) speech is forwarded for downstream consumers (e.g.
		// RTVI) but not aggregated; the turn processor drives end-of-turn.
		return u.PushFrame(ctx, f, dir)

	case *frames.TranscriptionFrame:
		return u.handleTranscription(ctx, fr, dir)

	case *frames.LLMRunFrame:
		// Explicit trigger: run the LLM on the current context now (e.g. to make
		// the bot speak first), bypassing the turn-completion gating. The frame
		// is consumed and turned into the LLMContextFrame the LLM service runs.
		return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)

	case *frames.LLMMessagesAppendFrame, *frames.LLMMessagesUpdateFrame,
		*frames.LLMSetToolsFrame, *frames.LLMSetToolChoiceFrame:
		return u.handleContextUpdate(ctx, f, dir)

	default:
		return u.PushFrame(ctx, f, dir)
	}
}

// handleContextUpdate applies a frame that mutates the shared LLM context and
// then forwards it. Forwarding matters: a text LLM reads the context on its next
// run, but a realtime (speech-to-speech) service generates continuously and
// learns of a tool or message change only from the frame itself.
func (u *UserAggregator) handleContextUpdate(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	runLLM := false

	switch fr := f.(type) {
	case *frames.LLMMessagesAppendFrame:
		// The turn-completion re-prompt appends a message to the context before a
		// follow-up LLMRunFrame.
		u.appendMessages(fr.Messages)

	case *frames.LLMMessagesUpdateFrame:
		// Wholesale replacement of the conversation (restoring a saved session,
		// resetting the conversation).
		u.context.SetMessages(fr.Messages)
		runLLM = fr.RunLLM

	case *frames.LLMSetToolsFrame:
		u.context.SetTools(fr.Tools)

	case *frames.LLMSetToolChoiceFrame:
		u.context.SetToolChoice(fr.ToolChoice)
	}

	if err := u.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if runLLM {
		return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)
	}
	return nil
}

// appendMessages adds messages to the context (used by the turn-completion
// re-prompt).
func (u *UserAggregator) appendMessages(msgs []frames.Message) {
	for _, m := range msgs {
		if m.Role == frames.RoleAssistant {
			u.context.AddAssistantMessage(m.Text)
			continue
		}
		u.context.AddUserMessage(m.Text)
	}
}

func (u *UserAggregator) handleTranscription(
	ctx context.Context, fr *frames.TranscriptionFrame, dir processor.Direction,
) error {
	u.mu.Lock()
	if fr.Text != "" {
		if u.aggregation != "" {
			u.aggregation += " "
		}
		u.aggregation += fr.Text
	}
	u.mu.Unlock()

	// Forward the transcription so downstream processors (e.g. RTVI) see it.
	if err := u.PushFrame(ctx, fr, dir); err != nil {
		return err
	}

	if u.turnTaking {
		return u.maybeRun(ctx)
	}
	// Default: STT finalization marks the end of the user's turn.
	if fr.Finalized {
		return u.maybeRun(ctx)
	}
	return nil
}

// maybeRun commits the aggregated user message and triggers the LLM when the
// turn-completion conditions hold. With turn taking, the turn processor's
// end-of-turn is authoritative: the transcripts aggregated during the turn are
// committed then. It does not additionally require the STT's end-of-utterance
// flag, which some providers omit (dropping the turn). Without turn taking, a
// finalized transcript alone suffices.
func (u *UserAggregator) maybeRun(ctx context.Context) error {
	u.mu.Lock()
	ready := u.aggregation != ""
	if u.turnTaking {
		ready = ready && u.turnComplete
	}
	if !ready {
		u.mu.Unlock()
		return nil
	}
	text := u.aggregation
	u.aggregation = ""
	u.turnComplete = false
	u.mu.Unlock()

	u.context.AddUserMessage(text)
	return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)
}

// AssistantAggregator collects the LLM's streamed text into a single assistant
// message and appends it to the context when the response completes. If the
// response is interrupted (barge-in), the partial text gathered so far is
// committed so the context reflects what the bot actually said. The response
// fields are touched from both the process goroutine (text frames) and the
// input goroutine (the InterruptionFrame system frame), so they are
// mutex-guarded.
type AssistantAggregator struct {
	*processor.Base
	context *frames.LLMContext
	// summarize is non-nil when automatic context summarization is enabled; it
	// compacts older turns in the background once the context grows too large.
	summarize *summarizer

	mu          sync.Mutex
	aggregation string
	started     bool
	// spoken accumulates the words actually spoken this turn, in their original
	// written form, from the playback-aligned TTSTextFrames a word-timestamp TTS
	// service emits. It drives the assistant message when the TTS reports word
	// timings; spokenCommitted tracks whether that message has been written to
	// the context yet, so later words update it in place rather than appending.
	spoken          string
	spokenCommitted bool
	// Tool-call state for the current assistant turn. pendingIDs holds the
	// calls still awaiting a result; pendingResults collects results until all
	// have arrived and they can be written as one tool-result message.
	pendingResults []frames.ToolResult
	pendingIDs     map[string]bool
	// suppressRun is set when a result asks not to re-run (a stopped turn).
	suppressRun bool
}

func newAssistant(ctx *frames.LLMContext, sc *SummarizeConfig) *AssistantAggregator {
	a := &AssistantAggregator{context: ctx, pendingIDs: make(map[string]bool)}
	if sc != nil && sc.Summarizer != nil {
		a.summarize = newSummarizer(*sc)
	}
	a.Base = processor.New("AssistantContextAggregator", a)
	return a
}

// ProcessFrame collects LLM text into an assistant message.
func (a *AssistantAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		a.mu.Lock()
		a.started = true
		a.aggregation = ""
		a.spoken = ""
		a.spokenCommitted = false
		a.mu.Unlock()
	case *frames.LLMTextFrame:
		// When a word-timestamp TTS drives the turn it excludes its text from the
		// context (AppendToContext == false); the spoken words arrive as
		// TTSTextFrames instead. Otherwise (the default) accumulate as before.
		if !fr.AppendToContext {
			break
		}
		a.mu.Lock()
		if a.started {
			a.aggregation += fr.Text
		}
		a.mu.Unlock()
	case *frames.TTSTextFrame:
		a.handleSpoken(fr)
	case *frames.FunctionCallsStartedFrame:
		a.handleFunctionCallsStarted(fr)
	case *frames.FunctionCallResultFrame:
		if err := a.handleFunctionCallResult(ctx, fr); err != nil {
			return err
		}
	case *frames.LLMFullResponseEndFrame:
		a.commit()
	case *frames.TTSSpeakFrame:
		// Fixed bot speech becomes part of the conversation unless opted out.
		if fr.AppendToContext {
			a.context.AddAssistantMessage(fr.Text)
		}
	case *frames.InterruptionFrame:
		// The response was cut off; keep whatever the bot already said and
		// balance any tool calls that never got a result.
		a.commitInterrupted()
	}
	return a.PushFrame(ctx, f, dir)
}

// handleSpoken records one playback-aligned spoken word into the assistant
// turn. The word's original written form is appended to the running spoken text
// and the in-progress assistant message is kept up to date in the context, so
// that if the turn is interrupted the context already holds exactly the words
// spoken so far. Because the TTSTextFrames flow in step with audio playback,
// only the words already spoken have arrived at any given moment.
func (a *AssistantAggregator) handleSpoken(fr *frames.TTSTextFrame) {
	if !fr.AppendToContext {
		return
	}
	text := fr.Original()
	if text == "" {
		return
	}
	a.mu.Lock()
	if a.spoken != "" {
		a.spoken += " "
	}
	a.spoken += text
	spoken := a.spoken
	committed := a.spokenCommitted
	a.spokenCommitted = true
	a.mu.Unlock()

	if committed && a.context.ReplaceLastAssistantText(spoken) {
		return
	}
	a.context.AddAssistantMessage(spoken)
}

// handleFunctionCallsStarted writes the assistant turn that requested the tool
// calls — any preamble text plus the tool-use blocks — and records the calls as
// awaiting results.
func (a *AssistantAggregator) handleFunctionCallsStarted(fr *frames.FunctionCallsStartedFrame) {
	a.mu.Lock()
	text := a.aggregation
	a.aggregation = ""
	for _, c := range fr.Calls {
		a.pendingIDs[c.ID] = true
	}
	a.mu.Unlock()
	a.context.AddAssistantToolCalls(text, fr.Calls)
}

// handleFunctionCallResult buffers a tool result and, once every call from the
// assistant turn has one, writes them as a single tool-result message and
// re-triggers generation — unless a result asked to stop the turn. Writing the
// results before re-triggering keeps the tool calls balanced for the next
// inference.
func (a *AssistantAggregator) handleFunctionCallResult(ctx context.Context, fr *frames.FunctionCallResultFrame) error {
	a.mu.Lock()
	a.pendingResults = append(a.pendingResults, frames.ToolResult{
		ID:      fr.ToolCallID,
		Name:    fr.ToolName,
		Content: fr.Result,
		IsError: fr.IsError,
	})
	if !fr.RunLLM {
		a.suppressRun = true
	}
	delete(a.pendingIDs, fr.ToolCallID)
	var results []frames.ToolResult
	complete := len(a.pendingIDs) == 0
	if complete {
		results = a.pendingResults
		a.pendingResults = nil
	}
	runLLM := complete && !a.suppressRun
	if complete {
		a.suppressRun = false
	}
	a.mu.Unlock()

	if results != nil {
		a.context.AddToolResults(results)
	}
	if runLLM {
		return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
	}
	return nil
}

// commit appends the aggregated assistant message to the context, if any, and
// resets the response state.
func (a *AssistantAggregator) commit() {
	a.mu.Lock()
	text := a.aggregation
	a.aggregation = ""
	a.started = false
	a.mu.Unlock()
	if text != "" {
		a.context.AddAssistantMessage(text)
	}
	a.maybeSummarize()
}

// commitInterrupted closes out a turn cut off by an interruption. Any tool calls
// still awaiting a result get a synthetic error result so the assistant turn
// that requested them stays balanced (a tool-use block always has a matching
// tool-result), then any partial assistant text is committed. This keeps the
// context valid for the next turn.
func (a *AssistantAggregator) commitInterrupted() {
	a.mu.Lock()
	results := a.pendingResults
	a.pendingResults = nil
	for id := range a.pendingIDs {
		results = append(results, frames.ToolResult{ID: id, Content: "interrupted", IsError: true})
		delete(a.pendingIDs, id)
	}
	a.suppressRun = false
	text := a.aggregation
	a.aggregation = ""
	a.started = false
	// Word-timestamp path: the spoken text was written to the context live as
	// each word played, so it already holds exactly what was spoken before the
	// interruption. Just close the turn so the next one starts fresh.
	a.spoken = ""
	a.spokenCommitted = false
	a.mu.Unlock()
	if len(results) > 0 {
		a.context.AddToolResults(results)
	}
	if text != "" {
		a.context.AddAssistantMessage(text)
	}
	a.maybeSummarize()
}

// suppressed runs the mute strategies and reports whether this user-input frame
// should be dropped. It emits UserMute frames on a change of state.
func (u *UserAggregator) suppressed(ctx context.Context, f frames.Frame) bool {
	switch f.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
		// Lifecycle frames are never muted and must keep their ordering.
		return false
	}
	if len(u.muteStrategies) == 0 {
		return false
	}
	u.muteMu.Lock()
	defer u.muteMu.Unlock()

	should := false
	for _, m := range u.muteStrategies {
		if m.ShouldMute(f) { // call all, so each updates its state
			should = true
		}
	}
	if should != u.muted {
		u.muted = should
		if should {
			_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserMuteStartedFrame() })
		} else {
			_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserMuteStoppedFrame() })
		}
	}
	if !u.muted {
		return false
	}
	switch f.(type) {
	case *frames.InterruptionFrame, *frames.VADUserStartedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame, *frames.InputAudioRawFrame,
		*frames.InterimTranscriptionFrame, *frames.TranscriptionFrame:
		return true
	}
	return false
}

// Push implements turns.Emitter.
func (u *UserAggregator) Push(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	return u.PushFrame(ctx, f, dir)
}

// Broadcast, which turns.Emitter also requires, is promoted from processor.Base.

// onTurnStarted broadcasts the turn-start decision and barges in, and feeds the
// idle controller a synthetic user-started frame so it tracks the turn.
func (u *UserAggregator) onTurnStarted(ctx context.Context, params turns.UserTurnStartedParams) {
	u.mu.Lock()
	u.aggregation = ""
	u.turnComplete = false
	u.mu.Unlock()
	if params.EnableUserSpeakingFrames {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
	}
	u.idle.Process(frames.NewUserStartedSpeakingFrame())
	if params.EnableInterruptions {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewInterruptionFrame() })
	}
}

// onTurnStopped broadcasts the turn-stop decision, feeds the idle controller a
// synthetic user-stopped frame, and commits the turn.
//
// Committing here rather than on a received UserStoppedSpeakingFrame is the
// point of driving the turn from inside the aggregator: the frame that ended
// the turn has already been folded into the aggregation by the time this runs,
// so the user's last words are part of the message the model is given.
func (u *UserAggregator) onTurnStopped(ctx context.Context, params turns.UserTurnStoppedParams) {
	if params.EnableUserSpeakingFrames {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	}
	u.idle.Process(frames.NewUserStoppedSpeakingFrame())
	u.mu.Lock()
	u.turnComplete = true
	u.mu.Unlock()
	_ = u.maybeRun(ctx)
}

var _ turns.Emitter = (*UserAggregator)(nil)
