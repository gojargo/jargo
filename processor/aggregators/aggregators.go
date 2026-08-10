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
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/text"
)

// The placeholder contents a tool-result message holds while its call has not
// reported a result of its own.
const (
	// toolResultInProgress is written the moment a call starts, so the tool-use
	// block it answers is never left unanswered.
	toolResultInProgress = "IN_PROGRESS"
	// toolResultCancelled replaces the placeholder when the call is canceled.
	// The spelling is the protocol's, not prose.
	toolResultCancelled = "CANCELLED" //nolint:misspell // the literal written to the conversation
	// toolResultCompleted stands in for a call that finished having produced no
	// result of its own.
	toolResultCompleted = "COMPLETED"
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
func (p *Pair) User() *UserAggregator { return p.user }

// Assistant returns the assistant-side aggregator.
func (p *Pair) Assistant() *AssistantAggregator { return p.assistant }

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

	mu          sync.Mutex
	aggregation string
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
		Started:            u.onTurnStarted,
		Stopped:            u.onTurnStopped,
		InferenceTriggered: u.onInferenceTriggered,
		ResetAggregation:   u.onResetAggregation,
		Push: func(ctx context.Context, f frames.Frame, dir processor.Direction) {
			_ = u.PushFrame(ctx, f, dir)
		},
		Broadcast: func(ctx context.Context, build func() frames.Frame) {
			_ = u.Broadcast(ctx, build)
		},
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
	case *frames.EndFrame:
		// The session is over, so whatever the user said last is committed
		// rather than lost with the processor. It happens after the frame has
		// been forwarded, so the end of the pipeline is not held up behind it.
		if err := u.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return u.maybeRun(ctx)

	case *frames.CancelFrame:
		// Same, the other way round: a cancel tears the pipeline down at once, so
		// what is held is committed before the frame that stops everything.
		if err := u.maybeRun(ctx); err != nil {
			return err
		}
		return u.PushFrame(ctx, f, dir)

	case *frames.InterimTranscriptionFrame:
		// Consumed here, like the final transcript below: partial speech is not
		// aggregated either, and the turn strategies that read it run inside this
		// processor. What is downstream is the model, which is given the
		// conversation rather than the transcripts it was built from.
		return nil

	case *frames.TranscriptionFrame:
		return u.handleTranscription(ctx, fr)

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
		// follow-up LLMRunFrame; a conversation flow appends the node's objective
		// on entry.
		u.appendMessages(fr.Messages)
		runLLM = fr.RunLLM

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

// appendMessages adds messages to the context as they were written. The role
// each message carries is kept: a developer message is an out-of-band
// instruction to the model and reads differently from something the user said,
// and every adapter renders the distinction.
func (u *UserAggregator) appendMessages(msgs []frames.Message) {
	for _, m := range msgs {
		u.context.AddMessage(m)
	}
}

// handleTranscription folds one transcript into the aggregation. The frame is
// consumed rather than forwarded: what the user said reaches the model as the
// conversation, and a client is told about the transcript by the RTVI observer,
// which watches it where the transcription service pushes it rather than
// waiting for it to travel any further.
func (u *UserAggregator) handleTranscription(
	ctx context.Context, fr *frames.TranscriptionFrame,
) error {
	u.mu.Lock()
	if fr.Text != "" {
		if u.aggregation != "" {
			u.aggregation += " "
		}
		u.aggregation += fr.Text
	}
	u.mu.Unlock()

	if u.turnTaking {
		// A transcript only adds to the aggregation. What is aggregated is
		// committed when the turn controller says the turn is over, or when a stop
		// strategy says there is enough to answer: a transcript on its own is not
		// an end of turn, and one arriving after a turn closed is the beginning of
		// the next one rather than a second answer to the one that just ended.
		return nil
	}
	// Without turn taking, STT finalization marks the end of the user's turn.
	if fr.Finalized {
		return u.maybeRun(ctx)
	}
	return nil
}

// maybeRun commits the aggregated user message and triggers the LLM on it.
// Nothing is committed when the user has said nothing since the last time.
//
// With turn taking it is called only from the turn controller's hooks, which
// are the sole authority on when the aggregation becomes a message. Without turn
// taking a finalized transcript alone suffices.
func (u *UserAggregator) maybeRun(ctx context.Context) error {
	u.mu.Lock()
	if u.aggregation == "" {
		u.mu.Unlock()
		return nil
	}
	text := u.aggregation
	u.aggregation = ""
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

	mu sync.Mutex
	// aggregation is what the turn has said so far, in the order it was said:
	// the model's own text where that is what reaches the conversation, and the
	// playback-aligned words a word-timestamp TTS reports where it is not. The
	// two kinds space themselves differently, so each piece carries the answer
	// and the join respects it.
	aggregation []text.Part
	// turnStarted reports whether an assistant turn is open. A turn opens on the
	// model's response starting, or, for an utterance the service speaks with no
	// response around it, on the start of that speech.
	turnStarted bool
	// inProgress holds every tool call of the current turn that has not reported
	// a final result yet, keyed by tool call id. The value is nil between the
	// FunctionCallsStartedFrame that announced the call and the
	// FunctionCallInProgressFrame that starts it.
	inProgress map[string]*frames.FunctionCallInProgressFrame
	// userSpeaking and botSpeaking track who holds the floor. A tool result that
	// arrives while the user is speaking never re-runs generation, and one that
	// arrives while the bot is speaking defers it until the bot finishes.
	userSpeaking bool
	botSpeaking  bool
	// pushOnBotStopped records a deferred re-run, so several results arriving
	// while the bot speaks are answered by a single inference once it stops.
	pushOnBotStopped bool

	// Lifetime of the context-updated callbacks, which run off the frame path.
	// Cleanup cancels them and waits for them to return.
	taskCtx    context.Context
	taskCancel context.CancelFunc
	taskWG     sync.WaitGroup
}

func newAssistant(ctx *frames.LLMContext, sc *SummarizeConfig) *AssistantAggregator {
	a := &AssistantAggregator{
		context:    ctx,
		inProgress: make(map[string]*frames.FunctionCallInProgressFrame),
	}
	if sc != nil && sc.Summarizer != nil {
		a.summarize = newSummarizer(*sc)
	}
	a.Base = processor.New("AssistantContextAggregator", a)
	return a
}

// HasFunctionCallsInProgress reports whether any tool call of the current turn
// has yet to report a final result.
//
// It is what tells something waiting on the whole batch, rather than on one
// call, that the turn's calls are done: a caller acting on a tool result in its
// context-updated callback reads this to know whether it is the last one, since
// acting while siblings are still running would act on a half-finished turn.
func (a *AssistantAggregator) HasFunctionCallsInProgress() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.inProgress) > 0
}

// Setup opens the lifetime the context-updated callbacks run under.
func (a *AssistantAggregator) Setup(ctx context.Context, s processor.Setup) error {
	if err := a.Base.Setup(ctx, s); err != nil {
		return err
	}
	a.taskCtx, a.taskCancel = context.WithCancel(ctx)
	return nil
}

// Cleanup cancels the context-updated callbacks and waits for them to return.
func (a *AssistantAggregator) Cleanup(ctx context.Context) error {
	if a.taskCancel != nil {
		a.taskCancel()
	}
	a.taskWG.Wait()
	return a.Base.Cleanup(ctx)
}

// ProcessFrame collects LLM text into an assistant message.
func (a *AssistantAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		a.openTurn()
	case *frames.TTSStartedFrame:
		// Speech that will be recorded opens a turn of its own when none is open.
		// It is how an utterance the service speaks, with no response around it to
		// mark where the turn begins, still gets one, while deferring to a turn the
		// model already started.
		if fr.AppendToContext {
			a.openTurn()
		}
	case *frames.LLMTextFrame:
		// When a word-timestamp TTS drives the turn it excludes its text from the
		// context (AppendToContext == false); the spoken words arrive as
		// TTSTextFrames instead. Otherwise (the default) accumulate as before.
		a.handleText(&fr.TextFrame, fr.Text)
	case *frames.TTSTextFrame:
		a.handleText(&fr.TextFrame, fr.Original())
	case *frames.AggregatedTextFrame:
		// A unit that is not spoken at all, a code block held back from the
		// synthesizer, reaches the conversation as itself once the speech ahead of
		// it has been said.
		a.handleText(&fr.TextFrame, rawOrText(fr))
	case *frames.LLMAssistantPushAggregationFrame:
		// The end of an utterance that had no response around it to end.
		a.commit()
	case *frames.FunctionCallsStartedFrame, *frames.FunctionCallInProgressFrame,
		*frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame:
		// Consumed, not forwarded. This aggregator is where a tool call becomes
		// conversation, and it is the last processor in the pipeline, so there is
		// nothing beyond it to tell. Every other consumer is reached by the LLM
		// service broadcasting each of these frames upstream as well as down: the
		// idle watchdog and the mute strategies run inside the user aggregator, and
		// an RTVI processor sits between the LLM and the output.
		return a.handleFunctionCallFrame(ctx, f)
	case *frames.LLMFullResponseEndFrame:
		a.commit()
	case *frames.EndFrame, *frames.CancelFrame:
		// The session is over, so the turn is closed out where it stands rather
		// than lost with the processor.
		a.commit()
	case *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame:
		return a.handleSpeakingState(ctx, f, dir)
	case *frames.InterruptionFrame:
		// The response was cut off; keep whatever the bot already said. The tool
		// calls still running are left alone: each already has a placeholder
		// result in the context, so the turn is balanced as it stands, and the
		// LLM service resolves them by canceling the calls it registered to
		// cancel.
		a.commitInterrupted()
	}
	return a.PushFrame(ctx, f, dir)
}

// handleText folds one piece of the turn's text into the aggregation. original
// is what the piece contributes to the conversation, which for a spoken word is
// its written form rather than the pronunciation handed to the synthesizer.
//
// A piece that says it does not belong in the conversation is dropped here. That
// is how a turn driven by a word-timestamp TTS keeps the model's own text out:
// the text arrives saying so, and the words the synthesizer actually spoke
// arrive later saying the opposite, which is what makes an interruption leave
// exactly the words that were heard.
func (a *AssistantAggregator) handleText(fr *frames.TextFrame, original string) {
	if !fr.AppendToContext || fr.Text == "" {
		return
	}
	a.mu.Lock()
	a.aggregation = append(a.aggregation,
		text.Part{Text: original, IncludesInterPartSpaces: fr.IncludesInterFrameSpaces})
	a.mu.Unlock()
}

// rawOrText returns what an aggregated unit contributes to the conversation: its
// original written form where it has one, such as a code block with its
// delimiters, and its text otherwise.
func rawOrText(fr *frames.AggregatedTextFrame) string {
	if fr.RawText != "" {
		return fr.RawText
	}
	return fr.Text
}

// handleFunctionCallFrame applies one frame of a tool call's life to the
// conversation. The four arrive on two different goroutines: the two system
// frames as they are queued, the other two in turn behind the frames of the turn
// they belong to.
func (a *AssistantAggregator) handleFunctionCallFrame(ctx context.Context, f frames.Frame) error {
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		a.handleFunctionCallsStarted(fr)
	case *frames.FunctionCallInProgressFrame:
		a.handleFunctionCallInProgress(fr)
	case *frames.FunctionCallResultFrame:
		return a.handleFunctionCallResult(ctx, fr)
	case *frames.FunctionCallCancelFrame:
		a.handleFunctionCallCancel(fr)
	}
	return nil
}

// handleFunctionCallsStarted records the announced calls as awaiting results. It
// writes nothing to the context: each call's own FunctionCallInProgressFrame
// writes its tool-use block and the placeholder answering it, which is what keeps
// the conversation balanced at every instant rather than only once the turn ends.
func (a *AssistantAggregator) handleFunctionCallsStarted(fr *frames.FunctionCallsStartedFrame) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, c := range fr.Calls {
		a.inProgress[c.ID] = nil
	}
}

// handleFunctionCallInProgress writes the assistant message requesting this one
// call, immediately followed by the message that answers it: a placeholder
// result for an ordinary call, or an async-tool started message for a call the
// model does not wait on. The pair is written together, so no inference can ever
// read a tool-use block with nothing answering it.
func (a *AssistantAggregator) handleFunctionCallInProgress(fr *frames.FunctionCallInProgressFrame) {
	a.context.AddAssistantToolCall(frames.ToolCall{ID: fr.ToolCallID, Name: fr.ToolName, Args: fr.Args})
	if fr.CancelOnInterruption {
		a.context.AddToolResult(frames.ToolResult{
			ID:      fr.ToolCallID,
			Name:    fr.ToolName,
			Content: toolResultInProgress,
		})
	} else {
		a.context.AddMessage(frames.NewAsyncToolStartedMessage(fr.ToolCallID))
	}
	a.mu.Lock()
	a.inProgress[fr.ToolCallID] = fr
	a.mu.Unlock()
}

// handleFunctionCallResult applies one tool result to the context and decides
// whether generation re-runs on it. A final result replaces the call's
// placeholder in place, which is what keeps it next to the tool call it answers
// however long the call took and whatever was appended meanwhile.
func (a *AssistantAggregator) handleFunctionCallResult(
	ctx context.Context, fr *frames.FunctionCallResultFrame,
) error {
	a.mu.Lock()
	started, running := a.inProgress[fr.ToolCallID]
	a.mu.Unlock()
	if !running {
		slog.WarnContext(ctx, "tool result for a call that is not running",
			"processor", a.Name(), "tool", fr.ToolName, "tool_call_id", fr.ToolCallID)
		return nil
	}

	groupID := ""
	if started != nil {
		groupID = started.GroupID
	}
	props := fr.Properties
	if props.Final() {
		a.finishFunctionCall(fr, started)
	} else {
		a.recordIntermediateResult(ctx, fr)
	}

	if a.shouldRunLLM(fr, props, groupID) {
		a.mu.Lock()
		speaking := a.userSpeaking
		a.mu.Unlock()
		if !speaking {
			if err := a.maybePushContextAfterFunctionResult(ctx); err != nil {
				return err
			}
		}
	}

	// The callback is told the context now holds the result. It runs off the
	// frame path so that whatever it does cannot hold up the pipeline.
	if props != nil && props.OnContextUpdated != nil {
		a.runContextUpdated(props.OnContextUpdated)
	}
	return nil
}

// shouldRunLLM decides whether this result re-runs generation. An explicit
// choice, from the properties or the frame, wins. Otherwise a call that belongs
// to a group re-runs only as the last of its group to report, so a turn that
// requested several calls answers once rather than once per call; a call with no
// group re-runs on its own.
func (a *AssistantAggregator) shouldRunLLM(
	fr *frames.FunctionCallResultFrame, props *frames.FunctionCallResultProperties, groupID string,
) bool {
	if fr.Result == "" {
		return false
	}
	switch {
	case props != nil && props.RunLLM != nil:
		return *props.RunLLM
	case fr.RunLLM != nil:
		return *fr.RunLLM
	case groupID != "":
		return !a.groupStillRunning(groupID, fr.ToolCallID)
	default:
		return true
	}
}

// groupStillRunning reports whether a call of groupID other than exclude has yet
// to report. exclude is the call reporting now, which an intermediate result
// leaves in the map.
func (a *AssistantAggregator) groupStillRunning(groupID, exclude string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	for id, started := range a.inProgress {
		if started != nil && started.GroupID == groupID && id != exclude {
			return true
		}
	}
	return false
}

// finishFunctionCall applies a call's final result and stops tracking it. An
// ordinary call's placeholder is replaced in place; a call the model did not wait
// on gets an async-tool final message appended instead, since by now the
// conversation has moved past the point its placeholder sits at.
func (a *AssistantAggregator) finishFunctionCall(
	fr *frames.FunctionCallResultFrame, started *frames.FunctionCallInProgressFrame,
) {
	// A result that arrives before the call's in-progress frame leaves nothing to
	// read the registration off, so treat it as an ordinary call: that is the
	// shape whose placeholder is waiting to be replaced.
	async := started != nil && !started.CancelOnInterruption

	a.mu.Lock()
	delete(a.inProgress, fr.ToolCallID)
	a.mu.Unlock()

	result := fr.Result
	if result == "" {
		result = toolResultCompleted
	}
	if async {
		a.context.AddMessage(frames.NewAsyncToolFinalMessage(fr.ToolCallID, result))
		return
	}
	a.context.UpdateToolResult(fr.ToolCallID, result)
}

// recordIntermediateResult appends an async-tool intermediate message, leaving
// the call running. Only a call the model does not wait on reports these.
func (a *AssistantAggregator) recordIntermediateResult(
	ctx context.Context, fr *frames.FunctionCallResultFrame,
) {
	if fr.Result == "" {
		slog.WarnContext(ctx, "intermediate tool result with no result",
			"processor", a.Name(), "tool", fr.ToolName, "tool_call_id", fr.ToolCallID)
		return
	}
	a.context.AddMessage(frames.NewAsyncToolIntermediateMessage(fr.ToolCallID, fr.Result))
}

// handleFunctionCallCancel marks a canceled call's placeholder result canceled,
// so the tool-use block it answers stays answered. A call the model does not wait
// on is left alone: it survives the interruption that canceled the others.
func (a *AssistantAggregator) handleFunctionCallCancel(fr *frames.FunctionCallCancelFrame) {
	a.mu.Lock()
	started, running := a.inProgress[fr.ToolCallID]
	if !running || started == nil || !started.CancelOnInterruption {
		a.mu.Unlock()
		return
	}
	delete(a.inProgress, fr.ToolCallID)
	a.mu.Unlock()
	a.context.UpdateToolResult(fr.ToolCallID, toolResultCancelled)
}

// maybePushContextAfterFunctionResult re-runs generation on the updated context,
// unless doing so now would produce an answer that another result is about to
// make stale, or one spoken over the bot's current turn.
//
// A cascaded LLM reads the new result out of the context frame; a realtime
// service reads it out of the context the same way, since neither is handed the
// result frame itself. The push is what carries the result to the model in both.
func (a *AssistantAggregator) maybePushContextAfterFunctionResult(ctx context.Context) error {
	if a.HasQueuedFrame(isFunctionCallResult) {
		// More results are already on their way. Leaving the push to the last of
		// them answers the whole batch with one inference instead of one each.
		return nil
	}

	a.mu.Lock()
	speaking := a.botSpeaking
	if speaking {
		// Deferred rather than dropped: BotStoppedSpeakingFrame pushes it. Results
		// arriving meanwhile accumulate in the context and share this one push, so
		// the bot answers once rather than talking over itself.
		a.pushOnBotStopped = true
	}
	a.mu.Unlock()
	if speaking {
		return nil
	}
	return a.pushContextFrame(ctx)
}

// isFunctionCallResult reports whether f is a tool result, for the queue check
// above.
func isFunctionCallResult(f frames.Frame) bool {
	_, ok := f.(*frames.FunctionCallResultFrame)
	return ok
}

// pushContextFrame runs generation on the current context, clearing any deferred
// push since this one covers it.
func (a *AssistantAggregator) pushContextFrame(ctx context.Context) error {
	a.mu.Lock()
	a.pushOnBotStopped = false
	a.mu.Unlock()
	return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
}

// pushDeferredContext runs the generation a tool result deferred while the bot
// was speaking. It stays deferred while the user is speaking, whose turn ends by
// running the LLM anyway.
func (a *AssistantAggregator) pushDeferredContext(ctx context.Context) error {
	a.mu.Lock()
	deferred := a.pushOnBotStopped && !a.userSpeaking
	a.mu.Unlock()
	if !deferred {
		return nil
	}
	return a.pushContextFrame(ctx)
}

// runContextUpdated runs a result's context-updated callback on a goroutine of
// its own, tracked so Cleanup can cancel it and wait for it.
func (a *AssistantAggregator) runContextUpdated(fn func(ctx context.Context) error) {
	taskCtx := a.taskCtx
	if taskCtx == nil {
		// Setup has not run, so there is no lifetime to attach to.
		return
	}
	a.taskWG.Go(func() {
		if err := fn(taskCtx); err != nil && taskCtx.Err() == nil {
			slog.Error("tool result context-updated callback", "processor", a.Name(), "err", err)
		}
	})
}

// handleSpeakingState tracks who holds the floor, which is what tells a tool
// result whether re-running generation now would talk over somebody. The bot
// falling silent is also when a deferred re-run finally happens.
func (a *AssistantAggregator) handleSpeakingState(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	a.mu.Lock()
	switch f.(type) {
	case *frames.UserStartedSpeakingFrame:
		a.userSpeaking = true
	case *frames.UserStoppedSpeakingFrame:
		a.userSpeaking = false
	case *frames.BotStartedSpeakingFrame:
		a.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		a.botSpeaking = false
	}
	a.mu.Unlock()

	if err := a.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, stopped := f.(*frames.BotStoppedSpeakingFrame); !stopped {
		return nil
	}
	return a.pushDeferredContext(ctx)
}

// openTurn marks an assistant turn open. Opening one that is already open
// changes nothing: the turn belongs to the response, not to each frame of it.
func (a *AssistantAggregator) openTurn() {
	a.mu.Lock()
	a.turnStarted = true
	a.mu.Unlock()
}

// commit closes the assistant turn out, writing what it said to the
// conversation as one message. A turn that was never opened writes nothing, so
// text arriving with no turn around it waits for one rather than landing on its
// own.
func (a *AssistantAggregator) commit() {
	a.mu.Lock()
	open := a.turnStarted
	a.turnStarted = false
	a.mu.Unlock()
	if !open {
		return
	}
	a.pushAggregation()
}

// pushAggregation writes what the turn has said so far to the conversation and
// empties the aggregation. Nothing is written when nothing was said.
func (a *AssistantAggregator) pushAggregation() {
	a.mu.Lock()
	parts := a.aggregation
	a.aggregation = nil
	// This covers whatever a tool result deferred, so the deferred run is
	// dropped rather than repeated behind it.
	a.pushOnBotStopped = false
	a.mu.Unlock()

	said := text.Concatenate(parts)
	if said == "" {
		return
	}
	a.context.AddAssistantMessage(said)
	a.maybeSummarize()
}

// commitInterrupted closes out a turn cut off by an interruption: whatever the
// bot had said by then is committed, so the context reflects what was actually
// heard, and anything left over is dropped rather than carried into the next
// turn.
//
// It does no tool balancing. Every call in flight already has a message
// answering it, written when the call started, so the context is valid as it
// stands. A call the interruption cancels is resolved by its own cancel frame,
// which marks that message canceled where it sits. Appending anything here
// instead would put it after whatever a new user turn has meanwhile added,
// separating a result from the call it answers for the rest of the session.
func (a *AssistantAggregator) commitInterrupted() {
	a.commit()
	a.mu.Lock()
	a.aggregation = nil
	a.pushOnBotStopped = false
	a.mu.Unlock()
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
// The aggregation deliberately survives this. The words that opened the turn are
// the turn's first words: a start strategy that holds out for a few of them only
// decides once they have been transcribed, and by then they are already
// aggregated. Clearing here would throw away exactly the speech that caused the
// turn, leaving it to close having produced nothing. Speech that must not count
// is dropped explicitly instead, by a strategy asking for it through
// onResetAggregation.
func (u *UserAggregator) onTurnStarted(ctx context.Context, params turns.UserTurnStartedParams) {
	if params.EnableUserSpeakingFrames {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
	}
	u.idle.Process(frames.NewUserStartedSpeakingFrame())
	if params.EnableInterruptions {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewInterruptionFrame() })
	}
}

// onInferenceTriggered commits what the user has said so far and runs the LLM on
// it, without the turn being over.
//
// For most strategies this arrives together with the turn ending and is the
// moment the answer starts being written. It matters on its own when
// finalization is deferred to a separate judge: the detector says there is
// enough to answer, and the answer begins while the judge is still deciding,
// rather than after it. Anything the user adds before the turn actually ends is
// committed by onTurnStopped, which runs the LLM again on it.
func (u *UserAggregator) onInferenceTriggered(ctx context.Context) {
	_ = u.maybeRun(ctx)
}

// onResetAggregation drops the speech aggregated so far, at a start strategy's
// request. It is how words that must not count toward a turn are discarded:
// anything said before a wake phrase, or an utterance too short to open one.
// Since a turn beginning no longer clears the aggregation, this is the only way
// such speech is kept out of the conversation.
func (u *UserAggregator) onResetAggregation(context.Context) {
	u.mu.Lock()
	u.aggregation = ""
	u.mu.Unlock()
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
	_ = u.maybeRun(ctx)
}

var _ turns.Emitter = (*UserAggregator)(nil)
