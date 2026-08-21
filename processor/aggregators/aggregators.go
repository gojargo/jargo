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
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vad/controller"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/text"
)

// The placeholder contents a tool-result message holds while its call has not
// reported a result of its own. The one written while a call is still running is
// frames.ToolResultInProgress, which lives beside the async-tool protocol
// because summarization has to recognize it too.
const (
	// toolResultCanceled replaces the placeholder when the call is canceled.
	// The spelling is the protocol's, not prose.
	toolResultCanceled = "CANCELLED" //nolint:misspell // the literal written to the conversation
	// toolResultCompleted stands in for a call that finished having produced no
	// result of its own.
	toolResultCompleted = "COMPLETED"
)

// The events the user half of a pair raises around each turn it collects.
const (
	// EventUserTurnStarted fires when the user's turn begins, carrying the
	// turns.StartStrategy that decided it began.
	//
	//	events.On(agg.User().Events(), aggregators.EventUserTurnStarted,
	//	    func(ctx context.Context, s turns.StartStrategy) { … })
	EventUserTurnStarted = "on_user_turn_started"
	// EventUserTurnStopped fires when the user's turn ends, carrying a
	// UserTurnStopped describing what they said and which strategy decided the
	// turn was over.
	EventUserTurnStopped = "on_user_turn_stopped"
	// EventUserTurnInferenceTriggered fires when a stop strategy decides there
	// is enough to answer, which is usually but not always the moment the turn
	// ends. It carries the turns.StopStrategy that decided.
	EventUserTurnInferenceTriggered = "on_user_turn_inference_triggered"
	// EventUserTurnStopTimeout fires when a turn was closed because no stop
	// strategy decided it had ended. It carries no argument.
	EventUserTurnStopTimeout = "on_user_turn_stop_timeout"
	// EventUserTurnIdle fires when the user has said nothing for the configured
	// idle timeout. It carries no argument.
	EventUserTurnIdle = "on_user_turn_idle"
	// EventUserTurnMessageAdded fires when the user's words are written to the
	// conversation, carrying a UserTurnMessageAdded. A turn can write more than
	// once, when an early inference answers part of it, so this fires per write
	// where EventUserTurnStopped fires once per turn.
	EventUserTurnMessageAdded = "on_user_turn_message_added"
	// EventUserMuteStarted fires when the user becomes muted and their input
	// stops reaching the bot. It carries no argument.
	EventUserMuteStarted = "on_user_mute_started"
	// EventUserMuteStopped fires when the user is unmuted. It carries no
	// argument.
	EventUserMuteStopped = "on_user_mute_stopped"
)

// UserTurnStopped describes a user turn that has just ended.
type UserTurnStopped struct {
	// Strategy is the stop strategy that decided the turn was over. It is nil
	// when nothing decided: a turn closed because no strategy did, or one closed
	// by the session ending.
	Strategy turns.StopStrategy
	// Content is everything the user said during the turn, including whatever an
	// earlier inference already answered. It is empty for a turn that said
	// nothing.
	Content string
	// Timestamp is when the turn began, as ISO 8601.
	Timestamp string
	// UserID identifies the speaker, and is empty when the transcription service
	// does not say who spoke.
	UserID string
}

// UserTurnMessageAdded describes a user message written to the conversation.
type UserTurnMessageAdded struct {
	// Content is the message that was written. It is the segment written now,
	// not the whole turn: a turn answered early writes what it had, then writes
	// the rest when it ends.
	Content string
	// Timestamp is when the turn began, as ISO 8601.
	Timestamp string
	// UserID identifies the speaker, and is empty when the transcription service
	// does not say who spoke.
	UserID string
}

// The events the assistant half of a pair raises around each turn it writes.
const (
	// EventAssistantTurnStarted fires when the bot's turn begins, which is the
	// model starting to answer or, for an utterance spoken with no answer around
	// it, that speech starting. It carries no argument.
	//
	//	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStarted,
	//	    func(ctx context.Context, _ any) { … })
	EventAssistantTurnStarted = "on_assistant_turn_started"
	// EventAssistantTurnStopped fires when the bot's turn ends, carrying an
	// AssistantTurnStopped describing what it said.
	//
	//	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
	//	    func(ctx context.Context, t aggregators.AssistantTurnStopped) { … })
	EventAssistantTurnStopped = "on_assistant_turn_stopped"
	// EventAssistantThought fires when a reasoning model finishes a thought,
	// carrying an AssistantThought with what it reasoned.
	EventAssistantThought = "on_assistant_thought"
)

// AssistantThought is one completed piece of a reasoning model's thinking.
type AssistantThought struct {
	// Content is the thought, whole.
	Content string
	// Timestamp is when the thought began, as ISO 8601.
	Timestamp string
}

// AssistantTurnStopped describes a bot turn that has just ended.
type AssistantTurnStopped struct {
	// Content is everything the bot said during the turn, as one message. It is
	// empty for a turn that said nothing.
	Content string
	// Interrupted reports whether the turn was cut off rather than finishing.
	Interrupted bool
	// Timestamp is when the turn began, as ISO 8601.
	Timestamp string
}

// Pair is a user and assistant aggregator sharing one conversation context.
type Pair struct {
	context   *frames.LLMContext
	user      *UserAggregator
	assistant *AssistantAggregator
}

// Option configures an aggregator Pair.
type Option func(*options)

type options struct {
	turns *turns.Config
	// summarize configures the automatic summarization thresholds, and is nil
	// when only on-demand summarization is wanted.
	summarize *frames.AutoSummarizationConfig
	// mute suppresses user input while the bot is engaged. It is separate from
	// the turn configuration because muting the user and deciding when their
	// turn ended are different things, and a pipeline can want either alone.
	mute []turns.MuteStrategy
	// vad configures voice-activity detection run inside the user aggregator,
	// and is nil when the pipeline detects speech somewhere else.
	vad *VADConfig
}

// VADConfig configures the voice-activity detection a user aggregator runs.
type VADConfig struct {
	// Analyzer detects voice activity in the incoming audio. Required.
	Analyzer vad.Analyzer
	// AudioIdleTimeout is how long to wait, with the user speaking and no audio
	// arriving at all, before taking the speech to have stopped. It covers the
	// audio going away mid-utterance, a muted microphone being the usual case.
	//
	// Leave it nil for one second. A zero duration turns the watch off.
	AudioIdleTimeout *time.Duration
}

// WithSummarization compresses the conversation automatically once it passes
// either of the configured thresholds. Older turns are replaced by a summary
// written into the conversation ahead of the messages kept back, so the model
// keeps the history without carrying every message of it.
//
// Summarization runs beside the conversation rather than in the path of it, so
// it never adds latency to a turn. Without this option the conversation is still
// compressed on demand, whenever an frames.LLMSummarizeContextFrame is pushed
// into the pipeline.
//
// The summary is generated by the pipeline's own LLM unless the configuration
// names one of its own, which is how summarization is routed to a cheaper model
// than the one carrying the conversation.
func WithSummarization(cfg frames.AutoSummarizationConfig) Option {
	return func(o *options) { o.summarize = &cfg }
}

// WithMuteStrategies suppresses the user's input while the bot is engaged: while
// it is speaking, or while a tool call it is waiting on runs. Input that arrives
// muted is dropped rather than queued, so the bot is not answered by something
// the user said while it was not listening.
//
// The strategies are consulted for every frame, so each keeps up with the
// conversation, and any one of them asking for silence is enough. A change of
// state is announced to the pipeline as a UserMuteStarted or UserMuteStopped
// frame, and raised as an event.
//
// It is independent of WithTurns: muting the user and deciding when their turn
// ended are different questions.
func WithMuteStrategies(strategies ...turns.MuteStrategy) Option {
	return func(o *options) { o.mute = strategies }
}

// WithVAD detects voice activity inside the user aggregator rather than in a
// processor of its own.
//
// The frames it produces are queued back into the aggregator rather than pushed
// at a neighbor, so the turn strategies running here see the speech their own
// detector heard.
//
// Use it when the aggregator is the only thing that needs the detection. Where a
// transport, an interruption decision or a recorder needs it too, put a
// vadproc.Processor after the input transport instead and leave this unset: the
// detection is then done once, where everything downstream of it can see it.
// Running both means analyzing the same audio twice.
//
// Muted input never reaches the detector, so a muted microphone does not read as
// speech.
func WithVAD(cfg VADConfig) Option {
	return func(o *options) { o.vad = &cfg }
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
	mute := o.mute
	if o.turns != nil && len(o.turns.MuteStrategies) > 0 {
		mute = append(append([]turns.MuteStrategy(nil), mute...), o.turns.MuteStrategies...)
	}
	return &Pair{
		context:   ctx,
		user:      newUser(ctx, o.turns, mute, o.vad),
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
	// vad detects voice activity for the strategies running here, and is nil
	// when the pipeline detects speech somewhere else.
	vad *controller.Controller

	muteStrategies []turns.MuteStrategy
	muteMu         sync.Mutex
	muted          bool

	mu sync.Mutex
	// aggregation is what the user has said so far this turn, in the order they
	// said it. Each piece records whether it carries its own spacing, because a
	// transcription service may deliver either kind and joining them all the
	// same way either doubles the spaces or runs the words together.
	aggregation []text.Part
	// turnStartedAt is when the open turn began, as ISO 8601, and is what every
	// report of the turn is stamped with.
	turnStartedAt string
	// wholeTurn is everything the turn has said across every write it has made.
	// A turn answered early writes what it had and empties the aggregation, so
	// without this the report of the turn ending would carry only the tail.
	wholeTurn string
	// userID is who the transcription service said was speaking, carried on what
	// the turn reports.
	userID string
}

func newUser(
	ctx *frames.LLMContext, cfg *turns.Config, mute []turns.MuteStrategy, vadCfg *VADConfig,
) *UserAggregator {
	u := &UserAggregator{context: ctx, turnTaking: cfg != nil, muteStrategies: mute}
	u.Base = processor.New("UserContextAggregator", u)
	if vadCfg != nil {
		u.vad = newVADController(u, *vadCfg)
	}
	// All asynchronous: a handler may do anything, and the turn must not wait on
	// whatever is listening to it.
	for _, name := range []string{
		EventUserTurnStarted, EventUserTurnStopped, EventUserTurnInferenceTriggered,
		EventUserTurnStopTimeout, EventUserTurnIdle, EventUserTurnMessageAdded,
		EventUserMuteStarted, EventUserMuteStopped,
	} {
		u.Events().Register(name, false)
	}
	if cfg == nil {
		return u
	}
	u.turn = turns.NewUserTurnController(cfg.Strategies, cfg.StopTimeout)
	// The event fires alongside the configured callback, so something watching
	// the aggregator hears about an idle conversation without having to be the
	// one thing the pipeline was built with.
	onIdle := func(ctx context.Context, c *turns.UserIdleController) error {
		u.Events().Call(ctx, EventUserTurnIdle, u)
		if cfg.OnIdle == nil {
			return nil
		}
		return cfg.OnIdle(ctx, c)
	}
	u.idle = turns.NewUserIdleController(turns.IdleConfig{Timeout: cfg.IdleTimeout, Callback: onIdle})
	u.turn.SetHooks(turns.ControllerHooks{
		Started:            u.onTurnStarted,
		Stopped:            u.onTurnStopped,
		InferenceTriggered: u.onInferenceTriggered,
		StopTimeout:        u.onStopTimeout,
		ResetAggregation:   u.onResetAggregation,
		Push: func(ctx context.Context, f frames.Frame, dir processor.Direction) {
			// Queued, not pushed: a frame a strategy emits has to travel through
			// this processor like any other, so the aggregation and the rest of
			// the strategies see it. Pushing it at a neighbor would send it past
			// the very things it was emitted to reach.
			_ = u.QueueFrame(ctx, f, dir)
		},
		Broadcast: u.queuedBroadcast,
	})
	return u
}

// newVADController builds the detector the aggregator runs, wired so everything
// it produces is queued back into the aggregator rather than pushed at a
// neighbor.
//
// Queueing is what makes the detection reach the strategies at all. A frame
// pushed at a neighbor leaves this processor without ever being handled by it,
// so the turn controller running here would never see the speech its own
// detector heard. Queued, it arrives like any other frame and is handled in
// turn. The copy going the other way is pushed, since there is nothing on this
// side of the pipeline waiting for it.
func newVADController(u *UserAggregator, cfg VADConfig) *controller.Controller {
	return controller.New(cfg.Analyzer, controller.Handlers{
		OnSpeechStarted: func(ctx context.Context) {
			startSecs := u.vad.Params().StartSecs
			// Taken once, outside the builder, so the frame sent each way reports
			// the same moment.
			ts := time.Now()
			u.queuedBroadcast(ctx, func() frames.Frame {
				return frames.NewVADUserStartedSpeakingFrame(startSecs, ts)
			})
		},
		OnSpeechStopped: func(ctx context.Context) {
			stopSecs := u.vad.Params().StopSecs
			ts := time.Now()
			u.queuedBroadcast(ctx, func() frames.Frame {
				return frames.NewVADUserStoppedSpeakingFrame(stopSecs, ts)
			})
		},
		OnSpeechActivity: func(ctx context.Context) {
			u.queuedBroadcast(ctx, func() frames.Frame { return frames.NewUserSpeakingFrame() })
		},
		OnPushFrame: func(ctx context.Context, f frames.Frame, dir processor.Direction) {
			_ = u.QueueFrame(ctx, f, dir)
		},
		OnBroadcastFrame: u.queuedBroadcast,
	}, controller.Config{AudioIdleTimeout: cfg.AudioIdleTimeout})
}

// Setup wires the controllers.
func (u *UserAggregator) Setup(ctx context.Context, s processor.Setup) error {
	if err := u.Base.Setup(ctx, s); err != nil {
		return err
	}
	if u.vad != nil {
		if err := u.vad.Setup(s); err != nil {
			return err
		}
	}
	if u.turn != nil {
		if err := u.turn.Setup(ctx, s); err != nil {
			return err
		}
		u.idle.Setup(ctx, u)
	}
	return nil
}

// Cleanup tears the controllers down.
func (u *UserAggregator) Cleanup(ctx context.Context) error {
	if u.vad != nil {
		u.vad.Cleanup()
	}
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
	if u.suppressed(ctx, f) {
		return nil
	}
	// Detection runs before the frame is handled, so the speaking frames it
	// raises are queued behind this one and reach the strategies in order with
	// it.
	if u.vad != nil {
		if err := u.vad.ProcessFrame(ctx, f); err != nil {
			return err
		}
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
		u.reportTurnStopped(ctx, nil, true)
		return nil

	case *frames.CancelFrame:
		// Same, the other way round: a cancel tears the pipeline down at once, so
		// what is held is committed before the frame that stops everything.
		u.reportTurnStopped(ctx, nil, true)
		return u.PushFrame(ctx, f, dir)

	case *frames.InterimTranscriptionFrame, *frames.TranslationFrame:
		// Consumed here, like the final transcript below: partial speech is not
		// aggregated either, and the turn strategies that read it run inside this
		// processor. What is downstream is the model, which is given the
		// conversation rather than the transcripts it was built from.
		//
		// A translation is consumed for the same reason and never aggregated:
		// only the transcription is the user's own words, so a provider that
		// reports both must not have the turn counted twice.
		return nil

	case *frames.TranscriptionFrame:
		return u.handleTranscription(ctx, fr)

	case *frames.LLMRunFrame:
		// Explicit trigger: run the LLM on the current context now (e.g. to make
		// the bot speak first), bypassing the turn-completion gating. The frame
		// is consumed and turned into the LLMContextFrame the LLM service runs.
		return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)

	case *frames.LLMMessagesAppendFrame, *frames.LLMMessagesUpdateFrame,
		*frames.LLMMessagesTransformFrame, *frames.LLMSetToolsFrame,
		*frames.LLMSetToolChoiceFrame:
		return u.handleContextUpdate(ctx, f, dir)

	default:
		return u.PushFrame(ctx, f, dir)
	}
}

// handleContextUpdate applies a frame that mutates the shared LLM context.
//
// Only the toolset change is forwarded. A text LLM picks up a context change on
// its next run, but a speech-to-speech service is generating continuously and
// would never learn that the tools it may call had changed, so that one frame
// travels on. The message changes are consumed: the aggregators share the
// context, so a frame that reached one of them has already been applied, and
// forwarding it would let the other apply it a second time.
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

	case *frames.LLMMessagesTransformFrame:
		// Rewriting the conversation from what it currently is, which is what a
		// wholesale replacement cannot express.
		u.context.TransformMessages(fr.Transform)
		runLLM = fr.RunLLM

	case *frames.LLMSetToolsFrame:
		u.context.SetTools(fr.Tools)
		// The one frame that travels on, for a speech-to-speech service that
		// cannot pick the change up on a next run because it never stops running.
		if err := u.PushFrame(ctx, f, dir); err != nil {
			return err
		}

	case *frames.LLMSetToolChoiceFrame:
		u.context.SetToolChoice(fr.ToolChoice)
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
	// Whitespace is not something the user said. A service that reports a final
	// transcript of nothing but spaces would otherwise open a turn, commit it as
	// a message and have the model answer it.
	if strings.TrimSpace(fr.Text) == "" {
		return nil
	}
	// A turn opened by a strategy is stamped when it opens. Without turn taking
	// nothing opens one, so the first transcript of the turn stamps it, and every
	// report of that turn has a moment to carry.
	u.mu.Lock()
	if u.turnStartedAt == "" {
		u.turnStartedAt = frames.NowTimestamp()
	}
	u.userID = fr.UserID
	u.mu.Unlock()

	u.aggregate(text.Part{Text: fr.Text, IncludesInterPartSpaces: fr.IncludesInterFrameSpaces})

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
		_, err := u.maybeRun(ctx)
		return err
	}
	return nil
}

// aggregate folds a transcript into what the turn has said so far.
//
// With turn taking it runs under the turn controller's lock. Ending a turn
// commits the aggregation twice: once when a stop strategy says there is enough
// to answer, and again when it finalizes, so that anything the user added
// between the two still reaches the model. Both run from the strategy's timer,
// under that lock, and a transcript folded in between them would be committed on
// its own: a second user message holding the tail of a sentence, and a second
// inference answering it. A streaming service is entitled to deliver one
// utterance as several final transcripts, so this is speech arriving as
// expected, not a fault. Taking the lock puts the transcript either side of the
// pair rather than inside it.
func (u *UserAggregator) aggregate(part text.Part) {
	fold := func() {
		u.mu.Lock()
		defer u.mu.Unlock()
		u.aggregation = append(u.aggregation, part)
	}
	if u.turn != nil {
		u.turn.Locked(fold)
		return
	}
	fold()
}

// maybeRun commits the aggregated user message and triggers the LLM on it.
// Nothing is committed when the user has said nothing since the last time.
//
// With turn taking it is called only from the turn controller's hooks, which
// are the sole authority on when the aggregation becomes a message. Without turn
// taking a finalized transcript alone suffices.
func (u *UserAggregator) maybeRun(ctx context.Context) (string, error) {
	u.mu.Lock()
	parts := u.aggregation
	u.aggregation = nil
	startedAt, userID := u.turnStartedAt, u.userID
	u.mu.Unlock()

	said := text.Concatenate(parts)
	if said == "" {
		return "", nil
	}
	u.context.AddUserMessage(said)
	err := u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)
	u.Events().Call(ctx, EventUserTurnMessageAdded, u,
		UserTurnMessageAdded{Content: said, Timestamp: startedAt, UserID: userID})
	return said, err
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
	// summarizer compresses the conversation once it has grown long. It is
	// always present, so an on-demand LLMSummarizeContextFrame is always
	// honored; the automatic thresholds are what WithSummarization enables.
	summarizer *Summarizer

	mu sync.Mutex
	// aggregation is what the turn has said so far, in the order it was said:
	// the model's own text where that is what reaches the conversation, and the
	// playback-aligned words a word-timestamp TTS reports where it is not. The
	// two kinds space themselves differently, so each piece carries the answer
	// and the join respects it.
	aggregation []text.Part
	// turnStartedAt is when the open assistant turn began, and empty when no turn
	// is open. A turn opens on the model's response starting, or, for an
	// utterance the service speaks with no response around it, on the start of
	// that speech. It doubles as the flag because a turn is reported with the
	// moment it began, so the two can never disagree.
	turnStartedAt string
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

	// thought is the reasoning being streamed now, and the moment it began. A
	// thought is kept apart from the turn's own text: it is the model reasoning
	// with itself, so it is never spoken and never joins what the bot said.
	thought        []text.Part
	thoughtStarted string
	// thoughtToContext and thoughtLLM record whether the thought being streamed
	// is written to the conversation, and as whose provider's native message.
	thoughtToContext bool
	thoughtLLM       string

	// Lifetime of the context-updated callbacks, which run off the frame path.
	// Cleanup cancels them and waits for them to return.
	taskCtx    context.Context
	taskCancel context.CancelFunc
	taskWG     sync.WaitGroup
}

func newAssistant(ctx *frames.LLMContext, sc *frames.AutoSummarizationConfig) *AssistantAggregator {
	a := &AssistantAggregator{
		context:    ctx,
		inProgress: make(map[string]*frames.FunctionCallInProgressFrame),
	}
	// The summarizer exists whether or not the thresholds are enabled, so a
	// pushed LLMSummarizeContextFrame is always acted on; sc is what decides
	// whether it also triggers on its own.
	cfg := frames.NewAutoSummarizationConfig()
	if sc != nil {
		cfg = *sc
	}
	a.summarizer = NewSummarizer(ctx, cfg, sc != nil)
	a.summarizer.Add(EventRequestSummarization, a.onRequestSummarization)
	a.Base = processor.New("AssistantContextAggregator", a)
	// Both run on their own goroutine: a handler may do anything, and a turn
	// beginning or ending must not be held up by whatever is listening for it.
	a.Events().Register(EventAssistantTurnStarted, false)
	a.Events().Register(EventAssistantTurnStopped, false)
	a.Events().Register(EventAssistantThought, false)
	return a
}

// Summarizer is the conversation summarizer this aggregator owns, for attaching
// handlers to the events it raises.
func (a *AssistantAggregator) Summarizer() *Summarizer { return a.summarizer }

// onRequestSummarization puts a summary request to the LLM service. It travels
// upstream, which is where the service sits relative to the assistant half of
// the pair.
func (a *AssistantAggregator) onRequestSummarization(ctx context.Context, _ any, args ...any) {
	request, ok := args[0].(*frames.LLMContextSummaryRequestFrame)
	if !ok {
		return
	}
	if err := a.PushFrame(ctx, request, processor.Upstream); err != nil {
		slog.Error("putting the summary request to the LLM failed", "error", err)
	}
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

// Cleanup cancels the context-updated callbacks and the summarization in
// flight, and waits for them to return.
func (a *AssistantAggregator) Cleanup(ctx context.Context) error {
	a.releaseWork(ctx)
	return a.Base.Cleanup(ctx)
}

// ProcessFrame collects LLM text into an assistant message.
func (a *AssistantAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if mf, ok := f.(*frames.LLMMarkerFrame); ok {
		return a.handleMarker(ctx, mf)
	}
	if handled, err := a.handleContextUpdate(ctx, f); handled {
		return err
	}
	return a.route(ctx, f, dir)
}

// route folds one frame into the turn being aggregated and forwards it. It is
// split from ProcessFrame so each stays readable: this is the frame taxonomy,
// ProcessFrame is the sideband handling that runs ahead of it.
func (a *AssistantAggregator) route(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		a.openTurn(ctx)
	case *frames.TTSStartedFrame:
		// Speech that will be recorded opens a turn of its own when none is open.
		// It is how an utterance the service speaks, with no response around it to
		// mark where the turn begins, still gets one, while deferring to a turn the
		// model already started.
		if fr.AppendToContext {
			a.openTurn(ctx)
		}
	case *frames.TextFrame, *frames.LLMTextFrame, *frames.TTSTextFrame, *frames.AggregatedTextFrame:
		a.routeText(f)
	case *frames.LLMThoughtStartFrame:
		a.startThought(fr)
		return nil
	case *frames.LLMThoughtTextFrame:
		a.aggregateThought(fr)
		return nil
	case *frames.LLMThoughtEndFrame:
		a.endThought(ctx, fr)
		return nil
	case *frames.LLMAssistantPushAggregationFrame, *frames.LLMFullResponseEndFrame,
		*frames.EndFrame, *frames.CancelFrame, *frames.InterruptionFrame:
		if err := a.closeTurn(ctx, f); err != nil {
			return err
		}
	case *frames.LLMRunFrame:
		// An explicit request to run the model, pushed down at this half rather
		// than up at the user half: the LLM service's own re-prompt after an
		// incomplete turn is the one that arrives here. The service sits
		// upstream of this processor, so the run it asks for has to travel back
		// the other way. The frame is consumed, since running the model is what
		// it asked for and nothing beyond this processor acts on it.
		return a.pushContextFrame(ctx)
	case *frames.FunctionCallsStartedFrame, *frames.FunctionCallInProgressFrame,
		*frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame:
		// Consumed, not forwarded. This aggregator is where a tool call becomes
		// conversation, and it is the last processor in the pipeline, so there is
		// nothing beyond it to tell. Every other consumer is reached by the LLM
		// service broadcasting each of these frames upstream as well as down: the
		// idle watchdog and the mute strategies run inside the user aggregator, and
		// an RTVI processor sits between the LLM and the output.
		return a.handleFunctionCallFrame(ctx, f)
	case *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame:
		return a.handleSpeakingState(ctx, f, dir)
	}
	if err := a.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	// The summarizer watches the conversation from beside it, and sees each
	// frame once it has gone on, so compressing never delays a turn.
	a.summarizer.ProcessFrame(ctx, f)
	return nil
}

// closeTurn ends the assistant turn, for each of the things that ends one.
//
// A response ending and an utterance ending are the turn finishing. An
// interruption and a cancellation cut it off, so what the bot said is not what
// it meant to say, and the turn is reported as interrupted. The session ending
// closes the turn out where it stands rather than losing it with the processor.
func (a *AssistantAggregator) closeTurn(ctx context.Context, f frames.Frame) error {
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		_, canceled := f.(*frames.CancelFrame)
		err := a.commit(ctx, canceled)
		// The session is over, so what was running beside the conversation stops
		// here rather than waiting for teardown: a callback still running would
		// otherwise act on a conversation nothing is left to carry.
		a.releaseWork(ctx)
		return err
	case *frames.InterruptionFrame:
		// Whatever the bot already said is kept. The tool calls still running are
		// left alone: each already has a placeholder result in the context, so the
		// turn is balanced as it stands, and the LLM service resolves them by
		// canceling the calls it registered to cancel.
		return a.commitInterrupted(ctx)
	default:
		return a.commit(ctx, false)
	}
}

// releaseWork stops what the aggregator has running beside the conversation: the
// context-updated callbacks a tool result started, and the summarizer. It is
// idempotent, so the session ending and teardown can both run it.
func (a *AssistantAggregator) releaseWork(ctx context.Context) {
	if a.taskCancel != nil {
		a.taskCancel()
	}
	a.taskWG.Wait()
	a.summarizer.Cleanup(ctx)
}

// routeText folds one piece of the turn's text into the aggregation, taking each
// kind of text frame for what it contributes to the conversation.
func (a *AssistantAggregator) routeText(f frames.Frame) {
	switch fr := f.(type) {
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
	case *frames.TextFrame:
		// Plain text put into the pipeline by something other than the model. It
		// is part of the turn like anything else the bot says.
		a.handleText(fr, fr.Text)
	}
}

// startThought begins a new thought, discarding anything a previous one left.
func (a *AssistantAggregator) startThought(fr *frames.LLMThoughtStartFrame) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.thought = nil
	a.thoughtToContext = fr.AppendToContext
	a.thoughtLLM = fr.LLM
	a.thoughtStarted = frames.NowTimestamp()
}

// aggregateThought folds one chunk of reasoning into the thought being streamed.
func (a *AssistantAggregator) aggregateThought(fr *frames.LLMThoughtTextFrame) {
	if fr.Text == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.thought = append(a.thought,
		text.Part{Text: fr.Text, IncludesInterPartSpaces: fr.IncludesInterFrameSpaces})
}

// endThought closes the thought out: it is written to the conversation when the
// provider asked for it back, and reported either way.
//
// A thought that is written goes in as that provider's own message, signature
// and all, because only the provider it came from can make sense of it. Every
// other provider skips a message that is not theirs.
func (a *AssistantAggregator) endThought(ctx context.Context, fr *frames.LLMThoughtEndFrame) {
	a.mu.Lock()
	parts, startedAt := a.thought, a.thoughtStarted
	toContext, llm := a.thoughtToContext, a.thoughtLLM
	a.thought, a.thoughtStarted = nil, ""
	a.mu.Unlock()

	thought := text.Concatenate(parts)
	if toContext {
		a.context.AddMessage(frames.NewLLMSpecificMessage(llm, map[string]any{
			"type":      "thought",
			"text":      thought,
			"signature": fr.Signature,
		}))
	}
	a.Events().Call(ctx, EventAssistantThought, a,
		AssistantThought{Content: thought, Timestamp: startedAt})
}

// handleMarker records a sideband marker the LLM service emitted.
//
// A marker that stands alone is the whole assistant turn, which is what an
// incomplete-turn signal is: the spoken reply was suppressed and the marker is
// the only thing that happened. A marker that does not is the prefix of a reply
// still being aggregated, so it joins the aggregation and is written with the
// text as one message.
func (a *AssistantAggregator) handleMarker(ctx context.Context, fr *frames.LLMMarkerFrame) error {
	if fr.AppendToContextImmediately {
		a.context.AddAssistantMessage(fr.Marker)
		return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
	}
	a.mu.Lock()
	a.aggregation = append(a.aggregation, text.Part{Text: fr.Marker, IncludesInterPartSpaces: false})
	a.mu.Unlock()
	return nil
}

// handleContextUpdate applies a conversation change arriving from this side of
// the pipeline, reporting whether it was one. Both halves of the pair handle
// these, and both consume them: they share the conversation, so a frame that
// reached one of them has been applied and must not travel on to the other.
func (a *AssistantAggregator) handleContextUpdate(ctx context.Context, f frames.Frame) (bool, error) {
	switch fr := f.(type) {
	case *frames.LLMMessagesAppendFrame:
		return true, a.handleMessagesAppend(ctx, fr)
	case *frames.LLMMessagesUpdateFrame:
		return true, a.handleMessagesUpdate(ctx, fr)
	case *frames.LLMMessagesTransformFrame:
		return true, a.handleMessagesTransform(ctx, fr)
	case *frames.LLMSetToolsFrame:
		// Consumed here. The user half forwards this one so a continuously
		// running service learns of the change, which is what brings it this far;
		// the two halves share the conversation, so applying it again settles
		// nothing new and it stops here.
		a.context.SetTools(fr.Tools)
		return true, nil
	case *frames.LLMSetToolChoiceFrame:
		a.context.SetToolChoice(fr.ToolChoice)
		return true, nil
	}
	return false, nil
}

// handleMessagesAppend adds messages to the conversation from this side of the
// pipeline, and runs the model on them when asked. It is what the LLM service's
// own re-prompt reaches: the service pushes it downstream, so it arrives here
// rather than at the user aggregator, and the run it asks for has to travel back
// upstream to the service.
//
// The frame is consumed, like its counterpart on the user side, so a
// conversation both halves share is never appended to twice.
func (a *AssistantAggregator) handleMessagesAppend(ctx context.Context, fr *frames.LLMMessagesAppendFrame) error {
	for _, m := range fr.Messages {
		a.context.AddMessage(m)
	}
	if !fr.RunLLM {
		return nil
	}
	return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
}

// handleMessagesUpdate replaces the conversation from this side of the pipeline,
// and runs the model on it when asked.
func (a *AssistantAggregator) handleMessagesUpdate(ctx context.Context, fr *frames.LLMMessagesUpdateFrame) error {
	a.context.SetMessages(fr.Messages)
	if !fr.RunLLM {
		return nil
	}
	return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
}

// handleMessagesTransform rewrites the conversation from this side of the
// pipeline, and runs the model on it when asked.
func (a *AssistantAggregator) handleMessagesTransform(
	ctx context.Context, fr *frames.LLMMessagesTransformFrame,
) error {
	a.context.TransformMessages(fr.Transform)
	if !fr.RunLLM {
		return nil
	}
	return a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Upstream)
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
		return a.handleFunctionCallCancel(ctx, fr)
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
			Content: frames.ToolResultInProgress,
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

	if started == nil {
		// The call was announced but has not begun: there is no tool-use block in
		// the conversation yet, so there is nothing for this result to answer.
		slog.WarnContext(ctx, "tool result for a call that is not in progress",
			"processor", a.Name(), "tool", fr.ToolName, "tool_call_id", fr.ToolCallID)
		return nil
	}

	groupID := started.GroupID
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

// handleFunctionCallCancel settles a canceled call in the conversation, so the
// tool-use block it answers stays answered rather than being left in progress
// for the rest of the session.
//
// An ordinary call's placeholder is marked canceled where it sits. A call the
// model does not wait on is settled with an async-tool message instead, the same
// channel its results would have arrived on.
//
// The frame says whether inference follows. Only a call canceled by its own
// deadline asks for it, and only then is there nothing else to run it: an
// interruption must not answer over the user, and a cancellation the model asked
// for is answered by the result of the tool that asked.
func (a *AssistantAggregator) handleFunctionCallCancel(
	ctx context.Context, fr *frames.FunctionCallCancelFrame,
) error {
	a.mu.Lock()
	started, running := a.inProgress[fr.ToolCallID]
	if !running {
		a.mu.Unlock()
		return nil
	}
	delete(a.inProgress, fr.ToolCallID)
	sync := started == nil || started.CancelOnInterruption
	groupID := ""
	if started != nil {
		groupID = started.GroupID
	}
	speaking := a.userSpeaking
	a.mu.Unlock()

	if sync {
		a.context.UpdateToolResult(fr.ToolCallID, toolResultCanceled)
	} else {
		a.context.AddMessage(frames.NewAsyncToolCanceledMessage(fr.ToolCallID))
	}

	if !fr.RunLLM || speaking {
		return nil
	}
	// Hold off while siblings from the same response are still running:
	// whichever of them settles last runs inference, with this cancellation
	// already in the conversation.
	if groupID != "" && a.groupStillRunning(groupID, fr.ToolCallID) {
		return nil
	}
	return a.maybePushContextAfterFunctionResult(ctx)
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

// openTurn marks an assistant turn open and reports that the bot has begun.
// Opening one that is already open changes nothing: the turn belongs to the
// response, not to each frame of it.
func (a *AssistantAggregator) openTurn(ctx context.Context) {
	a.mu.Lock()
	if a.turnStartedAt != "" {
		a.mu.Unlock()
		return
	}
	a.turnStartedAt = frames.NowTimestamp()
	a.mu.Unlock()
	a.Events().Call(ctx, EventAssistantTurnStarted, a)
}

// commit closes the assistant turn out, writing what it said to the
// conversation as one message and reporting the turn. A turn that was never
// opened writes nothing, so text arriving with no turn around it waits for one
// rather than landing on its own.
//
// interrupted says whether the turn was cut off rather than finishing, which is
// what tells a consumer that what the bot said is not what it meant to say.
func (a *AssistantAggregator) commit(ctx context.Context, interrupted bool) error {
	a.mu.Lock()
	startedAt := a.turnStartedAt
	a.turnStartedAt = ""
	a.mu.Unlock()
	if startedAt == "" {
		return nil
	}

	said, err := a.pushAggregation(ctx)
	if err != nil {
		return err
	}
	// What is reported is what the bot said, so the protocol markers that
	// prefixed the reply come out. The conversation keeps them, since the model
	// reads its own earlier verdicts back.
	if said != "" {
		said = frames.StripUserTurnMarkers(said)
	}

	a.Events().Call(ctx, EventAssistantTurnStopped, a, AssistantTurnStopped{
		Content:     said,
		Interrupted: interrupted,
		Timestamp:   startedAt,
	})

	// A turn that said nothing is reported but not announced to the pipeline:
	// there is no text for anything downstream to record.
	if said == "" {
		return nil
	}
	return a.Broadcast(ctx, func() frames.Frame {
		return frames.NewLLMContextAssistantTurnFrame(said, startedAt)
	})
}

// pushAggregation writes what the turn has said so far to the conversation,
// empties the aggregation, and announces the conversation as it now stands. It
// returns what was written, which is empty when nothing was said.
//
// The timestamp frame that follows marks when the message was written, for
// anything building a transcript alongside the conversation.
func (a *AssistantAggregator) pushAggregation(ctx context.Context) (string, error) {
	a.mu.Lock()
	parts := a.aggregation
	a.aggregation = nil
	// This covers whatever a tool result deferred, so the deferred run is
	// dropped rather than repeated behind it.
	a.pushOnBotStopped = false
	a.mu.Unlock()

	said := text.Concatenate(parts)
	if said == "" {
		return "", nil
	}
	a.context.AddAssistantMessage(said)

	if err := a.PushFrame(ctx, frames.NewLLMContextFrame(a.context), processor.Downstream); err != nil {
		return said, err
	}
	return said, a.PushFrame(ctx,
		frames.NewLLMContextAssistantTimestampFrame(frames.NowTimestamp()), processor.Downstream)
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
func (a *AssistantAggregator) commitInterrupted(ctx context.Context) error {
	err := a.commit(ctx, true)
	a.mu.Lock()
	a.aggregation = nil
	a.pushOnBotStopped = false
	a.mu.Unlock()
	return err
}

// suppressed runs the mute strategies and reports whether this user-input frame
// should be dropped. It emits UserMute frames on a change of state.
//
// Whether this frame is dropped is decided by the mute state as it stood before
// the frame arrived, not by what the frame does to it. The frame that starts the
// bot speaking is what mutes the user, and it is not itself user input; the
// frame that unmutes them is likewise the bot falling silent. Deciding after the
// strategies had run would shift the mute window by one frame at each end.
func (u *UserAggregator) suppressed(ctx context.Context, f frames.Frame) bool {
	switch f.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
		// Lifecycle frames are never muted and must not move the mute state:
		// deciding on the StartFrame would announce the user muted before the
		// StartFrame had reached anything downstream.
		return false
	}
	u.muteMu.Lock()
	defer u.muteMu.Unlock()

	drop := u.muted && isUserInput(f)
	if drop {
		slog.DebugContext(ctx, "frame suppressed, the user is muted",
			"processor", u.Name(), "frame", f.Name())
	}

	muted := false
	for _, m := range u.muteStrategies {
		if m.ShouldMute(f) { // call all, so each updates its state
			muted = true
		}
	}
	if muted != u.muted {
		u.muted = muted
		if muted {
			u.Events().Call(ctx, EventUserMuteStarted, u)
			_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserMuteStartedFrame() })
		} else {
			u.Events().Call(ctx, EventUserMuteStopped, u)
			_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserMuteStoppedFrame() })
		}
	}
	return drop
}

// isUserInput reports whether a frame carries something the user did, which is
// what muting suppresses. Everything else passes whatever the mute state is.
func isUserInput(f frames.Frame) bool {
	switch f.(type) {
	case *frames.InterruptionFrame,
		*frames.VADUserStartedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame,
		*frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.InputAudioRawFrame,
		*frames.InterimTranscriptionFrame, *frames.TranscriptionFrame:
		return true
	}
	return false
}

// queuedBroadcast sends a frame a strategy raised both ways: the downstream copy
// is queued back into this processor rather than pushed, so it flows through the
// aggregation and the strategies on its way out, while the upstream copy goes
// straight to the neighbor.
//
// The two copies are unrelated frames rather than a paired broadcast. The
// downstream one has not been sent anywhere yet: it still has to be handled
// here, and whether it travels on at all is for that handling to decide.
func (u *UserAggregator) queuedBroadcast(ctx context.Context, build func() frames.Frame) {
	_ = u.QueueFrame(ctx, build(), processor.Downstream)
	_ = u.PushFrame(ctx, build(), processor.Upstream)
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
func (u *UserAggregator) onTurnStarted(
	ctx context.Context, strategy turns.StartStrategy, params turns.UserTurnStartedParams,
) {
	u.mu.Lock()
	u.turnStartedAt = frames.NowTimestamp()
	u.wholeTurn = ""
	u.mu.Unlock()

	if params.EnableUserSpeakingFrames {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
	}
	u.idle.Process(frames.NewUserStartedSpeakingFrame())
	if params.EnableInterruptions {
		_ = u.BroadcastInterruption(ctx)
	}
	u.Events().Call(ctx, EventUserTurnStarted, u, strategy)
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
func (u *UserAggregator) onInferenceTriggered(ctx context.Context, strategy turns.StopStrategy) {
	// What is committed now is the segment since the last commit, so it is kept
	// alongside the rest of the turn: the report of the turn ending carries
	// everything the user said, not just the tail nobody answered yet.
	segment, _ := u.maybeRun(ctx)
	u.rememberSegment(segment)
	u.Events().Call(ctx, EventUserTurnInferenceTriggered, u, strategy)
}

// onResetAggregation drops the speech aggregated so far, at a start strategy's
// request. It is how words that must not count toward a turn are discarded:
// anything said before a wake phrase, or an utterance too short to open one.
// Since a turn beginning no longer clears the aggregation, this is the only way
// such speech is kept out of the conversation.
func (u *UserAggregator) onResetAggregation(context.Context, turns.StartStrategy) {
	u.mu.Lock()
	u.aggregation = nil
	u.mu.Unlock()
}

// onStopTimeout reports a turn closed because no stop strategy decided it had
// ended.
func (u *UserAggregator) onStopTimeout(ctx context.Context) {
	u.Events().Call(ctx, EventUserTurnStopTimeout, u)
}

// rememberSegment folds a committed segment into what the whole turn has said.
func (u *UserAggregator) rememberSegment(segment string) {
	if segment == "" {
		return
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.wholeTurn == "" {
		u.wholeTurn = segment
		return
	}
	u.wholeTurn = strings.TrimSpace(u.wholeTurn + " " + segment)
}

// onTurnStopped broadcasts the turn-stop decision, feeds the idle controller a
// synthetic user-stopped frame, and commits the turn.
//
// Committing here rather than on a received UserStoppedSpeakingFrame is the
// point of driving the turn from inside the aggregator: the frame that ended
// the turn has already been folded into the aggregation by the time this runs,
// so the user's last words are part of the message the model is given.
func (u *UserAggregator) onTurnStopped(
	ctx context.Context, strategy turns.StopStrategy, params turns.UserTurnStoppedParams,
) {
	if params.EnableUserSpeakingFrames {
		_ = u.Broadcast(ctx, func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	}
	u.idle.Process(frames.NewUserStoppedSpeakingFrame())
	u.reportTurnStopped(ctx, strategy, false)
}

// reportTurnStopped commits whatever the turn had left to say and reports the
// turn as a whole.
//
// onSessionEnd reports only a turn that has something unreported left in it. The
// session ending is not itself the end of a turn, so a turn already reported
// must not be reported a second time on the way out.
func (u *UserAggregator) reportTurnStopped(
	ctx context.Context, strategy turns.StopStrategy, onSessionEnd bool,
) {
	segment, _ := u.maybeRun(ctx)
	u.rememberSegment(segment)

	u.mu.Lock()
	content, startedAt, userID := u.wholeTurn, u.turnStartedAt, u.userID
	u.wholeTurn = ""
	u.mu.Unlock()

	if onSessionEnd && content == "" {
		return
	}
	u.mu.Lock()
	u.turnStartedAt = ""
	u.mu.Unlock()
	u.Events().Call(ctx, EventUserTurnStopped, u, UserTurnStopped{
		Strategy:  strategy,
		Content:   content,
		Timestamp: startedAt,
		UserID:    userID,
	})
}

var _ turns.Emitter = (*UserAggregator)(nil)
