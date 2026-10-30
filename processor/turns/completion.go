package turns

import (
	"log/slog"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// LLMTurnCompletionStop decides the user is done only when the LLM marks the
// turn complete.
//
// It turns on turn-completion markers in the LLM, which then starts every
// response with the complete marker, the short one (the user will continue
// shortly) or the long one (the user needs more time), and reports each as a
// UserTurnCompletionFrame. The settings update is marked to reach inactive
// services, so every LLM behind a switcher is configured rather than only the
// one in use at startup.
//
// The strategy waits for the turn's inference rather than listening to the
// user, so it needs other stop strategies to decide the user stopped: once they
// have decided, the inference runs, as FilterIncompleteUserTurnStrategies sets
// up. It decides the user is done on that inference's complete marker, on a
// tool call, or on a response with no marker. The short and long markers leave
// the turn open. If the user then stays quiet for the configured timeout, the
// strategy decides the user is done and asks the LLM to re-engage them, and the
// controller's stop timeout waits for that decision. Another inference
// answering the user in the meantime, such as one for words transcribed after
// the turn's inference, cancels the re-prompt. When the user speaks again, it
// waits for the next inference instead.
//
// An eager end of turn answers the turn speculatively, and the strategy
// decides on that answer too. An incomplete marker withdraws it, and the
// committed turn brings an inference of its own.
type LLMTurnCompletionStop struct {
	StopStrategyBase
	config llm.UserTurnCompletionConfig
	// awaited is the inference whose marker this strategy decides on.
	awaited *frames.LLMContextFrame
	// cancelReprompt stops the scheduled re-prompt, and is nil when none is.
	cancelReprompt func()
}

// NewLLMTurnCompletionStop builds the stop strategy that decides on the LLM's
// turn-completion marker. Pair it with stop strategies that listen to the user,
// as FilterIncompleteUserTurnStrategies does.
func NewLLMTurnCompletionStop(cfg llm.UserTurnCompletionConfig) *LLMTurnCompletionStop {
	s := &LLMTurnCompletionStop{config: cfg}
	s.EnableUserSpeakingFrames = true
	return s
}

// Config is the turn-completion configuration this strategy applies.
func (s *LLMTurnCompletionStop) Config() llm.UserTurnCompletionConfig { return s.config }

// WaitsForInference reports that the strategy decides on the LLM's marker,
// which only the turn's inference produces.
func (s *LLMTurnCompletionStop) WaitsForInference() bool { return true }

// DecisionScheduled reports whether a re-prompt is scheduled, which decides
// when it fires.
func (s *LLMTurnCompletionStop) DecisionScheduled() bool { return s.cancelReprompt != nil }

// TurnStarted forgets the previous turn's inference and cancels any pending
// re-prompt.
func (s *LLMTurnCompletionStop) TurnStarted() {
	s.awaited = nil
	s.stopReprompt()
}

// TurnStopped forgets this turn's inference, keeping a pending re-prompt.
func (s *LLMTurnCompletionStop) TurnStopped() { s.awaited = nil }

// Cleanup cancels any pending re-prompt.
func (s *LLMTurnCompletionStop) Cleanup() { s.stopReprompt() }

// Process configures the LLM on start, and follows the turn's inference and the
// user.
func (s *LLMTurnCompletionStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.StartFrame:
		s.configureLLM()
	case *frames.LLMContextFrame:
		// An inference answering the user makes a pending re-prompt redundant.
		s.awaited = fr
		s.stopReprompt()
	case *frames.UserTurnCompletionFrame:
		s.handleReport(fr)
	case *frames.UserTurnResponseCancelFrame:
		// The answer it was waiting for is gone.
		s.awaited = nil
	case *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame,
		*frames.ProposedUserStartedSpeakingFrame:
		// The user speaking again clears the decisions, and the next inference
		// brings its own marker. The user does not need re-engaging either.
		s.awaited = nil
		s.stopReprompt()
	}
	return Continue
}

// configureLLM turns the markers on over a settings update, which is how the
// service learns of them once the pipeline is running.
func (s *LLMTurnCompletionStop) configureLLM() {
	var delta settings.LLM
	delta.FilterIncompleteUserTurns = settings.Set(true)
	delta.UserTurnCompletionConfig = settings.Set[any](s.config)
	update := frames.NewLLMUpdateSettingsFrame(&delta)
	update.ReachInactiveServices = true
	s.Push(update, processor.Downstream)
}

// handleReport decides on the marker of the inference the strategy waits for.
func (s *LLMTurnCompletionStop) handleReport(report *frames.UserTurnCompletionFrame) {
	if s.awaited == nil || report.ContextFrameID != s.awaited.ID() {
		// Another inference is answering the user, so a pending re-prompt would
		// answer them twice.
		s.stopReprompt()
		return
	}

	speculation := s.awaited.Speculation
	s.awaited = nil
	incomplete := report.Kind == "short" || report.Kind == "long"
	switch {
	case incomplete && speculation:
		s.CancelSpeculation()
	case incomplete:
		s.startReprompt(report.Kind)
	default:
		s.stopReprompt()
		s.ReportStopped()
	}
}

// startReprompt schedules the re-prompt for an incomplete turn of the given
// kind.
func (s *LLMTurnCompletionStop) startReprompt(kind string) {
	s.stopReprompt()
	timeout, prompt := s.config.ShortTimeout(), s.config.ShortPrompt()
	if kind == "long" {
		timeout, prompt = s.config.LongTimeout(), s.config.LongPrompt()
	}
	s.cancelReprompt = s.after(timeout, func() {
		s.cancelReprompt = nil
		s.reprompt(timeout, prompt)
	})
}

// reprompt ends the turn the user went quiet in, and asks the LLM to re-engage
// them.
func (s *LLMTurnCompletionStop) reprompt(timeout time.Duration, prompt string) {
	slog.Debug("turns: the user was quiet after an incomplete turn, re-prompting", "timeout", timeout)
	s.ReportStopped()
	// The turn end runs an inference of its own for words transcribed after the
	// turn's inference, and that inference answers the user instead.
	if s.awaited != nil {
		return
	}
	appendFrame := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleDeveloper, Text: prompt}})
	appendFrame.RunLLM = true
	s.Push(appendFrame, processor.Downstream)
}

// stopReprompt cancels the scheduled re-prompt, if there is one.
func (s *LLMTurnCompletionStop) stopReprompt() {
	if s.cancelReprompt != nil {
		s.cancelReprompt()
		s.cancelReprompt = nil
	}
}

// FilterIncompleteUserTurnStrategies builds a stop chain gated on the LLM's own
// verdict of whether the user had finished speaking.
//
// The LLM is asked to begin every response with one of three markers: complete,
// incomplete short, or incomplete long. Only the complete one ends the user
// turn; the incomplete ones keep the turn open so the user can continue
// speaking and the LLM can re-evaluate later.
//
// It keeps the stop strategies given (the defaults when there are none) and
// adds an LLMTurnCompletionStop, which waits for the turn's inference. Once the
// other stop strategies decide the user stopped, the inference runs, and the
// turn ends once the LLM marks it complete. This works with either StopMode.
// The LLM service is configured by that strategy itself when the pipeline
// starts, so nothing else has to be set up.
//
// With UserTurnStrategiesParams{StopMergeMode: MergeModeExtend}, the LLM gate
// and any stop strategies given join the ones a service recommends instead,
// such as a transcription service's own turn detection.
func FilterIncompleteUserTurnStrategies(stop []StopStrategy, cfg llm.UserTurnCompletionConfig) UserTurnStrategies {
	gate := NewLLMTurnCompletionStop(cfg)
	given := append(append([]StopStrategy(nil), stop...), gate)
	if len(stop) == 0 {
		stop = DefaultStopStrategies()
	}
	return UserTurnStrategies{
		Stop: append(append([]StopStrategy(nil), stop...), gate),
		// Extending a service's stop strategies adds the LLM gate and the stop
		// strategies given, never the default ones.
		stopExtension:    given,
		hasStopExtension: true,
	}
}

var _ StopStrategy = (*LLMTurnCompletionStop)(nil)
