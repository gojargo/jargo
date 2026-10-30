package turns

import (
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// DefaultSpeculationTimeout is how long a prediction may go unresolved before it
// is withdrawn.
const DefaultSpeculationTimeout = 5 * time.Second

// EagerStopConfig configures an EagerStop strategy.
type EagerStopConfig struct {
	// ExternalStopConfig configures the external strategy underneath, which is
	// what actually ends the turn: the service owns turn detection here.
	ExternalStopConfig
	// MatchPolicy decides whether the committed transcript is close enough to
	// the eager one to keep the speculative reply. Nil uses NormalizedMatch,
	// which ignores the capitalization and punctuation services commonly add
	// when they commit a transcript. Use ExactMatch to require the two to be
	// identical.
	MatchPolicy EagerMatchPolicy
	// SpeculationTimeout is how long a prediction may go unresolved before it is
	// withdrawn; zero uses five seconds. A service that stops sending turn
	// signals mid-speculation would otherwise leave the reply held and the bot
	// silent for the rest of the session.
	SpeculationTimeout time.Duration
}

// EagerStop answers an eager end of turn while the turn is still open.
//
// Some transcribers predict the end of a turn before committing to it, and
// withdraw the prediction if the user turns out to be mid-sentence. This
// strategy starts generating a reply on that prediction, so the gap between it
// and the committed end of turn is spent generating rather than waiting.
//
// The prediction can be wrong in three ways, and each withdraws the reply: the
// user resumes speaking and the service withdraws the eager end of turn, the
// committed transcript differs from the eager one per the match policy, or the
// turn never commits within the speculation timeout. Every one of those leaves
// here as a withdrawal, so a reply is never dropped somewhere the rest of the
// pipeline cannot see.
//
// Nothing the speculation produces reaches the user or the conversation. The
// inference runs against a provisional conversation, and its reply is held by
// the model service until the turn is confirmed. The turn ends normally: the
// user message written to the conversation is always the committed transcript,
// never the eager one.
//
// It wraps an external strategy rather than replacing it: the service owns turn
// detection, so what ends the turn is unchanged, and what this adds is the
// speculation around it. The decision the strategy underneath reports comes
// back here first, which is where the speculation is kept or withdrawn.
//
// Install it with EagerStrategies rather than directly.
type EagerStop struct {
	inner   *ExternalStop
	policy  EagerMatchPolicy
	timeout time.Duration

	self StopStrategy
	env  strategyEnv

	// mu guards the speculation and its timer against the timer goroutine.
	mu          sync.Mutex
	speculation *UserTurnSpeculation
	stopTimer   func()
	// confirmed is the speculation this turn's end confirmed. Other stop
	// strategies may still be deciding, and a stop reported again confirms it
	// again. It is only touched under the controller's lock.
	confirmed *UserTurnSpeculation
}

// NewEagerStop builds an eager stop strategy.
func NewEagerStop(cfg EagerStopConfig) *EagerStop {
	s := &EagerStop{
		inner:   NewExternalStop(cfg.ExternalStopConfig),
		policy:  cfg.MatchPolicy,
		timeout: cfg.SpeculationTimeout,
	}
	if s.policy == nil {
		s.policy = NormalizedMatch{}
	}
	if s.timeout <= 0 {
		s.timeout = DefaultSpeculationTimeout
	}
	return s
}

// MatchPolicy is the policy deciding whether a speculative reply still applies.
func (s *EagerStop) MatchPolicy() EagerMatchPolicy { return s.policy }

// attach hands the strategy underneath an environment whose decisions come back
// here first, so a turn ending can be answered with what the speculation knows.
func (s *EagerStop) attach(self StopStrategy, env strategyEnv) {
	s.self, s.env = self, env
	inner := env
	inner.inferenceTriggered = func(_ StopStrategy, speculation *UserTurnSpeculation) {
		if env.inferenceTriggered != nil {
			env.inferenceTriggered(self, speculation)
		}
	}
	inner.stopped = s.onInnerStopped
	s.inner.attach(s.inner, inner)
}

// Process starts a speculation on an eager end of turn, withdraws one, or
// forgets one, then lets the strategy underneath see the frame as usual.
func (s *EagerStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.EagerTranscriptionFrame:
		s.speculate(fr)
	case *frames.UserTurnResponseCancelFrame:
		// The service withdrew its prediction, and its frame reaches every
		// consumer on its own. Only our own state is left to clear.
		s.takeSpeculation()
		s.confirmed = nil
	case *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame,
		*frames.ProposedUserStartedSpeakingFrame:
		// Speech voids the turn's answer, and with it the confirmation of the
		// speculation that gave it.
		s.confirmed = nil
	}
	return s.inner.Process(f)
}

// TurnStarted readies per-turn state, withdrawing a speculation left over from
// the turn before.
func (s *EagerStop) TurnStarted() {
	s.reset(s.inner.TurnStarted)
}

// TurnStopped clears per-turn state, withdrawing a speculation the turn ended
// without resolving: the stop watchdog, an interruption, the session ending.
// Nothing else would withdraw it, so the reply would stay held.
func (s *EagerStop) TurnStopped() {
	s.reset(s.inner.TurnStopped)
}

// reset clears per-turn state at a turn boundary, around the strategy
// underneath doing the same.
func (s *EagerStop) reset(inner func()) {
	speculation := s.takeSpeculation()
	s.confirmed = nil
	inner()
	if speculation != nil {
		slog.Debug("turns: the turn ended unresolved, withdrawing the speculative reply")
		s.cancelSpeculation()
	}
}

// Setup hands the pipeline's configuration to the strategy underneath.
func (s *EagerStop) Setup(st processor.Setup) error { return s.inner.Setup(st) }

// Cleanup stops the speculation timer and tears the strategy down.
func (s *EagerStop) Cleanup() {
	s.takeSpeculation()
	s.inner.Cleanup()
}

// ResolvesProposedTurnStopFrames reports what the strategy underneath does with
// proposals: speculating changes what happens during a turn, not who ends it.
func (s *EagerStop) ResolvesProposedTurnStopFrames() bool {
	return s.inner.ResolvesProposedTurnStopFrames()
}

// WaitsForInference reports that the strategy listens to the service rather
// than waiting for the turn's inference.
func (s *EagerStop) WaitsForInference() bool { return false }

// DecisionScheduled reports that the strategy schedules no decision of its own.
func (s *EagerStop) DecisionScheduled() bool { return false }

// WaitForTranscript reports whether the turn is held open for a transcript.
func (s *EagerStop) WaitForTranscript() bool { return s.inner.WaitForTranscript() }

// SetWaitForTranscript changes it for the turns that follow.
func (s *EagerStop) SetWaitForTranscript(wait bool) { s.inner.SetWaitForTranscript(wait) }

// onInnerStopped reports that the user is done, keeping the speculative reply
// only if it still applies.
func (s *EagerStop) onInnerStopped(_ StopStrategy, params UserTurnStoppedParams) {
	speculation := s.takeSpeculation()
	if speculation == nil {
		speculation = s.confirmed
	}
	if speculation == nil {
		s.reportStopped(params)
		return
	}

	if s.policy.Matches(speculation.Text, s.inner.text) {
		slog.Debug("turns: the eager end of turn held, keeping the speculative reply")
		s.confirmed = speculation
		// The inference already ran, on the eager transcript. The turn frame the
		// turn end emits is what releases the reply, and the flag stops a second
		// inference answering the same turn.
		params.ConfirmsSpeculation = true
		s.reportStopped(params)
		return
	}

	slog.Debug("turns: the eager end of turn missed, withdrawing the speculative reply",
		"eager", speculation.Text, "committed", s.inner.text)
	s.confirmed = nil
	s.cancelSpeculation()
	// The committed transcript needs an inference of its own, which the
	// controller runs for this decision.
	params.ConfirmsSpeculation = false
	s.reportStopped(params)
}

// reportStopped passes the decision on as this strategy's.
func (s *EagerStop) reportStopped(params UserTurnStoppedParams) {
	if s.env.stopped != nil {
		s.env.stopped(s.self, params)
	}
}

// speculate answers an eager end of turn, leaving the turn open.
func (s *EagerStop) speculate(fr *frames.EagerTranscriptionFrame) {
	// Segments committed earlier in this turn are part of what the model will
	// see, so they are part of what the committed transcript is compared to.
	speculation := &UserTurnSpeculation{Text: s.inner.text + fr.Text}

	timer := time.AfterFunc(s.timeout, func() { s.speculationTimedOut(speculation) })
	s.mu.Lock()
	s.speculation, s.stopTimer = speculation, func() { timer.Stop() }
	s.mu.Unlock()
	s.confirmed = nil

	slog.Debug("turns: speculating on an eager end of turn", "text", speculation.Text)
	if s.env.inferenceTriggered != nil {
		s.env.inferenceTriggered(s.self, speculation)
	}
}

// speculationTimedOut withdraws a prediction the turn never resolved.
func (s *EagerStop) speculationTimedOut(speculation *UserTurnSpeculation) {
	s.env.locked(func() {
		s.mu.Lock()
		current := s.speculation
		if current != speculation {
			s.mu.Unlock()
			return
		}
		s.speculation, s.stopTimer = nil, nil
		s.mu.Unlock()

		slog.Debug("turns: the eager end of turn went unresolved, withdrawing the speculative reply",
			"timeout", s.timeout)
		s.cancelSpeculation()
	})
}

// cancelSpeculation tells the pipeline the speculative reply is void.
func (s *EagerStop) cancelSpeculation() {
	if s.env.speculationCanceled != nil {
		s.env.speculationCanceled(s.self)
	}
}

// takeSpeculation takes the speculation in flight, stopping the clock on it.
//
// It is the single exit, so a resolved speculation can never be withdrawn again
// by a timer still running for it.
func (s *EagerStop) takeSpeculation() *UserTurnSpeculation {
	s.mu.Lock()
	speculation, stop := s.speculation, s.stopTimer
	s.speculation, s.stopTimer = nil, nil
	s.mu.Unlock()
	if stop != nil {
		stop()
	}
	return speculation
}

// Compile-time interface checks.
var (
	_ StopStrategy     = (*EagerStop)(nil)
	_ TranscriptWaiter = (*EagerStop)(nil)
)
