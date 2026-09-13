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
// It wraps an external strategy rather than replacing it, the way Deferred does:
// the service owns turn detection, so what ends the turn is unchanged, and what
// this adds is the speculation around it.
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
	// pendingInference records that the strategy underneath asked for inference
	// while a speculation was in flight. The ask is held until the turn's
	// verdict is known: a speculation the committed transcript confirms has
	// answered already, and asking again would answer the same turn twice.
	pendingInference bool
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
	inner.inferenceTriggered = s.onInnerInference
	inner.stopped = s.onInnerStopped
	s.inner.attach(s.inner, inner)
}

// Process starts a speculation on an eager end of turn, or withdraws one, then
// lets the strategy underneath see the frame as usual.
func (s *EagerStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.EagerTranscriptionFrame:
		s.speculate(fr)
	case *frames.EagerEndOfTurnCancelFrame:
		// The service withdrew its prediction, and its frame reaches every
		// consumer on its own. Only our own state is left to clear.
		s.takeSpeculation()
	}
	return s.inner.Process(f)
}

// TurnStarted readies per-turn state, withdrawing a speculation left over from
// the turn before.
func (s *EagerStop) TurnStarted() {
	s.withdrawIfPending("a new turn began before the speculation resolved")
	s.inner.TurnStarted()
}

// TurnStopped clears per-turn state, withdrawing a speculation the turn ended
// without resolving: the stop watchdog, an interruption, the session ending.
// Nothing else would withdraw it, so the reply would stay held.
func (s *EagerStop) TurnStopped() {
	s.withdrawIfPending("the turn ended unresolved")
	s.inner.TurnStopped()
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

// WaitForTranscript reports whether the turn is held open for a transcript.
func (s *EagerStop) WaitForTranscript() bool { return s.inner.WaitForTranscript() }

// SetWaitForTranscript changes it for the turns that follow.
func (s *EagerStop) SetWaitForTranscript(wait bool) { s.inner.SetWaitForTranscript(wait) }

// onInnerInference holds the ask for inference while a speculation is in flight,
// and passes it on otherwise.
func (s *EagerStop) onInnerInference(_ StopStrategy, speculation *UserTurnSpeculation) {
	s.mu.Lock()
	inFlight := s.speculation != nil
	if inFlight {
		s.pendingInference = true
	}
	s.mu.Unlock()
	if inFlight {
		return
	}
	if s.env.inferenceTriggered != nil {
		s.env.inferenceTriggered(s.self, speculation)
	}
}

// onInnerStopped ends the turn, keeping the speculative reply only if it still
// applies.
func (s *EagerStop) onInnerStopped(_ StopStrategy, params UserTurnStoppedParams) {
	speculation := s.takeSpeculation()
	pending := s.takePendingInference()

	if speculation == nil {
		s.forwardStop(params, pending)
		return
	}

	if s.policy.Matches(speculation.Text, s.inner.text) {
		slog.Debug("turns: the eager end of turn held, keeping the speculative reply")
		// The inference already ran, on the eager transcript. Only finalize: the
		// turn frame it emits is what releases the reply, and the flag stops a
		// second inference answering the same turn.
		params.ConfirmsSpeculation = true
		if s.env.stopped != nil {
			s.env.stopped(s.self, params)
		}
		return
	}

	slog.Debug("turns: the eager end of turn missed, withdrawing the speculative reply",
		"eager", speculation.Text, "committed", s.inner.text)
	s.cancelSpeculation()
	// Inference has to run again, on the committed transcript, so the ask that
	// was held goes out after the withdrawal rather than being dropped with it.
	s.forwardStop(params, pending)
}

// forwardStop passes the turn end on, preceded by the ask for inference that was
// held while a speculation was in flight.
func (s *EagerStop) forwardStop(params UserTurnStoppedParams, pending bool) {
	if pending && s.env.inferenceTriggered != nil {
		s.env.inferenceTriggered(s.self, nil)
	}
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

// withdrawIfPending withdraws a speculation still in flight, saying why.
func (s *EagerStop) withdrawIfPending(reason string) {
	if s.takeSpeculation() == nil {
		return
	}
	slog.Debug("turns: withdrawing the speculative reply", "reason", reason)
	s.cancelSpeculation()
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

// takePendingInference takes the held ask for inference, if there was one.
func (s *EagerStop) takePendingInference() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	pending := s.pendingInference
	s.pendingInference = false
	return pending
}

// Compile-time interface checks.
var (
	_ StopStrategy     = (*EagerStop)(nil)
	_ TranscriptWaiter = (*EagerStop)(nil)
)
