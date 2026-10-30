package turns

import "time"

// Config configures turn taking, for the user aggregator that drives it or for
// a UserTurnProcessor of its own.
//
// Run inside the aggregator, the strategies see the same frames in the same
// order as the aggregation, so a turn that ends on a finalized transcript ends
// with that transcript already folded into the user's message. Run in a
// processor of their own, the decision can instead be made once and shared by
// several aggregators.
type Config struct {
	// Strategies are the start/stop strategy chains; empty chains use the
	// defaults, which are VAD and transcription to start and a speech timeout to
	// stop. For end-of-turn from a model, build Stop with NewTurnAnalyzerStop.
	Strategies UserTurnStrategies
	// StrategiesParams is how the strategies combine with the ones a service
	// recommends, and how they decide. The zero value is the defaults: the
	// application's lists replace the service's, the first start strategy to
	// decide starts the turn, and the turn ends once every stop strategy
	// agrees.
	StrategiesParams UserTurnStrategiesParams
	// StopTimeout is how long an open turn goes without user activity before it
	// is ended, unless a stop strategy has a decision scheduled; 0 uses 5s.
	StopTimeout time.Duration
	// IdleTimeout enables the idle watchdog; a value <= 0 disables it.
	IdleTimeout time.Duration
	// OnIdle fires when the conversation goes idle. Required to enable idle.
	OnIdle IdleCallback
	// MuteStrategies suppress user input while engaged (e.g. while the bot
	// speaks or a tool call runs). They are OR-reduced; empty means never mute.
	MuteStrategies []MuteStrategy
	// EmptyUserTurn is how the user aggregator responds to a user turn that
	// ends with no transcript. Nil uses the defaults: such a turn is answered
	// when it interrupted the bot, and left unanswered otherwise. It is ignored
	// with a realtime LLM service, which hears the user's audio directly, and by
	// a UserTurnProcessor, which aggregates nothing.
	EmptyUserTurn *EmptyUserTurnConfig
}
