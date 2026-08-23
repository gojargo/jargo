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
	// StopTimeout is the watchdog that force-stops a stuck turn; 0 uses 5s.
	StopTimeout time.Duration
	// IdleTimeout enables the idle watchdog; a value <= 0 disables it.
	IdleTimeout time.Duration
	// OnIdle fires when the conversation goes idle. Required to enable idle.
	OnIdle IdleCallback
	// MuteStrategies suppress user input while engaged (e.g. while the bot
	// speaks or a tool call runs). They are OR-reduced; empty means never mute.
	MuteStrategies []MuteStrategy
}
