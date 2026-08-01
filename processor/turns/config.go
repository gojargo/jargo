package turns

import "time"

// Config configures the turn taking a user aggregator drives.
//
// The strategies run inside the aggregator rather than in a processor of their
// own, so a turn that ends on a finalized transcript ends with that transcript
// already folded into the user's message.
type Config struct {
	// Strategies are the start/stop strategy chains; the zero value uses the
	// defaults (VAD + transcription start, Smart-Turn stop) — but the Smart-Turn
	// default needs a turn.Analyzer, so most callers build Strategies explicitly.
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
