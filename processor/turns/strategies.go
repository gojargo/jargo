package turns

// UserTurnStrategies holds the start and stop strategy chains a controller runs.
// Per frame, start strategies run in order until one returns Stop; then stop
// strategies run the same way (they usually return Continue and signal via their
// triggers).
type UserTurnStrategies struct {
	Start []StartStrategy
	Stop  []StopStrategy

	// external records that these chains were built by ExternalStrategies, and
	// the interruption setting they were built with. A service recommending
	// external turns carries that setting here, so a caller overruling the
	// recommendation can be told what it discarded.
	external *bool
}

// ExternalInterruptions reports whether these chains were built by
// ExternalStrategies and, when they were, whether they broadcast an
// interruption as a proposal opens a turn.
func (s UserTurnStrategies) ExternalInterruptions() (enabled, isExternal bool) {
	if s.external == nil {
		return false, false
	}
	return *s.external, true
}

// fillDefaults populates empty chains with the model-free defaults: VAD +
// transcription to start, a speech-timeout to stop. For Smart-Turn end-of-turn,
// build Stop explicitly with NewTurnAnalyzerStop.
func (s *UserTurnStrategies) fillDefaults() {
	if len(s.Start) == 0 {
		s.Start = DefaultStartStrategies()
	}
	if len(s.Stop) == 0 {
		s.Stop = DefaultStopStrategies()
	}
}

// DefaultStartStrategies returns the default start chain: VAD onset, with
// transcription as a fallback for soft speech the VAD misses.
func DefaultStartStrategies() []StartStrategy {
	return []StartStrategy{NewVADStart(), NewTranscriptionStart(TranscriptionStartConfig{})}
}

// DefaultStopStrategies returns the default, model-free stop chain: a
// speech-timeout after VAD stop. For Smart-Turn, pass a chain built with
// NewTurnAnalyzerStop instead.
func DefaultStopStrategies() []StopStrategy {
	return []StopStrategy{NewSpeechTimeoutStop(SpeechTimeoutConfig{})}
}

// ExternalStrategiesConfig configures ExternalStrategies.
type ExternalStrategiesConfig struct {
	// EnableInterruptions broadcasts an interruption when a proposal opens a
	// turn; nil defaults to true. A service routes its own should-interrupt
	// setting here. It does not apply on the adopt path, where the emitter has
	// already broadcast one.
	EnableInterruptions *bool
}

// ExternalStrategies returns strategies driven by another component in the
// pipeline: a service with its own turn detection, or a shared turn processor
// fanning turns out to several aggregators.
//
// What the aggregator emits depends on which signal drives the turn. A
// ProposedUserStarted/StoppedSpeakingFrame leaves the decision here, so the
// aggregator pushes the turn frames and broadcasts interruptions. A
// UserStarted/StoppedSpeakingFrame means the emitter already announced the turn,
// so the aggregator emits nothing and EnableInterruptions does not apply.
func ExternalStrategies(cfg ExternalStrategiesConfig) UserTurnStrategies {
	enabled := boolOr(cfg.EnableInterruptions, true)
	return UserTurnStrategies{
		Start:    []StartStrategy{NewExternalStart(ExternalStartConfig{EnableInterruptions: &enabled})},
		Stop:     []StopStrategy{NewExternalStop(ExternalStopConfig{})},
		external: &enabled,
	}
}
