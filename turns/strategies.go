package turns

// UserTurnStrategies holds the start and stop strategy chains a controller runs.
// Per frame, start strategies run in order until one returns Stop; then stop
// strategies run the same way (they usually return Continue and signal via their
// triggers).
type UserTurnStrategies struct {
	Start []StartStrategy
	Stop  []StopStrategy
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

// ExternalStrategies returns strategies for when an external processor (e.g. a
// speech-to-speech service) emits the UserStarted/StoppedSpeakingFrames itself:
// the turn processor relays them without re-emitting speaking frames or
// interruptions.
func ExternalStrategies() UserTurnStrategies {
	return UserTurnStrategies{
		Start: []StartStrategy{NewExternalStart()},
		Stop:  []StopStrategy{NewExternalStop(ExternalStopConfig{})},
	}
}
