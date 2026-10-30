package turns

import (
	"fmt"
	"time"

	"github.com/gojargo/jargo/audio/turn"
)

// MergeMode is how an application's list of user turn strategies combines with
// the one a service recommends.
type MergeMode int

const (
	// MergeModeReplace uses the application's list on its own. It is the
	// default.
	MergeModeReplace MergeMode = iota
	// MergeModeExtend adds the strategies the application gave to the
	// service's list, or to the defaults when no service recommends any.
	MergeModeExtend
)

// StrategyMode is how a list of user turn strategies reaches a decision.
type StrategyMode int

const (
	// StrategyModeDefault is the list's own default: StrategyModeFirst for the
	// start strategies and StrategyModeAll for the stop strategies.
	StrategyModeDefault StrategyMode = iota
	// StrategyModeAll has every strategy agree, and each one sees every frame.
	StrategyModeAll
	// StrategyModeFirst has the first strategy to decide win.
	StrategyModeFirst
)

// UserTurnStrategiesParams is how the user turn strategies combine with a
// service's and decide.
//
// A service with its own turn detection, such as a transcription service, can
// recommend user turn strategies. Only the application sets these parameters: a
// service's recommendation contributes its lists. The zero value is the
// defaults.
type UserTurnStrategiesParams struct {
	// StartMergeMode is how the application's start strategies combine with the
	// ones a service recommends. It defaults to MergeModeReplace.
	StartMergeMode MergeMode
	// StopMergeMode is how the application's stop strategies combine with the
	// ones a service recommends. It defaults to MergeModeReplace.
	StopMergeMode MergeMode
	// StartMode is whether the turn starts on the first start strategy that
	// decides the user started, or once every start strategy has. It defaults
	// to StrategyModeFirst, so speech or a transcript starts a turn on its own.
	// In a StrategyModeAll list every strategy is a condition the turn needs,
	// so a fallback such as TranscriptionStart, which starts a turn when the
	// VAD misses speech, belongs in a StrategyModeFirst list.
	StartMode StrategyMode
	// StopMode is whether the turn ends once every stop strategy has decided the
	// user is done, or on the first one that does. A stop strategy that waits
	// for the turn's inference, such as LLMTurnCompletionStop, decides on it in
	// either mode. The user speaking again clears the decisions made so far. It
	// defaults to StrategyModeAll.
	StopMode StrategyMode
}

// ReplacesServiceStrategies reports whether the application's strategies
// replace both of a service's lists.
func (p UserTurnStrategiesParams) ReplacesServiceStrategies() bool {
	return p.StartMergeMode == MergeModeReplace && p.StopMergeMode == MergeModeReplace
}

// UserTurnStrategies holds the start and stop strategy chains a controller runs.
//
// How each chain reaches its decision follows the StartMode and StopMode of the
// UserTurnStrategiesParams it runs with: by default the first start strategy to
// decide starts the turn, and the turn ends once every stop strategy agrees.
//
// Empty chains use the defaults: VAD and transcription to start, the Smart Turn
// v3 model to stop. An application's strategies replace the ones a service
// recommends, unless UserTurnStrategiesParams extends a list.
type UserTurnStrategies struct {
	Start []StartStrategy
	Stop  []StopStrategy

	// external records that these chains were built by ExternalStrategies, and
	// the interruption setting they were built with. A service recommending
	// external turns carries that setting here, so a caller overruling the
	// recommendation can be told what it discarded.
	external *bool
	// stopExtension is what extending a service's stop strategies adds, when it
	// is not the Stop list as given: a container that adds a strategy of its
	// own after the defaults adds only that strategy and the ones given, never
	// the defaults. It is used once hasStopExtension is set.
	stopExtension    []StopStrategy
	hasStopExtension bool
}

// MergeWith combines these strategies with the ones a service recommends, list
// by list as params says. recommended is nil when no service recommends any,
// and the defaults then stand in for it.
//
// A list that replaces the service's is returned as it is, so an empty one
// still falls back to the defaults where the strategies run. A list that
// extends it adds the strategies given after the service's.
func (s UserTurnStrategies) MergeWith(
	recommended *UserTurnStrategies, params UserTurnStrategiesParams,
) UserTurnStrategies {
	var recStart []StartStrategy
	var recStop []StopStrategy
	if recommended != nil {
		recStart, recStop = recommended.Start, recommended.Stop
	}
	stopExtension := s.Stop
	if s.hasStopExtension {
		stopExtension = s.stopExtension
	}
	return UserTurnStrategies{
		Start: mergeList(s.Start, s.Start, params.StartMergeMode, recStart, DefaultStartStrategies),
		Stop:  mergeList(s.Stop, stopExtension, params.StopMergeMode, recStop, DefaultStopStrategies),
	}
}

// mergeList merges one list with the service's, the defaults standing in for
// none: an empty list runs the defaults wherever it comes from.
func mergeList[S any](own, extension []S, mode MergeMode, recommended []S, defaults func() []S) []S {
	if mode == MergeModeReplace {
		return own
	}
	base := recommended
	if len(base) == 0 {
		base = defaults()
	}
	return append(append([]S(nil), base...), extension...)
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

// fillDefaults populates empty chains with the defaults: VAD and transcription
// to start, the Smart Turn v3 model to stop.
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

// DefaultStopStrategies returns the default stop chain: end-of-turn decided by
// the Smart Turn v3 model, so a pause it rates as unfinished does not end the
// turn.
//
// The model runs on the ONNX runtime, which is located at run time rather than
// linked in, so a pipeline that cannot find it fails to start with the reason
// (see the onnxrt package for where it looks). For turn-taking without a model,
// build the chain explicitly with NewSpeechTimeoutStop.
func DefaultStopStrategies() []StopStrategy {
	analyzer, err := turn.NewSmartTurnV3()
	if err != nil {
		return []StopStrategy{&TurnAnalyzerStop{analyzerErr: err}}
	}
	return []StopStrategy{NewTurnAnalyzerStop(TurnAnalyzerConfig{Analyzer: analyzer})}
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

// EagerStrategiesConfig configures the strategies an eager service recommends.
type EagerStrategiesConfig struct {
	ExternalStrategiesConfig
	// MatchPolicy decides whether the committed transcript is close enough to
	// the eager one to keep the speculative reply. Nil uses NormalizedMatch.
	MatchPolicy EagerMatchPolicy
	// SpeculationTimeout bounds how long a prediction may go unresolved; zero
	// uses five seconds.
	SpeculationTimeout time.Duration
}

// EagerStrategies returns the strategies for a service that predicts the end of
// a turn before committing to it.
//
// They are the external strategies, the service owning turn detection either
// way, with the stop half wrapped so a predicted end of turn is answered while
// the turn is still open. Use them only with a service configured to report such
// a prediction: without one, nothing ever speculates and these behave as the
// external strategies do.
func EagerStrategies(cfg EagerStrategiesConfig) UserTurnStrategies {
	enabled := boolOr(cfg.EnableInterruptions, true)
	return UserTurnStrategies{
		Start: []StartStrategy{NewExternalStart(ExternalStartConfig{EnableInterruptions: &enabled})},
		Stop: []StopStrategy{NewEagerStop(EagerStopConfig{
			MatchPolicy:        cfg.MatchPolicy,
			SpeculationTimeout: cfg.SpeculationTimeout,
		})},
		external: &enabled,
	}
}

// ApplyRealtimeServiceMode strips the transcript dependence from these chains,
// for a pipeline whose conversation is driven by a speech-to-speech service.
//
// Two things go. The transcription start strategy, because a transcript must not
// open a turn when the service is the one listening to the audio. And the
// transcript wait on every stop strategy that has one, so a turn ends as soon as
// the VAD or the end-of-turn model says the speech is over rather than waiting
// on a transcript that is beside the point.
//
// It reports what it changed, so a caller can say so: strategies the application
// configured itself are worth a warning, since the mutation is not what it
// asked for.
func (s *UserTurnStrategies) ApplyRealtimeServiceMode() (dropped, flipped []string) {
	kept := make([]StartStrategy, 0, len(s.Start))
	for _, st := range s.Start {
		if _, ok := st.(*TranscriptionStart); ok {
			dropped = append(dropped, fmt.Sprintf("%T", st))
			continue
		}
		kept = append(kept, st)
	}
	s.Start = kept

	for _, st := range s.Stop {
		w, ok := st.(TranscriptWaiter)
		if !ok || !w.WaitForTranscript() {
			continue
		}
		w.SetWaitForTranscript(false)
		flipped = append(flipped, fmt.Sprintf("%T", st))
	}
	return dropped, flipped
}
