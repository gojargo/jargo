package turns

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// LLMTurnCompletionStop finalizes a turn on the LLM's own verdict that the user
// had finished speaking.
//
// It adds to ExternalCompletionStop the setup the marker protocol needs: on the
// StartFrame it configures the LLM service to gate its replies on the protocol
// and seeds the configuration the gating runs under. The update is marked to
// reach inactive services, so every LLM behind a switcher is configured rather
// than only the one in use at startup.
//
// Finalization itself is inherited. The LLM service detects the completion
// marker in its own output, reports the turn complete, and the base turns that
// into the stop. On an incomplete marker the service re-prompts internally and
// reports nothing, so the turn stays open.
type LLMTurnCompletionStop struct {
	*ExternalCompletionStop
	config llm.UserTurnCompletionConfig
}

// NewLLMTurnCompletionStop builds the stop strategy that finalizes a turn on the
// LLM's completion verdict. Pair it with deferred detectors via
// FilterIncompleteUserTurnStrategies.
func NewLLMTurnCompletionStop(cfg llm.UserTurnCompletionConfig) *LLMTurnCompletionStop {
	return &LLMTurnCompletionStop{ExternalCompletionStop: NewExternalCompletionStop(), config: cfg}
}

// Config is the turn-completion configuration this strategy applies.
func (s *LLMTurnCompletionStop) Config() llm.UserTurnCompletionConfig { return s.config }

// Process configures the LLM on start and leaves finalization to the base.
func (s *LLMTurnCompletionStop) Process(f frames.Frame) ProcessFrameResult {
	if _, ok := f.(*frames.StartFrame); ok {
		s.configureLLM()
	}
	return s.ExternalCompletionStop.Process(f)
}

// configureLLM turns the gating on over a settings update, which is how the
// service learns of it once the pipeline is running.
func (s *LLMTurnCompletionStop) configureLLM() {
	var delta settings.LLM
	delta.FilterIncompleteUserTurns = settings.Set(true)
	delta.UserTurnCompletionConfig = settings.Set[any](s.config)
	update := frames.NewLLMUpdateSettingsFrame(&delta)
	update.ReachInactiveServices = true
	s.Push(update, processor.Downstream)
}

// FilterIncompleteUserTurnStrategies builds a stop chain gated on the LLM's own
// verdict of whether the user had finished speaking.
//
// The detector chain is preserved but deferred, so it only triggers inference
// and leaves finalization to the LLM gate appended after it. Pass your detector
// stop strategies; empty uses the defaults. The LLM service is configured by the
// gate itself when the pipeline starts, so nothing else has to be set up.
func FilterIncompleteUserTurnStrategies(detectors []StopStrategy, cfg llm.UserTurnCompletionConfig) UserTurnStrategies {
	if len(detectors) == 0 {
		detectors = DefaultStopStrategies()
	}
	stop := make([]StopStrategy, 0, len(detectors)+1)
	for _, d := range detectors {
		stop = append(stop, Deferred(d))
	}
	stop = append(stop, NewLLMTurnCompletionStop(cfg))
	return UserTurnStrategies{Stop: stop}
}
