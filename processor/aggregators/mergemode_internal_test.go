package aggregators

import (
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/turns"
)

// An application's stop list that extends a service's joins it, or the
// defaults when no service recommends any.

func stopTypes(s turns.UserTurnStrategies) []string {
	out := make([]string, 0, len(s.Stop))
	for _, st := range s.Stop {
		out = append(out, fmt.Sprintf("%T", st))
	}
	return out
}

func TestAnExtendingListJoinsTheServiceRecommendation(t *testing.T) {
	existing := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 100 * time.Millisecond})
	pair := New(frames.NewLLMContext(""), WithTurns(turns.Config{
		Strategies:       turns.UserTurnStrategies{Stop: []turns.StopStrategy{existing}},
		StrategiesParams: turns.UserTurnStrategiesParams{StopMergeMode: turns.MergeModeExtend},
	}))
	u := pair.User()

	// Not set up, so the defaults standing in until now need no model. The
	// strategies are swapped in all the same.
	u.handleServiceUserTurnStrategies("some-stt", turns.ExternalStrategies(turns.ExternalStrategiesConfig{}))

	strategies := u.turn.Strategies()
	want := []string{"*turns.ExternalStop", "*turns.SpeechTimeoutStop"}
	if got := stopTypes(strategies); !slices.Equal(got, want) {
		t.Fatalf("stop = %v, want %v", got, want)
	}
	if strategies.Stop[len(strategies.Stop)-1] != turns.StopStrategy(existing) {
		t.Fatal("the extending strategy is not the one given")
	}
}

func TestAnExtendingListJoinsTheDefaultsWithoutAService(t *testing.T) {
	existing := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 100 * time.Millisecond})
	pair := New(frames.NewLLMContext(""), WithTurns(turns.Config{
		Strategies:       turns.UserTurnStrategies{Stop: []turns.StopStrategy{existing}},
		StrategiesParams: turns.UserTurnStrategiesParams{StopMergeMode: turns.MergeModeExtend},
	}))

	want := []string{"*turns.TurnAnalyzerStop", "*turns.SpeechTimeoutStop"}
	if got := stopTypes(pair.User().turn.Strategies()); !slices.Equal(got, want) {
		t.Fatalf("stop = %v, want %v", got, want)
	}
}
