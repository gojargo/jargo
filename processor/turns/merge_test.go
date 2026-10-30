package turns_test

import (
	"fmt"
	"testing"

	"github.com/gojargo/jargo/processor/turns"
)

// An application's user turn strategies merge with a service's, list by list.

// extendStop extends the service's stop strategies and replaces its start ones.
//
//nolint:gochecknoglobals // shared test fixture
var extendStop = turns.UserTurnStrategiesParams{StopMergeMode: turns.MergeModeExtend}

// typesOf names the concrete type of every strategy in a list, in order.
func typesOf[S any](list []S) []string {
	out := make([]string, 0, len(list))
	for _, s := range list {
		out = append(out, fmt.Sprintf("%T", s))
	}
	return out
}

func equalLists[S comparable](a, b []S) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestAnApplicationsListsReplaceTheServicesByDefault(t *testing.T) {
	stop := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})
	strategies := turns.UserTurnStrategies{Stop: []turns.StopStrategy{stop}}
	recommended := turns.ExternalStrategies(turns.ExternalStrategiesConfig{})

	merged := strategies.MergeWith(&recommended, turns.UserTurnStrategiesParams{})

	if !equalLists(merged.Stop, []turns.StopStrategy{stop}) {
		t.Fatalf("stop = %v, want the application's own", typesOf(merged.Stop))
	}
	if !equalLists(merged.Start, strategies.Start) {
		t.Fatalf("start = %v, want the application's own", typesOf(merged.Start))
	}
}

func TestAnExtendingListJoinsTheServices(t *testing.T) {
	stop := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})
	recommended := turns.ExternalStrategies(turns.ExternalStrategiesConfig{})

	merged := turns.UserTurnStrategies{Stop: []turns.StopStrategy{stop}}.MergeWith(&recommended, extendStop)

	want := append(append([]turns.StopStrategy(nil), recommended.Stop...), stop)
	if !equalLists(merged.Stop, want) {
		t.Fatalf("stop = %v, want %v", typesOf(merged.Stop), typesOf(want))
	}
}

func TestExtendingAListLeftOutKeepsTheServices(t *testing.T) {
	recommended := turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	params := turns.UserTurnStrategiesParams{StartMergeMode: turns.MergeModeExtend}

	merged := turns.UserTurnStrategies{
		Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
	}.MergeWith(&recommended, params)

	if !equalLists(merged.Start, recommended.Start) {
		t.Fatalf("start = %v, want the service's", typesOf(merged.Start))
	}
}

func TestWithoutARecommendationAnExtendingListJoinsTheDefaults(t *testing.T) {
	start := turns.NewExternalStart(turns.ExternalStartConfig{})
	strategies := turns.UserTurnStrategies{
		Start: []turns.StartStrategy{start},
		Stop:  []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
	}
	params := turns.UserTurnStrategiesParams{StartMergeMode: turns.MergeModeExtend}

	merged := strategies.MergeWith(nil, params)

	want := []string{"*turns.VADStart", "*turns.TranscriptionStart", "*turns.ExternalStart"}
	if got := typesOf(merged.Start); !equalLists(got, want) {
		t.Fatalf("start = %v, want %v", got, want)
	}
	if merged.Start[len(merged.Start)-1] != turns.StartStrategy(start) {
		t.Fatal("the extending strategy is not the one given")
	}
}

func TestOnlyTwoReplacingListsLeaveARecommendationUnused(t *testing.T) {
	if !(turns.UserTurnStrategiesParams{}).ReplacesServiceStrategies() {
		t.Fatal("the defaults should replace both lists")
	}
	if extendStop.ReplacesServiceStrategies() {
		t.Fatal("extending the stop list should not replace both")
	}
	if (turns.UserTurnStrategiesParams{StartMergeMode: turns.MergeModeExtend}).ReplacesServiceStrategies() {
		t.Fatal("extending the start list should not replace both")
	}
}
