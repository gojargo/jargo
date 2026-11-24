package turns_test

import (
	"testing"

	"github.com/gojargo/jargo/processor/turns"
)

// TestItBuildsEagerStrategiesOnlyWhenEnabled covers the eager strategies being
// built only for a service configured to predict the end of a turn: with eager
// end of turn off, the recommendation is left to the service.
func TestItBuildsEagerStrategiesOnlyWhenEnabled(t *testing.T) {
	strategies := turns.EagerUserTurnStrategies(true, turns.EagerStrategiesConfig{})
	if strategies == nil {
		t.Fatal("no strategies while eager end of turn is on")
	}
	if len(strategies.Stop) != 1 {
		t.Fatalf("stop chain = %d strategies, want 1", len(strategies.Stop))
	}
	if _, ok := strategies.Stop[0].(*turns.EagerStop); !ok {
		t.Errorf("stop strategy = %T, want *turns.EagerStop", strategies.Stop[0])
	}

	if got := turns.EagerUserTurnStrategies(false, turns.EagerStrategiesConfig{}); got != nil {
		t.Errorf("strategies while eager end of turn is off = %+v, want nil", got)
	}
}

// TestTheEagerStrategiesCarryTheMatchPolicyAndInterruptions covers the match
// policy and the interruption setting reaching the strategies built.
func TestTheEagerStrategiesCarryTheMatchPolicyAndInterruptions(t *testing.T) {
	interrupt := false
	strategies := turns.EagerUserTurnStrategies(true, turns.EagerStrategiesConfig{
		EnableInterruptions: &interrupt,
		MatchPolicy:         turns.ExactMatch{},
	})
	if strategies == nil {
		t.Fatal("no strategies while eager end of turn is on")
	}

	stop, ok := strategies.Stop[0].(*turns.EagerStop)
	if !ok {
		t.Fatalf("stop strategy = %T, want *turns.EagerStop", strategies.Stop[0])
	}
	if _, ok := stop.MatchPolicy().(turns.ExactMatch); !ok {
		t.Errorf("match policy = %T, want turns.ExactMatch", stop.MatchPolicy())
	}
	enabled, external := strategies.ExternalInterruptions()
	if !external || enabled {
		t.Errorf("ExternalInterruptions() = (%v, %v), want (false, true)", enabled, external)
	}
}
