package runner

import (
	"context"
	"testing"
)

// TestABotThatPanicsEndsOnlyItsSession checks a bot that panics is that
// session's failure alone: the panic is recovered, the session is forgotten as
// any ended session is, and the runner carries on serving.
func TestABotThatPanicsEndsOnlyItsSession(t *testing.T) {
	s := newServer(t.Context(), &Args{}, func(context.Context, Arguments) error {
		var sessions []string
		_ = sessions[len(sessions)] //nolint:gocritic,gosec // the bot's bug: an index out of range
		return nil
	})
	s.mu.Lock()
	s.activeSessions["s1"] = nil
	s.mu.Unlock()

	s.runBotSession(&RunnerArguments{SessionID: "s1"}, "s1", true)
	s.sessions.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.activeSessions["s1"]; ok {
		t.Error("the session is still listed after its bot panicked, want it forgotten")
	}
}
