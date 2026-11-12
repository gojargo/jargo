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
	s.waitSessions()

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.activeSessions["s1"]; ok {
		t.Error("the session is still listed after its bot panicked, want it forgotten")
	}
}

// TestASessionArrivingAfterShutdownIsRefused checks a session that arrives once
// shutdown is waiting for the running ones is not started, and is forgotten,
// rather than counted against a wait already under way.
func TestASessionArrivingAfterShutdownIsRefused(t *testing.T) {
	ran := make(chan struct{}, 1)
	s := newServer(t.Context(), &Args{}, func(context.Context, Arguments) error {
		ran <- struct{}{}
		return nil
	})
	s.waitSessions()

	s.mu.Lock()
	s.activeSessions["late"] = nil
	s.mu.Unlock()
	s.runBotSession(&RunnerArguments{SessionID: "late"}, "late", true)
	s.waitSessions()

	select {
	case <-ran:
		t.Error("the bot ran for a session that arrived after shutdown")
	default:
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.activeSessions["late"]; ok {
		t.Error("the refused session is still listed, want it forgotten")
	}
}
