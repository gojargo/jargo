package main

import (
	"testing"

	"github.com/gojargo/jargo/eval"
)

// TestGreeting plays the greeting scenario against the demo bot in-process — the
// same scenario `jargo eval run` drives over a WebSocket.
func TestGreeting(t *testing.T) {
	eval.Run(t, "scenarios/greeting.yaml", buildBot)
}
