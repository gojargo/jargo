package tts

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// turnSynth records the turns it is told were opened and completed.
type turnSynth struct {
	mu        sync.Mutex
	created   []string
	completed []string
}

func (s *turnSynth) SampleRate() int { return wsSampleRate }

func (s *turnSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *turnSynth) OnTurnContextCreated(_ context.Context, contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.created = append(s.created, contextID)
}

func (s *turnSynth) OnTurnContextCompleted(_ context.Context, contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.completed = append(s.completed, contextID)
}

// TestATurnThatSpokeNothingIsCompleted checks a provider that opened its
// context as the turn started is told the turn is over even when no text ever
// reached it, so it can close what it opened.
func TestATurnThatSpokeNothingIsCompleted(t *testing.T) {
	syn := &turnSynth{}
	runBase(t, New("TurnTTS", syn),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMFullResponseEndFrame(),
	)
	syn.mu.Lock()
	defer syn.mu.Unlock()
	if len(syn.created) != 1 || !slices.Equal(syn.completed, syn.created) {
		t.Errorf("created %q, completed %q, want the turn completed", syn.created, syn.completed)
	}
}
