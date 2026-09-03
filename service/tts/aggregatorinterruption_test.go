package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// clearingSpy records which of the aggregator's two clearing hooks it was
// given, so a test can tell an interruption from a reset.
type clearingSpy struct {
	*ttstext.SimpleAggregator

	mu            sync.Mutex
	interruptions int
	resets        int
}

func newClearingSpy(t *testing.T) *clearingSpy {
	t.Helper()
	tok, err := ttstext.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return &clearingSpy{
		SimpleAggregator: ttstext.NewSimpleAggregator(frames.AggregationSentence, tok),
	}
}

func (a *clearingSpy) HandleInterruption() {
	a.mu.Lock()
	a.interruptions++
	a.mu.Unlock()
	a.SimpleAggregator.HandleInterruption()
}

func (a *clearingSpy) Reset() {
	a.mu.Lock()
	a.resets++
	a.mu.Unlock()
	a.SimpleAggregator.Reset()
}

func (a *clearingSpy) counts() (interruptions, resets int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.interruptions, a.resets
}

// A barge-in reaches the text aggregator as an interruption, not as a reset.
// The two are separate hooks so an aggregator can treat them differently, and
// an aggregator that does would never hear about the barge-in otherwise.
func TestInterruptionTellsTheTextAggregatorItWasInterrupted(t *testing.T) {
	spy := newClearingSpy(t)
	base := tts.New("SpyTTS", &spacedSynth{})
	base.SetTextAggregator(spy)

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("half a "))
	task.QueueFrame(frames.NewInterruptionFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not stop")
	}

	interruptions, resets := spy.counts()
	if interruptions != 1 {
		t.Errorf("aggregator was interrupted %d times, want 1", interruptions)
	}
	if resets != 0 {
		t.Errorf("aggregator was reset %d times, want 0: an interruption is not a reset", resets)
	}
}
