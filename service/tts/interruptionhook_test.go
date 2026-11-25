package tts_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// hookSynth answers on its own receive loop, one chunk at a time and slowly, so
// an interruption lands while its context is still playing. It records which
// contexts the base reports interrupted or completed, and whether each context
// was still open when it was reported interrupted.
type hookSynth struct {
	mu          sync.Mutex
	host        tts.AudioContextHost
	requests    []string
	interrupted []string
	openOnHook  []bool
	completed   []string
	wg          sync.WaitGroup
}

var (
	_ tts.AudioContextInterrupter = (*hookSynth)(nil)
	_ tts.AudioContextCompleter   = (*hookSynth)(nil)
)

func (s *hookSynth) SampleRate() int { return orderingSampleRate }

func (s *hookSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *hookSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	host := s.host
	s.requests = append(s.requests, contextID)
	s.mu.Unlock()
	s.wg.Go(func() {
		for range 10 {
			time.Sleep(20 * time.Millisecond)
			if !host.AudioContextAvailable(contextID) {
				return
			}
			audio := frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1)
			audio.ContextID = contextID
			host.AppendToAudioContext(contextID, audio)
		}
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

func (s *hookSynth) OnAudioContextInterrupted(_ context.Context, contextID string) {
	s.mu.Lock()
	host := s.host
	s.mu.Unlock()
	open := host.AudioContextAvailable(contextID)
	s.mu.Lock()
	s.interrupted = append(s.interrupted, contextID)
	s.openOnHook = append(s.openOnHook, open)
	s.mu.Unlock()
}

func (s *hookSynth) OnAudioContextCompleted(_ context.Context, contextID string) {
	s.mu.Lock()
	s.completed = append(s.completed, contextID)
	s.mu.Unlock()
}

// TestInterruptionReportsOpenContextsToProvider covers the provider hook an
// interruption runs: the context being played and the ones queued behind it are
// each reported interrupted, in the order they were created and while still
// open, and none is reported completed. Speech requested after the
// interruption still plays.
func TestInterruptionReportsOpenContextsToProvider(t *testing.T) {
	synth := &hookSynth{}
	base := tts.New("HookTTS", synth)
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	audioFor := map[string]int{}
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			mu.Lock()
			audioFor[a.ContextID]++
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("One."))
	task.QueueFrame(frames.NewTTSSpeakFrame("Two."))
	task.QueueFrame(frames.NewTTSSpeakFrame("Three."))
	task.QueueFrame(frames.NewTTSSpeakFrame("Four."))
	time.Sleep(80 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(50 * time.Millisecond)
	task.QueueFrame(frames.NewTTSSpeakFrame("Five."))
	time.Sleep(400 * time.Millisecond)
	task.QueueFrame(frames.NewEndFrame())
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not end")
	}
	synth.wg.Wait()

	synth.mu.Lock()
	defer synth.mu.Unlock()
	if len(synth.requests) != 5 {
		t.Fatalf("requests = %v, want five", synth.requests)
	}
	// The first is playing and the next three are queued behind it; all four
	// are reported, in the order they were created.
	cut, after := synth.requests[:4], synth.requests[4]
	if !slices.Equal(synth.interrupted, cut) {
		t.Fatalf("interrupted = %v, want the playing and the queued contexts in creation order %v",
			synth.interrupted, cut)
	}
	for i, open := range synth.openOnHook {
		if !open {
			t.Errorf("context %s was already removed when reported interrupted", synth.interrupted[i])
		}
	}
	for _, id := range cut {
		if slices.Contains(synth.completed, id) {
			t.Errorf("completed = %v, an interrupted context must not be reported completed", synth.completed)
		}
	}
	if !slices.Contains(synth.completed, after) {
		t.Errorf("completed = %v, want the speech after the interruption %s", synth.completed, after)
	}
	mu.Lock()
	defer mu.Unlock()
	if audioFor[after] == 0 {
		t.Errorf("no audio played for the speech requested after the interruption")
	}
}
