package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// reconnectingSynth streams on its own connection and loses it after the first
// sentence: the first context gets part of its audio and never the rest, and
// the provider replaces its connection before the next sentence.
type reconnectingSynth struct {
	mu       sync.Mutex
	host     tts.AudioContextHost
	contexts []string
}

func (s *reconnectingSynth) SampleRate() int { return 24000 }

func (s *reconnectingSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *reconnectingSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	s.contexts = append(s.contexts, contextID)
	first := len(s.contexts) == 1
	host := s.host
	s.mu.Unlock()

	host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 480), 24000, 1))
	if first {
		// The connection drops with the rest of this context's audio still due.
		host.ReplaceConnection()
		return nil
	}
	host.RemoveAudioContext(contextID)
	return nil
}

func (s *reconnectingSynth) contextIDs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.contexts...)
}

// TestReplacingTheConnectionMovesTheTurnToANewContext checks that a provider
// replacing its connection mid-turn closes the contexts open on the old one,
// whose audio already received still plays, and speaks the rest of the turn
// under a new context ID the new connection opens fresh.
func TestReplacingTheConnectionMovesTheTurnToANewContext(t *testing.T) {
	syn := &reconnectingSynth{}
	base := tts.New("ReconnectingTTS", syn)
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	audio := 0
	ended := make(chan struct{})
	var once sync.Once
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.TTSAudioRawFrame:
			mu.Lock()
			audio++
			mu.Unlock()
		case *frames.LLMFullResponseEndFrame:
			once.Do(func() { close(ended) })
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		task.Cancel(context.Background(), "")
		<-done
	})

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("First sentence. Second sentence. "))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	// Well inside the stop-frame timeout: the context the old connection left
	// open is closed, not waited out.
	select {
	case <-ended:
	case <-time.After(2 * time.Second):
		t.Fatal("the reply never finished: a context left open on the old connection held it")
	}

	ids := syn.contextIDs()
	if len(ids) != 2 || ids[0] == ids[1] {
		t.Errorf("context IDs = %v, want the second sentence on a new one", ids)
	}
	mu.Lock()
	defer mu.Unlock()
	if audio != 2 {
		t.Errorf("audio frames = %d, want both, the old connection's included", audio)
	}
}
