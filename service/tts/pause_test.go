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

// pauseHarness runs a TTS base with pausing configured, and reports what reaches
// the end of the pipeline.
type pauseHarness struct {
	task *pipeline.Worker
	base *tts.Base

	mu    sync.Mutex
	texts []string

	runDone chan error
}

// gatedSynth answers on its own receive loop, the way a WebSocket provider
// does, and holds each context's audio until the test opens the gate, so the
// context stays open until then.
type gatedSynth struct {
	mu   sync.Mutex
	host tts.AudioContextHost
	gate chan struct{}
	wg   sync.WaitGroup
}

func (s *gatedSynth) SampleRate() int { return 24000 }

func (s *gatedSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *gatedSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	host := s.host
	s.mu.Unlock()
	s.wg.Go(func() {
		<-s.gate
		host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 480), 24000, 1))
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

func newPauseHarness(t *testing.T, opts tts.PauseOptions) *pauseHarness {
	t.Helper()
	syn := &gatedSynth{gate: make(chan struct{})}
	h := newPauseHarnessWith(t, opts, syn)
	// Registered after the harness's own cleanup, so it runs first and the
	// audio still held is let go before the task is stopped.
	t.Cleanup(func() { close(syn.gate) })
	return h
}

func newPauseHarnessWith(t *testing.T, opts tts.PauseOptions, syn tts.Synthesizer) *pauseHarness {
	t.Helper()

	base := tts.New("PauseTTS", syn)
	base.SetPauseFrameProcessing(opts)

	h := &pauseHarness{base: base, runDone: make(chan error, 1)}
	h.task = pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&h.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.AggregatedTextFrame); ok {
			h.mu.Lock()
			h.texts = append(h.texts, tf.Text)
			h.mu.Unlock()
		}
	})

	go func() { h.runDone <- h.task.Run(context.Background()) }()
	t.Cleanup(func() {
		h.task.Cancel(t.Context(), "")
		select {
		case <-h.runDone:
		case <-time.After(3 * time.Second):
			t.Error("task did not stop")
		}
	})
	return h
}

// speakTurn feeds one LLM turn carrying text.
func (h *pauseHarness) speakTurn(text string) {
	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(frames.NewLLMTextFrame(text))
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
}

// spoke reports the texts the service announced as about to be spoken.
func (h *pauseHarness) spoke() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.texts...)
}

// waitForSpoken waits until n texts have been announced.
func (h *pauseHarness) waitForSpoken(t *testing.T, n int, within time.Duration) []string {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		got := h.spoke()
		if len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			return got
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestPauseHoldsTheNextTurnUntilTheAudioContextEnds covers what pausing is for:
// a provider whose audio does not carry its request's context ID labels it with
// the context still open, so the next request waits for every open context to
// end, rather than for the bot to stop speaking.
func TestPauseHoldsTheNextTurnUntilTheAudioContextEnds(t *testing.T) {
	syn := &gatedSynth{gate: make(chan struct{})}
	h := newPauseHarnessWith(t, tts.PauseOptions{Enabled: true}, syn)

	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	// The first turn's context is still open, waiting on its audio.
	h.speakTurn("Second one.")
	if got := h.waitForSpoken(t, 2, 500*time.Millisecond); len(got) != 1 {
		t.Fatalf("second turn was synthesized while a context was open: %v", got)
	}

	// The audio arrives and the context ends: the held turn is released, in
	// order, with no bot-speaking frame involved.
	close(syn.gate)

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 {
		t.Fatalf("held turn was never released: %v", got)
	}
	if got[0] != "First one." || got[1] != "Second one." {
		t.Errorf("spoken = %v, want [First one. Second one.]", got)
	}
}

// TestPauseHoldsTheNextFramesAfterAStandaloneSpeak covers a TTSSpeakFrame
// outside an LLM turn, which holds the frames after it the way the end of a
// reply does.
func TestPauseHoldsTheNextFramesAfterAStandaloneSpeak(t *testing.T) {
	syn := &gatedSynth{gate: make(chan struct{})}
	h := newPauseHarnessWith(t, tts.PauseOptions{Enabled: true}, syn)

	h.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("the speak frame was not spoken: %v", got)
	}
	h.speakTurn("Next one.")
	if got := h.waitForSpoken(t, 2, 500*time.Millisecond); len(got) != 1 {
		t.Fatalf("the next turn ran while the speak frame's context was open: %v", got)
	}

	close(syn.gate)
	if got := h.waitForSpoken(t, 2, 3*time.Second); len(got) != 2 {
		t.Fatalf("held turn was never released: %v", got)
	}
}

func TestPauseDisabledByDefault(t *testing.T) {
	// The audio arrives at once, so only a pause could hold the second turn.
	syn := &gatedSynth{gate: make(chan struct{})}
	close(syn.gate)
	h := newPauseHarnessWith(t, tts.PauseOptions{}, syn)

	h.speakTurn("First one.")
	h.speakTurn("Second one.")

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 {
		t.Fatalf("spoken = %v, want both turns with pausing off", got)
	}
}

func TestPauseSkippedWhenTheTurnSentNoText(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true})

	// A turn that carries no text, a function call and nothing else, has no
	// audio to wait for, so it must not pause the service.
	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	h.speakTurn("Straight through.")
	got := h.waitForSpoken(t, 1, 3*time.Second)
	if len(got) != 1 || got[0] != "Straight through." {
		t.Errorf("spoken = %v, want [Straight through.]", got)
	}
}

func TestInterruptionReleasesAPausedService(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true})

	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	// The interruption drops the open context, which never ends on its own, so
	// the interruption itself has to release the service.
	h.task.QueueFrame(frames.NewInterruptionFrame())
	h.speakTurn("After the barge-in.")

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 || got[1] != "After the barge-in." {
		t.Errorf("spoken = %v, want the turn after the interruption to run", got)
	}
}
