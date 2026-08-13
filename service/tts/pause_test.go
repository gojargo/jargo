package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// pauseHarness runs a TTS base with pausing configured, and reports what reaches
// the end of the pipeline.
type pauseHarness struct {
	task *pipeline.Worker
	base *tts.Base

	mu    sync.Mutex
	texts []string
	errs  []*frames.ErrorFrame

	runDone chan error
}

func newPauseHarness(t *testing.T, opts tts.PauseOptions) *pauseHarness {
	t.Helper()

	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 8)}
	base := tts.New("PauseTTS", syn)
	base.SetPauseFrameProcessing(opts)

	h := &pauseHarness{base: base, runDone: make(chan error, 1)}
	h.task = pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&h.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.AggregatedTextFrame); ok {
			h.mu.Lock()
			h.texts = append(h.texts, tf.Text)
			h.mu.Unlock()
		}
	})
	events.On(&h.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			h.mu.Lock()
			h.errs = append(h.errs, ef)
			h.mu.Unlock()
		}
	})

	// Drain what the fake synthesizer reports so it never blocks.
	go func() {
		for range syn.spoken {
		}
	}()

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

func (h *pauseHarness) errors() []*frames.ErrorFrame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]*frames.ErrorFrame(nil), h.errs...)
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

func TestPauseHoldsTheNextTurnUntilTheAudioFinishes(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true, WatchdogTimeout: 5 * time.Second})

	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	// The service is paused now, waiting for the audio to play out.
	h.speakTurn("Second one.")
	if got := h.waitForSpoken(t, 2, 500*time.Millisecond); len(got) != 1 {
		t.Fatalf("second turn was synthesized while paused: %v", got)
	}

	// Playback finished: the held turn is released, in order.
	h.task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	h.task.QueueFrame(frames.NewBotStoppedSpeakingFrame())

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 {
		t.Fatalf("held turn was never released: %v", got)
	}
	if got[0] != "First one." || got[1] != "Second one." {
		t.Errorf("spoken = %v, want [First one. Second one.]", got)
	}
}

func TestPauseDisabledByDefault(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{})

	h.speakTurn("First one.")
	h.speakTurn("Second one.")

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 {
		t.Fatalf("spoken = %v, want both turns with pausing off", got)
	}
}

func TestPauseSkippedWhenTheTurnSentNoText(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true, WatchdogTimeout: 5 * time.Second})

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

func TestPauseWatchdogForceResumes(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true, WatchdogTimeout: 300 * time.Millisecond})

	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	// Nothing ever confirms the bot spoke, so the watchdog has to release the
	// service rather than leave it paused for good.
	h.speakTurn("Second one.")
	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 {
		t.Fatalf("watchdog never force-resumed: %v", got)
	}

	errs := h.errors()
	if len(errs) == 0 {
		t.Fatal("watchdog force-resumed without reporting an error")
	}
	if errs[0].Fatal {
		t.Error("watchdog error is fatal, want non-fatal")
	}
}

func TestPauseWatchdogNotArmedWhileTheBotIsSpeaking(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true, WatchdogTimeout: 300 * time.Millisecond})

	// Playback for this turn is already confirmed, which is the usual case for a
	// streaming provider, so the pause waits for BotStoppedSpeakingFrame and the
	// watchdog must not fire behind it.
	h.task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	h.speakTurn("Second one.")
	if got := h.waitForSpoken(t, 2, 1*time.Second); len(got) != 1 {
		t.Fatalf("second turn ran while the bot was still speaking: %v", got)
	}
	if errs := h.errors(); len(errs) != 0 {
		t.Errorf("watchdog fired while the bot was speaking: %v", errs[0].Error)
	}

	h.task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	if got := h.waitForSpoken(t, 2, 3*time.Second); len(got) != 2 {
		t.Fatalf("held turn was never released: %v", got)
	}
}

func TestInterruptionReleasesAPausedService(t *testing.T) {
	h := newPauseHarness(t, tts.PauseOptions{Enabled: true, WatchdogTimeout: 5 * time.Second})

	h.speakTurn("First one.")
	if got := h.waitForSpoken(t, 1, 2*time.Second); len(got) != 1 {
		t.Fatalf("first turn was not spoken: %v", got)
	}

	// No BotStoppedSpeakingFrame is coming for audio the interruption cut off,
	// so the interruption itself has to release the service.
	h.task.QueueFrame(frames.NewInterruptionFrame())
	h.speakTurn("After the barge-in.")

	got := h.waitForSpoken(t, 2, 3*time.Second)
	if len(got) != 2 || got[1] != "After the barge-in." {
		t.Errorf("spoken = %v, want the turn after the interruption to run", got)
	}
}
