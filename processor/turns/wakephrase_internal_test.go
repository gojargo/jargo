package turns

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// The wake phrase strategy starts turns on its own once awake, and combines
// with other start strategies in a StrategyModeAll list.

// endOnDemandStop ends the turn when a test says so.
type endOnDemandStop struct{ StopStrategyBase }

func (s *endOnDemandStop) Process(frames.Frame) ProcessFrameResult { return Continue }

// wakeHarness runs start strategies under a controller and counts turn starts.
type wakeHarness struct {
	c      *UserTurnController
	stop   *endOnDemandStop
	starts int
}

func newWakeHarness(t *testing.T, startMode StrategyMode, start ...StartStrategy) *wakeHarness {
	t.Helper()
	h := &wakeHarness{stop: &endOnDemandStop{}}
	h.stop.EnableUserSpeakingFrames = true
	h.c = NewUserTurnControllerWithParams(
		UserTurnStrategies{Start: start, Stop: []StopStrategy{h.stop}},
		UserTurnStrategiesParams{StartMode: startMode}, 0)
	h.c.SetHooks(ControllerHooks{
		Started: func(context.Context, StartStrategy, UserTurnStartedParams) { h.starts++ },
	})
	if err := h.c.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.c.Cleanup)
	return h
}

func (h *wakeHarness) end() { h.c.Locked(h.stop.ReportStopped) }

func (h *wakeHarness) transcribe(text string) { h.c.Process(transcript(text)) }

// say is the VAD hearing the user, then the transcript.
func (h *wakeHarness) say(text string) {
	h.c.Process(frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	h.c.Process(frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))
	h.transcribe(text)
}

func (h *wakeHarness) count() int {
	n := 0
	h.c.Locked(func() { n = h.starts })
	return n
}

func newWake(timeout time.Duration) *WakePhraseStart {
	return NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey jargo"}, Timeout: timeout})
}

// With StrategyModeAll, a turn needs the user speaking and the wake phrase.
func newAllWakeHarness(t *testing.T, wakeFirst bool, timeout time.Duration) (*wakeHarness, *WakePhraseStart) {
	t.Helper()
	wake, vad := newWake(timeout), NewVADStart()
	start := []StartStrategy{vad, wake}
	if wakeFirst {
		start = []StartStrategy{wake, vad}
	}
	return newWakeHarness(t, StrategyModeAll, start...), wake
}

func TestWakeAllSpeechWithoutTheWakePhraseStartsNoTurn(t *testing.T) {
	h, _ := newAllWakeHarness(t, false, 10*time.Second)

	h.say("what is the weather")

	if n := h.count(); n != 0 {
		t.Fatalf("starts = %d, want 0", n)
	}
}

func TestWakeAllTheWakePhraseStartsATurnInEitherOrder(t *testing.T) {
	for _, wakeFirst := range []bool{false, true} {
		h, _ := newAllWakeHarness(t, wakeFirst, 10*time.Second)

		h.say("hey jargo what is the weather")

		if n := h.count(); n != 1 {
			t.Fatalf("wake first = %v: starts = %d, want 1", wakeFirst, n)
		}
	}
}

func TestWakeAllSpeechStartsATurnWhileAwake(t *testing.T) {
	h, _ := newAllWakeHarness(t, false, 10*time.Second)
	h.say("hey jargo")
	h.end()

	h.say("what is the weather")

	if n := h.count(); n != 2 {
		t.Fatalf("starts = %d, want 2", n)
	}
}

func TestWakeAllSpeechStartsNoTurnOnceTheWakePhraseTimesOut(t *testing.T) {
	h, wake := newAllWakeHarness(t, false, 50*time.Millisecond)
	h.say("hey jargo")
	h.end()
	time.Sleep(150 * time.Millisecond)

	h.say("what is the weather")

	awake := true
	h.c.Locked(func() { awake = wake.awake })
	if awake {
		t.Fatal("the strategy is still awake after its timeout")
	}
	if n := h.count(); n != 1 {
		t.Fatalf("starts = %d, want 1", n)
	}
}

// On its own, the wake phrase strategy starts turns, even on speech the VAD
// missed.

func TestWakeAloneSpeechWithoutTheWakePhraseStartsNoTurn(t *testing.T) {
	h := newWakeHarness(t, StrategyModeDefault, newWake(10*time.Second))

	h.transcribe("what is the weather")

	if n := h.count(); n != 0 {
		t.Fatalf("starts = %d, want 0", n)
	}
}

func TestWakeAloneTheWakePhraseStartsATurnWithoutVAD(t *testing.T) {
	h := newWakeHarness(t, StrategyModeDefault, newWake(10*time.Second))

	h.transcribe("hey jargo what is the weather")

	if n := h.count(); n != 1 {
		t.Fatalf("starts = %d, want 1", n)
	}
}

func TestWakeAloneATranscriptionStartsATurnWhileAwakeWithoutVAD(t *testing.T) {
	h := newWakeHarness(t, StrategyModeDefault, newWake(10*time.Second))
	h.transcribe("hey jargo")
	h.end()

	h.transcribe("what is the weather")

	if n := h.count(); n != 2 {
		t.Fatalf("starts = %d, want 2", n)
	}
}

// In a StrategyModeAll list, the wake phrase combines with other start
// conditions.
func TestWakeOnlyEnoughWordsInterruptTheBotOnceAwake(t *testing.T) {
	h := newWakeHarness(t, StrategyModeAll, newWake(10*time.Second),
		NewMinWordsStart(MinWordsStartConfig{MinWords: 3}))
	h.transcribe("hey jargo")
	h.end()
	h.c.Process(frames.NewBotStartedSpeakingFrame())

	h.transcribe("uh huh")
	if n := h.count(); n != 1 {
		t.Fatalf("starts = %d after a short reply, want 1", n)
	}

	h.transcribe("wait a second please")
	if n := h.count(); n != 2 {
		t.Fatalf("starts = %d, want 2", n)
	}
}
