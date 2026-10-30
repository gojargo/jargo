package turns_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
)

// decidingStop is a stop strategy that decides the user is done when a test
// says so.
type decidingStop struct {
	turns.StopStrategyBase
	frames      []frames.Frame
	result      turns.ProcessFrameResult
	onInference bool
}

func newDecidingStop() *decidingStop {
	s := &decidingStop{}
	s.EnableUserSpeakingFrames = true
	return s
}

// newWaitingStop is a decidingStop that waits for the turn's inference.
func newWaitingStop() *decidingStop {
	s := newDecidingStop()
	s.onInference = true
	return s
}

func (s *decidingStop) WaitsForInference() bool { return s.onInference }

func (s *decidingStop) Process(f frames.Frame) turns.ProcessFrameResult {
	s.frames = append(s.frames, f)
	return s.result
}

func (s *decidingStop) decide()    { s.ReportStopped() }
func (s *decidingStop) speculate() { s.StartSpeculation(turns.UserTurnSpeculation{Text: "Hi"}) }
func (s *decidingStop) confirm() {
	s.ReportStoppedOverriding(turns.StoppedOverrides{ConfirmsSpeculation: true})
}

// decidingStart is a start strategy that decides the user started when a test
// says so.
type decidingStart struct {
	turns.StartStrategyBase
	frames []frames.Frame
	result turns.ProcessFrameResult
}

func newDecidingStart() *decidingStart {
	s := &decidingStart{}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

func (s *decidingStart) Process(f frames.Frame) turns.ProcessFrameResult {
	s.frames = append(s.frames, f)
	return s.result
}

func (s *decidingStart) decide() { s.ReportStarted() }

// scheduledStop has a decision of its own scheduled while scheduled is set.
type scheduledStop struct {
	turns.StopStrategyBase
	scheduled bool
}

func (s *scheduledStop) Process(frames.Frame) turns.ProcessFrameResult { return turns.Continue }
func (s *scheduledStop) DecisionScheduled() bool                       { return s.scheduled }

type stopEvent struct {
	strategy turns.StopStrategy
	params   turns.UserTurnStoppedParams
}

type startEvent struct {
	strategy turns.StartStrategy
	params   turns.UserTurnStartedParams
}

type inferenceEvent struct {
	strategy    turns.StopStrategy
	speculation *turns.UserTurnSpeculation
}

// harness drives a controller directly and records what it reports.
type harness struct {
	t          *testing.T
	c          *turns.UserTurnController
	starts     []startEvent
	stops      []stopEvent
	inferences []inferenceEvent
	resumes    []turns.UserTurnResumedParams
	timeouts   int
	// events names every inference and turn end, in order.
	events []string
}

func newHarness(
	t *testing.T, strategies turns.UserTurnStrategies, params turns.UserTurnStrategiesParams,
	stopTimeout time.Duration,
) *harness {
	t.Helper()
	h := &harness{t: t}
	h.c = turns.NewUserTurnControllerWithParams(strategies, params, stopTimeout)
	h.c.SetHooks(turns.ControllerHooks{
		Started: func(_ context.Context, s turns.StartStrategy, p turns.UserTurnStartedParams) {
			h.starts = append(h.starts, startEvent{s, p})
		},
		Stopped: func(_ context.Context, s turns.StopStrategy, p turns.UserTurnStoppedParams) {
			h.stops = append(h.stops, stopEvent{s, p})
			h.events = append(h.events, "stopped")
		},
		InferenceTriggered: func(_ context.Context, s turns.StopStrategy, sp *turns.UserTurnSpeculation) {
			h.inferences = append(h.inferences, inferenceEvent{s, sp})
			h.events = append(h.events, "inference_triggered")
		},
		Resumed: func(_ context.Context, p turns.UserTurnResumedParams) {
			h.resumes = append(h.resumes, p)
		},
		StopTimeout: func(context.Context) { h.timeouts++ },
	})
	if err := h.c.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(h.c.Cleanup)
	return h
}

// newStopHarness runs stop under a VAD start, as the stop mode tests do.
func newStopHarness(t *testing.T, stopMode turns.StrategyMode, stop ...turns.StopStrategy) *harness {
	t.Helper()
	return newHarness(t, turns.UserTurnStrategies{
		Start: []turns.StartStrategy{turns.NewVADStart()},
		Stop:  stop,
	}, turns.UserTurnStrategiesParams{StopMode: stopMode}, 0)
}

// do runs fn as a strategy's timer would, under the controller's lock.
func (h *harness) do(fn func()) { h.c.Locked(fn) }

func (h *harness) send(fs ...frames.Frame) {
	for _, f := range fs {
		h.c.Process(f)
	}
}

// speak is the user speaking and stopping.
func (h *harness) speak() {
	h.send(frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))
}

// read runs fn under the lock, for state the controller's timers write.
func (h *harness) read(fn func()) { h.c.Locked(fn) }

func (h *harness) stoppedBy() []turns.StopStrategy {
	out := make([]turns.StopStrategy, 0, len(h.stops))
	for _, e := range h.stops {
		out = append(out, e.strategy)
	}
	return out
}

func (h *harness) speculative() []bool {
	out := make([]bool, 0, len(h.inferences))
	for _, e := range h.inferences {
		out = append(out, e.speculation != nil)
	}
	return out
}

func vadStarted() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(0, time.Now()) }
func vadStopped() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(0, time.Now()) }

// captureWarnings points the default logger at a buffer for the test.
func captureWarnings(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &buf
}

func sameStrategies(got []turns.StopStrategy, want ...turns.StopStrategy) bool {
	return equalLists(got, want)
}

const shortStopTimeout = 100 * time.Millisecond

// The stop timeout ends the turn with what the decisions said, a confirmation
// included.
func TestATurnEndedByTheTimeoutKeepsAConfirmation(t *testing.T) {
	confirming, onInference := newDecidingStop(), newWaitingStop()
	h := newHarness(t, turns.UserTurnStrategies{Stop: []turns.StopStrategy{confirming, onInference}},
		turns.UserTurnStrategiesParams{}, shortStopTimeout)

	h.speak()
	h.do(confirming.speculate)
	h.do(confirming.confirm)
	time.Sleep(shortStopTimeout + 100*time.Millisecond)

	h.read(func() {
		if len(h.stops) != 1 || !h.stops[0].params.ConfirmsSpeculation {
			t.Fatalf("stops = %+v, want one confirming the speculation", h.stops)
		}
	})
}

func TestTheStopTimeoutWaitsForAScheduledDecision(t *testing.T) {
	scheduled := &scheduledStop{scheduled: true}
	h := newHarness(t, turns.UserTurnStrategies{Stop: []turns.StopStrategy{scheduled}},
		turns.UserTurnStrategiesParams{}, shortStopTimeout)

	h.speak()
	time.Sleep(2*shortStopTimeout + 100*time.Millisecond)
	h.read(func() {
		if len(h.stops) != 0 {
			t.Fatalf("stops = %d, want the timeout to wait for the scheduled decision", len(h.stops))
		}
		scheduled.scheduled = false
	})

	time.Sleep(shortStopTimeout + 100*time.Millisecond)
	h.read(func() {
		if len(h.stops) != 1 || h.stops[0].strategy != nil {
			t.Fatalf("stops = %+v, want the timeout to end the turn", h.stops)
		}
	})
}

// With the first decision winning, a deferred strategy is the one it wraps, and
// its decision runs the inference first.
func TestDeferredWrapperSkipsStoppedInAFirstList(t *testing.T) {
	wrapped := turns.Deferred( //nolint:staticcheck // the deprecated wrapper is what is tested
		turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 50 * time.Millisecond}))
	h := newStopHarness(t, turns.StrategyModeFirst, wrapped)

	h.send(vadStarted(), finalTranscript("Hello!"), vadStopped())
	time.Sleep(150 * time.Millisecond)

	h.read(func() {
		if len(h.events) == 0 || h.events[0] != "inference_triggered" {
			t.Fatalf("events = %v, want the inference first", h.events)
		}
	})
}

// The service says the user is done while the local VAD still hears a breath.
func TestAnExternalTurnDecidedWhileTheVADHearsSpeechEndsOnceTheVADStops(t *testing.T) {
	for _, mode := range []turns.StrategyMode{turns.StrategyModeAll, turns.StrategyModeFirst} {
		h := newHarness(t, turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
			turns.UserTurnStrategiesParams{StopMode: mode}, 3*time.Second)

		h.send(frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame(),
			vadStarted(), frames.NewTranscriptionFrame("Hello", "", "now"))
		// The transcript settles, and the strategy decides while the VAD still
		// hears speech.
		time.Sleep(700 * time.Millisecond)
		h.read(func() {
			if len(h.stops) != 0 {
				t.Fatalf("mode %v: the turn ended while the VAD heard speech", mode)
			}
		})

		h.send(vadStopped())

		h.read(func() {
			if len(h.stops) != 1 || h.timeouts != 0 {
				t.Fatalf("mode %v: stops = %d, timeouts = %d, want 1 and 0", mode, len(h.stops), h.timeouts)
			}
		})
	}
}

func TestTheTurnEndsOnceEveryStrategyDecides(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, last)
	h.speak()

	h.do(first.decide)
	if len(h.stops) != 0 {
		t.Fatal("one decision of two ended the turn")
	}
	h.do(last.decide)
	if !sameStrategies(h.stoppedBy(), last) {
		t.Fatalf("stopped by %v, want the last to decide", h.stoppedBy())
	}
}

func TestTheFirstDecisionEndsTheTurnInFirstMode(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeFirst, first, last)
	h.speak()

	h.do(first.decide)
	h.do(last.decide)

	if !sameStrategies(h.stoppedBy(), first) {
		t.Fatalf("stopped by %v, want the first to decide", h.stoppedBy())
	}
}

func TestAFirstListWaitsForTheDecisionOnTheInference(t *testing.T) {
	first, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeFirst, first, onInference)
	h.speak()

	h.do(first.decide)
	if len(h.inferences) != 1 || len(h.stops) != 0 {
		t.Fatalf("inferences = %d, stops = %d, want 1 and 0", len(h.inferences), len(h.stops))
	}
	h.do(onInference.decide)
	if !sameStrategies(h.stoppedBy(), onInference) {
		t.Fatalf("stopped by %v, want the strategy waiting for the inference", h.stoppedBy())
	}
}

func TestAFirstListKeepsAConfirmationUntilTheTurnEnds(t *testing.T) {
	first, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeFirst, first, onInference)
	h.speak()

	h.do(first.speculate)
	h.do(first.confirm)
	h.do(onInference.decide)

	if len(h.stops) != 1 || !h.stops[0].params.ConfirmsSpeculation {
		t.Fatalf("stops = %+v, want the turn end to confirm the speculation", h.stops)
	}
}

func TestNewSpeechClearsTheDecisions(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, last)
	h.speak()
	h.do(first.decide)

	h.speak()
	h.do(last.decide)
	if len(h.stops) != 0 {
		t.Fatal("a decision made before the user spoke again still counted")
	}
	h.do(first.decide)
	if !sameStrategies(h.stoppedBy(), first) {
		t.Fatalf("stopped by %v, want the first", h.stoppedBy())
	}
}

func TestADecisionMadeWhileTheUserSpeaksCounts(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, last)
	h.send(vadStarted())
	h.do(first.decide)
	h.send(vadStopped())

	h.do(last.decide)

	if !sameStrategies(h.stoppedBy(), last) {
		t.Fatalf("stopped by %v, want the last", h.stoppedBy())
	}
}

func TestTheTurnDoesNotEndWhileTheUserSpeaks(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, last)
	h.send(vadStarted())

	h.do(first.decide)
	h.do(last.decide)

	if len(h.stops) != 0 {
		t.Fatal("the turn ended while the user was speaking")
	}
}

func TestATurnDecidedWhileTheUserSpeaksEndsOnceTheyStop(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, last)
	h.send(vadStarted())
	h.do(first.decide)
	h.do(last.decide)

	h.send(vadStopped())

	if !sameStrategies(h.stoppedBy(), last) {
		t.Fatalf("stopped by %v, want the last", h.stoppedBy())
	}
}

func TestAFirstListEndsATurnDecidedWhileTheUserSpeaksOnceTheyStop(t *testing.T) {
	strategy := newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeFirst, strategy)
	h.send(vadStarted())
	h.do(strategy.decide)
	if len(h.stops) != 0 {
		t.Fatal("the turn ended while the user was speaking")
	}

	h.send(vadStopped())

	if !sameStrategies(h.stoppedBy(), strategy) {
		t.Fatalf("stopped by %v, want the strategy", h.stoppedBy())
	}
}

// A decision on the inference is about the answer, so it goes with it.
func TestACanceledAnswerVoidsTheDecisionsMadeOnIt(t *testing.T) {
	user, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, user, onInference)
	h.speak()
	h.do(user.speculate)
	h.do(onInference.decide)

	h.send(frames.NewUserTurnResponseCancelFrame())
	h.do(user.decide)

	if len(h.stops) != 0 {
		t.Fatal("a decision made on a canceled answer still ended the turn")
	}
}

// A decision on the user says nothing about the answer, so it outlives it.
func TestACanceledAnswerKeepsTheDecisionsOnTheUser(t *testing.T) {
	user, eager, onInference := newDecidingStop(), newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, user, eager, onInference)
	h.speak()
	h.do(eager.speculate)
	h.do(user.decide)

	h.send(frames.NewUserTurnResponseCancelFrame())
	h.do(eager.decide)

	if got := h.speculative(); !equalLists(got, []bool{true, false}) {
		t.Fatalf("speculative inferences = %v, want a speculation then the turn's own", got)
	}
}

// The turn's answer runs again for a decision that confirmed a canceled one.
func TestAConfirmationOutlivesTheAnswerItConfirmed(t *testing.T) {
	confirming, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, confirming, onInference)
	h.speak()
	h.do(confirming.speculate)
	h.do(confirming.confirm)
	h.send(frames.NewUserTurnResponseCancelFrame())

	if len(h.inferences) != 2 || h.inferences[0].strategy != confirming ||
		h.inferences[1].strategy != confirming {
		t.Fatalf("inferences = %+v, want two from the confirming strategy", h.inferences)
	}
	h.do(onInference.decide)
	if len(h.stops) != 1 || h.stops[0].params.ConfirmsSpeculation {
		t.Fatalf("stops = %+v, want one confirming nothing", h.stops)
	}
}

func TestAWithdrawnSpeculationReachesEveryStopStrategy(t *testing.T) {
	withdrawing, other := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, withdrawing, other)
	h.speak()

	h.do(withdrawing.CancelSpeculation)

	if _, ok := other.frames[len(other.frames)-1].(*frames.UserTurnResponseCancelFrame); !ok {
		t.Fatalf("last frame = %v, want the withdrawal", other.frames[len(other.frames)-1])
	}
}

// Decisions do not carry over into a turn that starts without speech frames.
func TestEachTurnNeedsItsOwnStopDecisions(t *testing.T) {
	first, last := newDecidingStop(), newDecidingStop()
	h := newHarness(t, turns.UserTurnStrategies{
		Start: []turns.StartStrategy{turns.NewTranscriptionStart(turns.TranscriptionStartConfig{})},
		Stop:  []turns.StopStrategy{first, last},
	}, turns.UserTurnStrategiesParams{}, 0)
	h.send(frames.NewTranscriptionFrame("Hi", "", ""))
	h.do(first.decide)
	h.do(last.decide)

	h.send(frames.NewTranscriptionFrame("Hi", "", ""))
	h.do(last.decide)

	if len(h.stops) != 1 {
		t.Fatalf("stops = %d, want 1", len(h.stops))
	}
}

func TestADeferredStrategyDecidesAsTheOneItWraps(t *testing.T) {
	inner, other := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault,
		turns.Deferred(inner), other) //nolint:staticcheck // the deprecated wrapper is what is tested
	h.speak()

	h.do(other.decide)
	if len(h.stops) != 0 {
		t.Fatal("the turn ended before the wrapped strategy decided")
	}
	h.do(inner.decide)
	if !sameStrategies(h.stoppedBy(), inner) {
		t.Fatalf("stopped by %v, want the wrapped strategy", h.stoppedBy())
	}
}

func TestEveryStopStrategySeesEveryFrame(t *testing.T) {
	blocking, other := newDecidingStop(), newDecidingStop()
	blocking.result = turns.Stop
	h := newStopHarness(t, turns.StrategyModeDefault, blocking, other)

	h.speak()

	if len(other.frames) != 2 {
		t.Fatalf("frames seen = %d, want 2", len(other.frames))
	}
}

func TestTheInferenceWaitsForEveryStrategyThatListensToTheUser(t *testing.T) {
	first, second, onInference := newDecidingStop(), newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, second, onInference)
	h.speak()

	h.do(first.decide)
	if len(h.inferences) != 0 {
		t.Fatal("the inference ran before every strategy listening to the user decided")
	}
	h.do(second.decide)
	if len(h.inferences) != 1 || h.inferences[0].strategy != second || len(h.stops) != 0 {
		t.Fatalf("inferences = %+v, stops = %d, want one from the second and none", h.inferences, len(h.stops))
	}
	h.do(onInference.decide)
	if !sameStrategies(h.stoppedBy(), onInference) {
		t.Fatalf("stopped by %v, want the strategy waiting for the inference", h.stoppedBy())
	}
}

// With no stop strategy listening to the user, only the stop timeout runs the
// inference.
func TestAListThatOnlyWaitsForTheInferenceWarns(t *testing.T) {
	logs := captureWarnings(t)
	h := newStopHarness(t, turns.StrategyModeDefault, newWaitingStop())
	if err := h.c.UpdateStrategies(turns.UserTurnStrategies{
		Stop: []turns.StopStrategy{newDecidingStop(), newWaitingStop()},
	}); err != nil {
		t.Fatal(err)
	}

	if n := strings.Count(logs.String(), "waits for the turn's inference"); n != 1 {
		t.Fatalf("warnings = %d, want 1:\n%s", n, logs.String())
	}
}

func TestTheInferenceRunsOncePerRoundOfDecisions(t *testing.T) {
	first, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, onInference)
	h.speak()

	h.do(first.decide)
	h.do(first.decide)
	if len(h.inferences) != 1 {
		t.Fatalf("inferences = %d, want 1", len(h.inferences))
	}
	h.speak()
	h.do(first.decide)
	if len(h.inferences) != 2 {
		t.Fatalf("inferences = %d, want 2", len(h.inferences))
	}
}

func TestASpeculationIsTheTurnsAnswer(t *testing.T) {
	eager, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, eager, onInference)
	h.speak()

	h.do(eager.speculate)
	h.do(eager.confirm)

	if got := h.speculative(); !equalLists(got, []bool{true}) {
		t.Fatalf("speculative inferences = %v, want only the speculation", got)
	}
}

func TestAWithdrawnSpeculationLetsTheInferenceRun(t *testing.T) {
	eager, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, eager, onInference)
	h.speak()

	h.do(eager.speculate)
	h.do(eager.CancelSpeculation)
	h.do(eager.decide)

	if got := h.speculative(); !equalLists(got, []bool{true, false}) {
		t.Fatalf("speculative inferences = %v, want a speculation then the turn's own", got)
	}
}

func TestSpeakingAgainInAnOpenTurnResumesIt(t *testing.T) {
	h := newStopHarness(t, turns.StrategyModeDefault, newDecidingStop(), newDecidingStop())
	h.speak()

	h.speak()

	want := []turns.UserTurnResumedParams{{EnableInterruptions: true, InferenceTriggered: false}}
	if !equalLists(h.resumes, want) {
		t.Fatalf("resumes = %+v, want %+v", h.resumes, want)
	}
}

func TestAResumeAfterTheInferenceWasTriggeredSaysSo(t *testing.T) {
	first, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, first, onInference)
	h.speak()
	h.do(first.decide)

	h.speak()

	if len(h.resumes) != 1 || !h.resumes[0].InferenceTriggered {
		t.Fatalf("resumes = %+v, want one after the inference", h.resumes)
	}
}

func TestOneSpeechReportedTwiceResumesTheTurnOnce(t *testing.T) {
	h := newStopHarness(t, turns.StrategyModeDefault, newDecidingStop(), newDecidingStop())
	h.speak()

	h.send(vadStarted(), frames.NewUserStartedSpeakingFrame())

	if len(h.resumes) != 1 {
		t.Fatalf("resumes = %d, want 1", len(h.resumes))
	}
}

func TestTheDeprecatedFinalizedTriggerStillDecides(t *testing.T) {
	strategy := newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, strategy)
	h.speak()

	h.do(strategy.TriggerFinalized) //nolint:staticcheck // the deprecated name is what is tested

	if !sameStrategies(h.stoppedBy(), strategy) {
		t.Fatalf("stopped by %v, want the strategy", h.stoppedBy())
	}
}

// A turn end announced elsewhere, or confirming a speculation, applies to the
// whole turn.
func TestTheTurnEndCombinesTheDecisions(t *testing.T) {
	announced, confirming := newDecidingStop(), newDecidingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, announced, confirming)
	h.speak()

	h.do(confirming.speculate)
	off := false
	h.do(func() { announced.ReportStoppedOverriding(turns.StoppedOverrides{EnableUserSpeakingFrames: &off}) })
	h.do(confirming.confirm)

	if len(h.stops) != 1 {
		t.Fatalf("stops = %d, want 1", len(h.stops))
	}
	if p := h.stops[0].params; p.EnableUserSpeakingFrames || !p.ConfirmsSpeculation {
		t.Fatalf("params = %+v, want no speaking frames and a confirmation", p)
	}
}

// A confirmation that crosses its speculation's withdrawal confirms nothing.
func TestConfirmingAWithdrawnSpeculationRunsTheTurnsAnswer(t *testing.T) {
	confirming, onInference := newDecidingStop(), newWaitingStop()
	h := newStopHarness(t, turns.StrategyModeDefault, confirming, onInference)
	h.speak()
	h.do(confirming.speculate)
	h.send(frames.NewUserTurnResponseCancelFrame())
	h.do(confirming.confirm)

	if got := h.speculative(); !equalLists(got, []bool{true, false}) {
		t.Fatalf("speculative inferences = %v, want a speculation then the turn's own", got)
	}
	h.do(onInference.decide)
	if len(h.stops) != 1 || h.stops[0].params.ConfirmsSpeculation {
		t.Fatalf("stops = %+v, want one confirming nothing", h.stops)
	}
}

// With StrategyModeAll the turn starts once every start strategy has decided.

func newStartHarness(
	t *testing.T, startMode turns.StrategyMode, start ...turns.StartStrategy,
) (*harness, *decidingStop) {
	t.Helper()
	stop := newDecidingStop()
	h := newHarness(t, turns.UserTurnStrategies{Start: start, Stop: []turns.StopStrategy{stop}},
		turns.UserTurnStrategiesParams{StartMode: startMode}, 0)
	return h, stop
}

func (h *harness) startedBy() []turns.StartStrategy {
	out := make([]turns.StartStrategy, 0, len(h.starts))
	for _, e := range h.starts {
		out = append(out, e.strategy)
	}
	return out
}

func TestTheTurnStartsOnceEveryStrategyDecides(t *testing.T) {
	first, last := newDecidingStart(), newDecidingStart()
	h, _ := newStartHarness(t, turns.StrategyModeAll, first, last)

	h.do(first.decide)
	if len(h.starts) != 0 {
		t.Fatal("one decision of two started the turn")
	}
	h.do(last.decide)
	if !equalLists(h.startedBy(), []turns.StartStrategy{last}) {
		t.Fatalf("started by %v, want the last", h.startedBy())
	}
}

func TestTheFirstDecisionStartsTheTurnInFirstMode(t *testing.T) {
	first, last := newDecidingStart(), newDecidingStart()
	h, _ := newStartHarness(t, turns.StrategyModeFirst, first, last)

	h.do(first.decide)

	if !equalLists(h.startedBy(), []turns.StartStrategy{first}) {
		t.Fatalf("started by %v, want the first", h.startedBy())
	}
}

func TestEveryStartStrategySeesEveryFrame(t *testing.T) {
	blocking, other := newDecidingStart(), newDecidingStart()
	blocking.result = turns.Stop
	h, _ := newStartHarness(t, turns.StrategyModeAll, blocking, other)

	h.send(vadStarted())

	if len(other.frames) != 1 {
		t.Fatalf("frames seen = %d, want 1", len(other.frames))
	}
}

func TestTheTurnStartCombinesTheDecisions(t *testing.T) {
	quiet, other := newDecidingStart(), newDecidingStart()
	h, _ := newStartHarness(t, turns.StrategyModeAll, quiet, other)

	off := false
	h.do(func() { quiet.ReportStartedOverriding(turns.StartedOverrides{EnableInterruptions: &off}) })
	h.do(other.decide)

	if len(h.starts) != 1 || h.starts[0].params.EnableInterruptions {
		t.Fatalf("starts = %+v, want one that does not interrupt", h.starts)
	}
}

func TestAFirstListStillHonorsAStopWithAWarning(t *testing.T) {
	logs := captureWarnings(t)
	blocking, other := newDecidingStart(), newDecidingStart()
	blocking.result = turns.Stop
	h, _ := newStartHarness(t, turns.StrategyModeFirst, blocking, other)

	h.send(vadStarted())

	if len(other.frames) != 0 {
		t.Fatal("a Stop in a first list let the frame through")
	}
	if n := strings.Count(logs.String(), "ProcessFrameResult is deprecated"); n != 1 {
		t.Fatalf("deprecation warnings = %d, want 1:\n%s", n, logs.String())
	}
}

func TestAWakePhraseAloneInAFirstListDoesNotWarn(t *testing.T) {
	logs := captureWarnings(t)
	h, _ := newStartHarness(t, turns.StrategyModeFirst,
		turns.NewWakePhraseStart(turns.WakePhraseStartConfig{Phrases: []string{"hey jargo"}}))

	h.send(vadStarted())

	if strings.Contains(logs.String(), "deprecated") {
		t.Fatalf("warned:\n%s", logs.String())
	}
}

func TestAWakePhraseInAnAllListDoesNotWarn(t *testing.T) {
	logs := captureWarnings(t)
	h, _ := newStartHarness(t, turns.StrategyModeAll, turns.NewVADStart(),
		turns.NewWakePhraseStart(turns.WakePhraseStartConfig{Phrases: []string{"hey jargo"}}))

	h.send(vadStarted())

	if strings.Contains(logs.String(), "deprecated") {
		t.Fatalf("warned:\n%s", logs.String())
	}
}

func TestEachTurnNeedsItsOwnStartDecisions(t *testing.T) {
	first, last := newDecidingStart(), newDecidingStart()
	h, stop := newStartHarness(t, turns.StrategyModeAll, first, last)
	h.do(first.decide)
	h.do(last.decide)
	h.do(stop.decide)

	h.do(last.decide)

	if len(h.starts) != 1 {
		t.Fatalf("starts = %d, want 1", len(h.starts))
	}
}
