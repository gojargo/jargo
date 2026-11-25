package turns_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// The timeouts the waiting tests run with, and how long they let a timer run to
// see whether it fires.
const (
	waitingIdleTimeout = 100 * time.Millisecond
	waitingTimeout     = 100 * time.Millisecond
	letTimersRun       = waitingTimeout + 80*time.Millisecond
)

// waitingHarness drives a UserIdleController directly, the way its owner does,
// and records which of its two events fired, in order.
type waitingHarness struct {
	ctrl *turns.UserIdleController

	mu     sync.Mutex
	events []string
}

// newWaitingHarness builds a controller with the given timeouts; 0 leaves the
// matching detection off.
func newWaitingHarness(t *testing.T, idle, waiting time.Duration) *waitingHarness {
	t.Helper()
	h := &waitingHarness{}
	record := func(event string) turns.IdleCallback {
		return func(context.Context, *turns.UserIdleController) error {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.events = append(h.events, event)
			return nil
		}
	}
	h.ctrl = turns.NewUserIdleController(turns.IdleConfig{
		Timeout: idle, Callback: record("idle"),
		WaitingTimeout: waiting, WaitingCallback: record("waiting"),
	})
	h.ctrl.Setup(t.Context(), nil)
	t.Cleanup(h.ctrl.Cleanup)
	return h
}

// newWaiting builds the controller most waiting tests use, with both timeouts.
func newWaiting(t *testing.T) *waitingHarness {
	t.Helper()
	return newWaitingHarness(t, waitingIdleTimeout, waitingTimeout)
}

func (h *waitingHarness) send(fs ...frames.Frame) {
	for _, f := range fs {
		h.ctrl.Process(f)
	}
}

func (h *waitingHarness) got() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.events)
}

func (h *waitingHarness) want(t *testing.T, want ...string) {
	t.Helper()
	if got := h.got(); !slices.Equal(got, want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
}

func workStarted(id string) frames.Frame  { return frames.NewBackgroundWorkStartedFrame(id) }
func workFinished(id string) frames.Frame { return frames.NewBackgroundWorkFinishedFrame(id) }

func callsStarted(ids ...string) frames.Frame {
	calls := make([]frames.ToolCall, 0, len(ids))
	for _, id := range ids {
		calls = append(calls, frames.ToolCall{ID: id, Name: "test"})
	}
	return frames.NewFunctionCallsStartedFrame(calls)
}

func callResult(id string) frames.Frame {
	return frames.NewFunctionCallResultFrame(id, "test", nil, "ok")
}

// A built-in tool whose calls are not announced still reports a result, which
// settles nothing.
func TestIdleAResultForACallNotSeenStartingSettlesNothing(t *testing.T) {
	h := newIdleHarness(t, idleTimeout)

	h.send(callsStarted("123"))
	h.send(frames.NewFunctionCallResultFrame("456", "unannounced", nil, "ok"))
	h.send(frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame())

	if h.waitFired(3 * idleTimeout) {
		t.Error("idle fired while the announced call was still running")
	}
	if !h.ctrl.FunctionCallsInProgress() {
		t.Error("a result for another call settled the announced one")
	}
}

// The waiting timer: the same silence as idle, with background work in progress.

func TestWaitingBotStoppingWithWorkInFlightStartsTheWaitingTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingBotStoppingWithNothingInFlightStartsTheIdleTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "idle")
}

// The call's work frames bracket it; its pending result alone suppresses idle.
func TestWaitingAnOpenAsyncCallCountsAsWorkInFlight(t *testing.T) {
	h := newWaiting(t)
	h.send(callsStarted("123"), workStarted("call_1"),
		frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
	if h.ctrl.WaitingForUser() {
		t.Error("waiting for the user while a function call is pending")
	}
}

func TestWaitingAPendingSynchronousCallStartsNoTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(callsStarted("123"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingWorkStartingInAQuietConversationStartsTheWaitingTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(waitingIdleTimeout * 3 / 10)
	h.send(workStarted("w1"))
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingWorkStartingWhileTheBotSpeaksWaitsForItToStop(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBotStartedSpeakingFrame(), workStarted("w1"))
	time.Sleep(letTimersRun)
	h.want(t)

	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingTheLastWorkFinishingHandsOverToTheIdleTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), workStarted("w2"),
		frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(waitingTimeout * 3 / 10)
	h.send(workFinished("w2"))
	time.Sleep(waitingTimeout * 3 / 10)
	h.want(t)

	h.send(workFinished("w1"))
	time.Sleep(letTimersRun)
	h.want(t, "idle")
	if !h.ctrl.WaitingForUser() {
		t.Error("not waiting for the user once the work was over")
	}
}

func TestWaitingAFinishForUnknownWorkIsIgnored(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), workFinished("other"), workFinished("w1"), workFinished("w1"),
		frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "idle")
}

func TestWaitingTheWaitingEventFiresOncePerStart(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(2 * letTimersRun)
	h.want(t, "waiting")

	h.send(frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "waiting", "waiting")
}

func TestWaitingAWaitingTimeoutUpdateAppliesAtOnce(t *testing.T) {
	h := newWaitingHarness(t, waitingIdleTimeout, 0)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame(),
		frames.NewUserWaitingTimeoutUpdateFrame(waitingTimeout))
	time.Sleep(letTimersRun)
	h.want(t, "waiting")

	h.send(frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame(),
		frames.NewUserWaitingTimeoutUpdateFrame(0))
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingQuietSecsCountsFromWhenTheConversationWentQuiet(t *testing.T) {
	h := newWaiting(t)
	if got := h.ctrl.QuietSecs(); got != 0 {
		t.Fatalf("quiet for %v before anything was said", got)
	}
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(100 * time.Millisecond)
	if got := h.ctrl.QuietSecs(); got < 0.1 {
		t.Errorf("quiet for %v, want at least 0.1", got)
	}

	h.send(frames.NewUserStartedSpeakingFrame())
	if got := h.ctrl.QuietSecs(); got != 0 {
		t.Errorf("quiet for %v while the user speaks", got)
	}
	h.send(frames.NewUserStoppedSpeakingFrame())
	stoppedAt := time.Now()
	time.Sleep(50 * time.Millisecond)
	got := h.ctrl.QuietSecs()
	if got < 0.05 {
		t.Errorf("quiet for %v, want at least 0.05", got)
	}
	// Counted from the user stopping, not from the bot.
	if elapsed := time.Since(stoppedAt).Seconds(); got > elapsed {
		t.Errorf("quiet for %v, more than the %v since the user stopped", got, elapsed)
	}
}

// The bot speaks, then calls a function, and says nothing on its result.
func TestWaitingACallSettlingWithWorkInFlightStartsTheWaitingTimerOver(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame(),
		callsStarted("123"), callResult("123"))
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingACallSettlingWhileTheBotSpeaksStartsNoTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), callsStarted("123"), frames.NewBotStartedSpeakingFrame(), callResult("123"))
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingSpeechCancelsTheWaitingTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(waitingTimeout * 3 / 10)
	h.send(frames.NewUserStartedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingWaitForUserWithWorkInFlightStartsTheWaitingTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame(),
		frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame())
	h.ctrl.WaitForUser()
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

// An async call stays pending while it runs as background work.
func TestWaitingWaitForUserDuringAnAsyncFunctionCallStartsTheWaitingTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(callsStarted("123"), workStarted("w1"),
		frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame(),
		frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame())
	h.ctrl.WaitForUser()
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingAnIdleTimeoutUpdateDoesNotArmDuringWork(t *testing.T) {
	h := newWaiting(t)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame(),
		frames.NewUserIdleTimeoutUpdateFrame(waitingIdleTimeout))
	time.Sleep(letTimersRun)
	h.want(t, "waiting")
}

func TestWaitingWorkStartingCancelsAnIdleTimerEvenWithNoWaitingTimeout(t *testing.T) {
	h := newWaitingHarness(t, waitingIdleTimeout, 0)
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(waitingIdleTimeout * 3 / 10)
	h.send(workStarted("w1"))
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingWorkFinishingCancelsAWaitingTimerEvenWithNoIdleTimeout(t *testing.T) {
	h := newWaitingHarness(t, 0, waitingTimeout)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(waitingTimeout * 3 / 10)
	h.send(workFinished("w1"))
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingDetectionIsOffByDefault(t *testing.T) {
	h := newWaitingHarness(t, waitingIdleTimeout, 0)
	h.send(workStarted("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t)
}

// The hosts expose the waiting timeout and event beside the idle pair, and
// broadcast both silences.

// silenceFrames runs the frames through the processors, waits out the timers,
// and returns the silence frames that reached each end of the pipeline.
func silenceFrames(t *testing.T, ps []processor.Processor, fs ...frames.Frame) (down, up []frames.Frame) {
	t.Helper()
	keep := func(f frames.Frame) bool {
		switch f.(type) {
		case *frames.UserTurnIdleFrame, *frames.UserWaitingFrame:
			return true
		}
		return false
	}
	task := pipeline.NewWorker(pipeline.New(ps...), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	var mu sync.Mutex
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			down = append(down, f)
			mu.Unlock()
		}
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			up = append(up, f)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range fs {
		task.QueueFrame(f)
	}
	time.Sleep(letTimersRun + 100*time.Millisecond)
	finish(t, task, done)
	mu.Lock()
	defer mu.Unlock()
	return down, up
}

// quietSecs reads how long a silence frame says it has been quiet.
func quietSecs(t *testing.T, f frames.Frame) float64 {
	t.Helper()
	switch fr := f.(type) {
	case *frames.UserTurnIdleFrame:
		return fr.QuietSecs
	case *frames.UserWaitingFrame:
		return fr.QuietSecs
	}
	t.Fatalf("%v is no silence frame", f)
	return 0
}

// wantOneEachWay checks a single silence frame of the wanted type reached each
// end, and that both say the same about how long it has been quiet, at least
// timeout.
func wantOneEachWay[T frames.Frame](t *testing.T, down, up []frames.Frame, timeout time.Duration) {
	t.Helper()
	if len(down) != 1 || len(up) != 1 {
		t.Fatalf("silence frames: %d down, %d up, want one each way", len(down), len(up))
	}
	if _, ok := down[0].(T); !ok {
		t.Fatalf("down = %v", down[0])
	}
	if _, ok := up[0].(T); !ok {
		t.Fatalf("up = %v", up[0])
	}
	d, u := quietSecs(t, down[0]), quietSecs(t, up[0])
	if d < timeout.Seconds() {
		t.Errorf("quiet for %v, want at least %v", d, timeout.Seconds())
	}
	if d != u {
		t.Errorf("quiet for %v down and %v up, want the same", d, u)
	}
}

func TestTheUserAggregatorEmitsOnUserWaiting(t *testing.T) {
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(turns.Config{
		WaitingTimeout: waitingTimeout,
	}))
	var mu sync.Mutex
	var fired []string
	event.OnSignal(pair.User().Events(), aggregators.EventUserWaiting, func(context.Context) {
		mu.Lock()
		fired = append(fired, "waiting")
		mu.Unlock()
	})

	down, up := silenceFrames(t, []processor.Processor{pair.User(), pair.Assistant()},
		workStarted("w1"), frames.NewBotStoppedSpeakingFrame())

	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(fired, []string{"waiting"}) {
		t.Errorf("events = %v, want one waiting", fired)
	}
	wantOneEachWay[*frames.UserWaitingFrame](t, down, up, waitingTimeout)
}

func TestTheUserAggregatorBroadcastsTheIdleFrame(t *testing.T) {
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(turns.Config{
		IdleTimeout: waitingIdleTimeout,
	}))

	down, up := silenceFrames(t, []processor.Processor{pair.User(), pair.Assistant()},
		frames.NewBotStoppedSpeakingFrame())

	wantOneEachWay[*frames.UserTurnIdleFrame](t, down, up, waitingIdleTimeout)
}

func TestTheUserTurnProcessorEmitsOnUserWaiting(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{WaitingTimeout: waitingTimeout})
	var mu sync.Mutex
	var fired []string
	event.OnSignal(p.Events(), turns.EventUserWaiting, func(context.Context) {
		mu.Lock()
		fired = append(fired, "waiting")
		mu.Unlock()
	})

	down, up := silenceFrames(t, []processor.Processor{p}, workStarted("w1"), frames.NewBotStoppedSpeakingFrame())

	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(fired, []string{"waiting"}) {
		t.Errorf("events = %v, want one waiting", fired)
	}
	wantOneEachWay[*frames.UserWaitingFrame](t, down, up, waitingTimeout)
}

func TestTheUserTurnProcessorBroadcastsTheIdleFrame(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{IdleTimeout: waitingIdleTimeout})

	down, up := silenceFrames(t, []processor.Processor{p}, frames.NewBotStoppedSpeakingFrame())

	wantOneEachWay[*frames.UserTurnIdleFrame](t, down, up, waitingIdleTimeout)
}

// silentReply is the report of a reply the LLM chose to leave unspoken.
func silentReply() frames.Frame {
	return frames.NewLLMMarkerResponseFrame("∅", "", frames.LLMMarkerKindSilent, nil)
}

// A reply the LLM left unspoken ends its turn like speech would.
func TestWaitingASilentReplyStartsTheWaitingTimerOver(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBackgroundWorkStartedFrame("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "waiting")

	h.send(silentReply())
	time.Sleep(letTimersRun)
	h.want(t, "waiting", "waiting")
}

func TestWaitingASilentReplyStartsTheIdleTimerOver(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.want(t, "idle")

	h.send(silentReply())
	time.Sleep(letTimersRun)
	h.want(t, "idle", "idle")
	if !h.ctrl.WaitingForUser() {
		t.Error("WaitingForUser() = false after a silent reply, want true")
	}
}

func TestWaitingASilentReplyStartsNoTimerWhileSomeoneSpeaks(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBackgroundWorkStartedFrame("w1"), frames.NewBotStartedSpeakingFrame(), silentReply())
	time.Sleep(letTimersRun)
	h.want(t)

	h.send(frames.NewBotStoppedSpeakingFrame(), frames.NewUserStartedSpeakingFrame(), silentReply())
	time.Sleep(letTimersRun)
	h.want(t)
}

func TestWaitingOtherMarkerReportsStartNoTimer(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.send(frames.NewLLMMarkerResponseFrame(frames.UserTurnCompleteMarker+" Hi", frames.UserTurnCompleteMarker,
		"complete", nil))
	time.Sleep(letTimersRun)
	h.want(t, "idle")
}

func TestWaitingQuietGoesOnThroughASilentReply(t *testing.T) {
	h := newWaiting(t)
	h.send(frames.NewBackgroundWorkStartedFrame("w1"), frames.NewBotStoppedSpeakingFrame())
	time.Sleep(letTimersRun)
	h.send(silentReply())
	if got := h.ctrl.QuietSecs(); got < waitingTimeout.Seconds() {
		t.Errorf("QuietSecs() = %v after a silent reply, want at least %v", got, waitingTimeout.Seconds())
	}
}
