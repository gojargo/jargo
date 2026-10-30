package turns

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// LLMTurnCompletionStop decides on the marker of the inference that answers the
// turn.

type completionHarness struct {
	t   *testing.T
	s   *LLMTurnCompletionStop
	spy *spy
}

func newCompletionHarness(t *testing.T) *completionHarness {
	t.Helper()
	s := NewLLMTurnCompletionStop(llm.UserTurnCompletionConfig{IncompleteShortTimeout: 50 * time.Millisecond})
	h := &completionHarness{t: t, s: s, spy: attachStop(s)}
	t.Cleanup(func() {
		h.spy.mu.Lock()
		s.Cleanup()
		h.spy.mu.Unlock()
	})
	return h
}

func (h *completionHarness) infer(speculation bool) *frames.LLMContextFrame {
	f := frames.NewLLMContextFrame(frames.NewLLMContext(""))
	f.Speculation = speculation
	h.spy.sendStop(h.s, f)
	return f
}

func (h *completionHarness) report(kind string, inference *frames.LLMContextFrame) {
	h.spy.sendStop(h.s, frames.NewUserTurnCompletionFrame(kind, "", inference.ID()))
}

func (h *completionHarness) decisions() int { return h.spy.stops() }

func (h *completionHarness) withdrawals() int {
	h.spy.mu.Lock()
	defer h.spy.mu.Unlock()
	return h.spy.withdrawals
}

func (h *completionHarness) pushed() []frames.Frame {
	h.spy.mu.Lock()
	defer h.spy.mu.Unlock()
	return append([]frames.Frame(nil), h.spy.pushedFrame...)
}

func TestLLMTurnCompletionStopEachMarkerGivesItsDecision(t *testing.T) {
	for _, tt := range []struct {
		kind      string
		decisions int
	}{{"complete", 1}, {"short", 0}, {"long", 0}, {"", 1}} {
		h := newCompletionHarness(t)
		h.report(tt.kind, h.infer(false))
		if got := h.decisions(); got != tt.decisions {
			t.Errorf("kind %q: decisions = %d, want %d", tt.kind, got, tt.decisions)
		}
	}
}

func TestLLMTurnCompletionStopAReportForAnotherInferenceIsIgnored(t *testing.T) {
	h := newCompletionHarness(t)
	earlier := h.infer(false)
	h.infer(false)

	h.report("complete", earlier)

	if got := h.decisions(); got != 0 {
		t.Fatalf("decisions = %d, want 0", got)
	}
}

func TestLLMTurnCompletionStopSpeakingAgainWaitsForTheNextInference(t *testing.T) {
	h := newCompletionHarness(t)
	inference := h.infer(false)
	h.spy.sendStop(h.s, frames.NewVADUserStartedSpeakingFrame(0, time.Now()))

	h.report("complete", inference)

	if got := h.decisions(); got != 0 {
		t.Fatalf("decisions = %d, want 0", got)
	}
}

func TestLLMTurnCompletionStopACanceledAnswerGetsNoDecision(t *testing.T) {
	h := newCompletionHarness(t)
	inference := h.infer(false)
	h.spy.sendStop(h.s, frames.NewUserTurnResponseCancelFrame())

	h.report("complete", inference)

	if got := h.decisions(); got != 0 {
		t.Fatalf("decisions = %d, want 0", got)
	}
}

func TestLLMTurnCompletionStopACompleteSpeculativeAnswerGetsADecision(t *testing.T) {
	h := newCompletionHarness(t)

	h.report("complete", h.infer(true))

	if got := h.decisions(); got != 1 {
		t.Fatalf("decisions = %d, want 1", got)
	}
}

// The committed turn brings an inference of its own, so nothing is re-prompted.
func TestLLMTurnCompletionStopAnIncompleteSpeculativeAnswerIsWithdrawn(t *testing.T) {
	h := newCompletionHarness(t)

	h.report("short", h.infer(true))
	time.Sleep(150 * time.Millisecond)

	if h.withdrawals() != 1 || h.decisions() != 0 || len(h.pushed()) != 0 {
		t.Fatalf("withdrawals = %d, decisions = %d, pushed = %v, want 1, 0 and none",
			h.withdrawals(), h.decisions(), h.pushed())
	}
}

func TestLLMTurnCompletionStopAQuietUserIsRepromptedAfterAnIncompleteTurn(t *testing.T) {
	h := newCompletionHarness(t)

	h.report("short", h.infer(false))
	time.Sleep(150 * time.Millisecond)

	if got := h.decisions(); got != 1 {
		t.Fatalf("decisions = %d, want 1", got)
	}
	pushed := h.pushed()
	if len(pushed) != 1 {
		t.Fatalf("pushed = %v, want the re-prompt", pushed)
	}
	reprompt, ok := pushed[0].(*frames.LLMMessagesAppendFrame)
	if !ok || !reprompt.RunLLM || reprompt.Messages[0].Text != h.s.Config().ShortPrompt() {
		t.Fatalf("pushed %v, want the short re-prompt run at once", pushed[0])
	}
}

func TestLLMTurnCompletionStopSpeakingAgainCancelsTheReprompt(t *testing.T) {
	h := newCompletionHarness(t)
	h.report("short", h.infer(false))
	h.spy.sendStop(h.s, frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	time.Sleep(150 * time.Millisecond)

	if h.decisions() != 0 || len(h.pushed()) != 0 {
		t.Fatalf("decisions = %d, pushed = %v, want neither", h.decisions(), h.pushed())
	}
}

func TestLLMTurnCompletionStopATurnEndedByTheStopTimeoutIsStillReprompted(t *testing.T) {
	h := newCompletionHarness(t)
	h.report("short", h.infer(false))
	h.spy.mu.Lock()
	h.s.TurnStopped()
	h.spy.mu.Unlock()
	time.Sleep(150 * time.Millisecond)

	pushed := h.pushed()
	if len(pushed) != 1 {
		t.Fatalf("pushed = %v, want the re-prompt", pushed)
	}
	if _, ok := pushed[0].(*frames.LLMMessagesAppendFrame); !ok {
		t.Fatalf("pushed %T, want the re-prompt", pushed[0])
	}
}

func TestLLMTurnCompletionStopAnotherInferenceCancelsTheReprompt(t *testing.T) {
	h := newCompletionHarness(t)
	h.report("short", h.infer(false))
	h.infer(false)
	time.Sleep(150 * time.Millisecond)

	if got := h.pushed(); len(got) != 0 {
		t.Fatalf("pushed = %v, want none", got)
	}
}

func TestLLMTurnCompletionStopAReportFromAnotherInferenceCancelsTheReprompt(t *testing.T) {
	h := newCompletionHarness(t)
	h.report("short", h.infer(false))
	h.report("", frames.NewLLMContextFrame(frames.NewLLMContext("")))
	time.Sleep(150 * time.Millisecond)

	if got := h.pushed(); len(got) != 0 {
		t.Fatalf("pushed = %v, want none", got)
	}
}

func TestLLMTurnCompletionStopALaterCompleteMarkerCancelsTheReprompt(t *testing.T) {
	h := newCompletionHarness(t)
	h.report("short", h.infer(false))
	h.report("complete", h.infer(false))
	time.Sleep(150 * time.Millisecond)

	if h.decisions() != 1 || len(h.pushed()) != 0 {
		t.Fatalf("decisions = %d, pushed = %v, want 1 and none", h.decisions(), h.pushed())
	}
}
