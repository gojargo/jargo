package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// emit sends frames through the gate, collecting everything it lets out.
func emit(g *speculationGate, fs ...frames.Frame) []frames.Frame {
	var emitted []frames.Frame
	for _, f := range fs {
		for _, out := range g.process(f, processor.Downstream) {
			emitted = append(emitted, out.frame)
		}
	}
	return emitted
}

// reply builds the frames of one model response. Leaving end unset models a
// response whose generation was canceled mid-flight, so no end frame is coming.
func reply(end bool, texts ...string) []frames.Frame {
	fs := []frames.Frame{frames.NewLLMFullResponseStartFrame()}
	for _, t := range texts {
		fs = append(fs, frames.NewLLMTextFrame(t))
	}
	if end {
		fs = append(fs, frames.NewLLMFullResponseEndFrame())
	}
	return fs
}

// speculate runs an inference through the gate the way the service does. The
// gate is told whether the inference is speculative before its frames arrive,
// which is what decides whether the reply is held.
func speculate(g *speculationGate, speculation, end bool, texts ...string) []frames.Frame {
	g.beginSpeculation(speculation)
	return emit(g, reply(end, texts...)...)
}

// toolResult is the frame an asynchronous tool's answer arrives on, which is
// uninterruptible and so must always be delivered.
func toolResult(value string) *frames.FunctionCallResultFrame {
	return frames.NewFunctionCallResultFrame("call-1", "book_flight", json.RawMessage(`{}`), value)
}

// names is the sequence of frame names the gate emitted, which is what most of
// these tests are about.
func names(fs []frames.Frame) string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		// Name carries the frame's id, which changes from run to run.
		name, _, _ := strings.Cut(f.Name(), "#")
		out = append(out, strings.TrimSuffix(name, "Frame"))
	}
	return strings.Join(out, "|")
}

// texts is the text of every model text frame emitted, in order.
func texts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// TestNonSpeculativeReplyPassesThrough covers the ordinary case: an inference
// answering a turn that has ended is not held at all.
func TestNonSpeculativeReplyPassesThrough(t *testing.T) {
	g := newSpeculationGate("Test")

	got := speculate(g, false, true, "Hello.")
	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Errorf("emitted %s, want %s", names(got), want)
	}
}

// TestSpeculativeReplyIsHeldUntilTheTurnEnds covers the whole point of the gate:
// the reply answers a turn that may not have ended, so nothing of it escapes
// until the turn is confirmed, and then all of it does, in order.
func TestSpeculativeReplyIsHeldUntilTheTurnEnds(t *testing.T) {
	g := newSpeculationGate("Test")

	if got := speculate(g, true, true, "Booking ", "your flight."); len(got) != 0 {
		t.Fatalf("emitted %s while the turn was open, want nothing", names(got))
	}
	if g.state != speculationHolding {
		t.Fatalf("state = %v, want holding", g.state)
	}

	released := emit(g, frames.NewUserStoppedSpeakingFrame())
	want := "UserStoppedSpeaking|LLMFullResponseStart|LLMText|LLMText|LLMFullResponseEnd"
	if names(released) != want {
		t.Fatalf("released %s, want %s", names(released), want)
	}
	if got := texts(released); len(got) != 2 || got[0] != "Booking " || got[1] != "your flight." {
		t.Errorf("texts = %v, want the reply in the order it was generated", got)
	}
	if g.state != speculationOpen {
		t.Errorf("state = %v, want open", g.state)
	}
}

// TestWithdrawnSpeculationIsDiscarded covers the withdrawal, including the tail
// of the reply still queued behind the cancellation that overtook it.
func TestWithdrawnSpeculationIsDiscarded(t *testing.T) {
	g := newSpeculationGate("Test")

	if got := speculate(g, true, false, "Canceling ", "your booking."); len(got) != 0 {
		t.Fatalf("emitted %s, want nothing", names(got))
	}
	if got := emit(g, frames.NewEagerEndOfTurnCancelFrame()); names(got) != "EagerEndOfTurnCancel" {
		t.Fatalf("emitted %s, want only the cancellation", names(got))
	}
	if got := emit(g, frames.NewLLMTextFrame(" Done.")); len(got) != 0 {
		t.Errorf("emitted %s, want the straggler dropped", names(got))
	}

	got := speculate(g, false, true, "Rescheduling instead.")
	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Errorf("emitted %s, want the next reply whole", names(got))
	}
}

// TestWithdrawalArrivingBeforeTheReplyItCancels covers the ordering a system
// frame makes possible: the cancellation can overtake the frames it withdraws.
func TestWithdrawalArrivingBeforeTheReplyItCancels(t *testing.T) {
	g := newSpeculationGate("Test")

	emit(g, frames.NewEagerEndOfTurnCancelFrame())
	if got := speculate(g, true, true, "Canceling."); len(got) != 0 {
		t.Fatalf("emitted %s, want nothing: the reply is void", names(got))
	}

	got := speculate(g, false, true, "Rescheduling instead.")
	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Errorf("emitted %s, want the next reply whole", names(got))
	}
}

// TestInterruptionDiscardsTheSpeculation covers a barge-in voiding a held reply.
func TestInterruptionDiscardsTheSpeculation(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	if got := emit(g, frames.NewInterruptionFrame()); names(got) != "Interruption" {
		t.Errorf("emitted %s, want only the interruption", names(got))
	}
}

// TestUpstreamFramesAreNeverHeld covers the gate minding one direction only.
func TestUpstreamFramesAreNeverHeld(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	up := frames.NewLLMTextFrame("upstream")
	got := g.process(up, processor.Upstream)
	if len(got) != 1 || got[0].frame != up || got[0].dir != processor.Upstream {
		t.Errorf("emitted %v, want the upstream frame straight through", got)
	}
}

// TestShutdownDeliversWhatHasToOutliveTheSpeculation covers the end of the
// session: the EndFrame is uninterruptible and awaited, so holding it would hang
// shutdown, and whatever was held that must be delivered goes ahead of it.
func TestShutdownDeliversWhatHasToOutliveTheSpeculation(t *testing.T) {
	g := newSpeculationGate("Test")

	g.beginSpeculation(true)
	emit(g, reply(false, "Booking.")...)
	emit(g, toolResult("booked"))

	got := emit(g, frames.NewEndFrame())
	if want := "FunctionCallResult|End"; names(got) != want {
		t.Errorf("emitted %s, want %s", names(got), want)
	}
}

// Whether the gate reports an inference as still answering an unconfirmed turn.
// The service asks the gate rather than tracking it alongside, so it has to hold
// up however the turn and the inference are ordered.

// TestAnOrdinaryInferenceIsNeverPending covers the baseline.
func TestAnOrdinaryInferenceIsNeverPending(t *testing.T) {
	g := newSpeculationGate("Test")
	g.beginSpeculation(false)

	if g.isSpeculating() {
		t.Error("an ordinary inference reads as speculative")
	}
}

// TestItNamesTheInferenceUntilTheTurnIsConfirmed covers the flag spanning the
// whole inference rather than only the stretch with frames in flight.
func TestItNamesTheInferenceUntilTheTurnIsConfirmed(t *testing.T) {
	g := newSpeculationGate("Test")
	if g.isSpeculating() {
		t.Fatal("a fresh gate reads as speculating")
	}

	speculate(g, true, false, "Booking.")
	if !g.isSpeculating() {
		t.Error("the inference does not read as speculative")
	}

	emit(g, frames.NewUserStoppedSpeakingFrame())
	if g.isSpeculating() {
		t.Error("the confirmed inference still reads as speculative")
	}
}

// TestATurnConfirmedMidInferenceClearsItBeforeTheReplyEnds covers the window the
// feature exists to exploit: the turn is confirmed while the inference is still
// generating, so what it does next is committed.
func TestATurnConfirmedMidInferenceClearsItBeforeTheReplyEnds(t *testing.T) {
	g := newSpeculationGate("Test")

	g.beginSpeculation(true)
	emit(g, frames.NewLLMFullResponseStartFrame())
	emit(g, frames.NewUserStoppedSpeakingFrame())

	if g.isSpeculating() {
		t.Error("the inference still reads as speculative after the turn was confirmed")
	}
}

// TestATurnConfirmedBeforeTheInferenceIsNeverPending covers the confirmation
// being a system frame, which can pass the context frame starting the inference
// it confirms.
func TestATurnConfirmedBeforeTheInferenceIsNeverPending(t *testing.T) {
	g := newSpeculationGate("Test")

	emit(g, frames.NewUserStoppedSpeakingFrame())
	g.beginSpeculation(true)

	if g.isSpeculating() {
		t.Error("the inference reads as speculative, but its turn is already over")
	}
	got := emit(g, reply(true, "Booking.")...)
	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Errorf("emitted %s, want the reply whole: its turn is already over", names(got))
	}
}

// TestAWithdrawalClearsIt covers a withdrawal settling the inference.
func TestAWithdrawalClearsIt(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	emit(g, frames.NewEagerEndOfTurnCancelFrame())

	if g.isSpeculating() {
		t.Error("the withdrawn inference still reads as speculative")
	}
}

// TestAnInterruptionClearsIt covers a barge-in settling it.
func TestAnInterruptionClearsIt(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	emit(g, frames.NewInterruptionFrame())

	if g.isSpeculating() {
		t.Error("the interrupted inference still reads as speculative")
	}
}

// TestASupersedingInferenceKeepsItsOwn covers dropping a held reply ending that
// hold while the inference that replaced it stays pending.
func TestASupersedingInferenceKeepsItsOwn(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	speculate(g, true, false, "Rescheduling.")

	if !g.isSpeculating() {
		t.Error("the superseding inference lost its own speculation")
	}
}

// TestAWithdrawnReplyIsNotReleasedByTheTurnEnding covers the order the mismatch
// path relies on: it withdraws before it ends the turn, so by the time the turn
// ends there is nothing left to release.
func TestAWithdrawnReplyIsNotReleasedByTheTurnEnding(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Canceling.")
	if got := emit(g, frames.NewEagerEndOfTurnCancelFrame()); names(got) != "EagerEndOfTurnCancel" {
		t.Fatalf("emitted %s, want only the cancellation", names(got))
	}
	if got := emit(g, frames.NewUserStoppedSpeakingFrame()); names(got) != "UserStoppedSpeaking" {
		t.Errorf("emitted %s, want only the turn end: the reply was already withdrawn", names(got))
	}
}

// TestANewReplySupersedesAHeldOne covers a withdrawal that arrives before the
// reply it voids needing no memory: the reply is held on arrival, and whatever
// answers the turn instead supersedes it.
func TestANewReplySupersedesAHeldOne(t *testing.T) {
	g := newSpeculationGate("Test")

	emit(g, frames.NewEagerEndOfTurnCancelFrame())
	if got := speculate(g, true, false, "Canceling."); len(got) != 0 {
		t.Fatalf("emitted %s, want nothing", names(got))
	}
	got := speculate(g, false, true, "Rescheduling instead.")
	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Fatalf("emitted %s, want the superseding reply whole", names(got))
	}
	if g.state != speculationOpen {
		t.Errorf("state = %v, want open", g.state)
	}
}

// TestAHeldReplyDoesNotSwallowTheOneThatSupersedesIt covers the held reply never
// ending, its generation having been canceled mid-flight, so only a new reply
// resolves it. Its frames are dropped, and the reply that supersedes it passes
// through whole: nothing of the held one can still be queued behind a frame that
// arrived after it.
func TestAHeldReplyDoesNotSwallowTheOneThatSupersedesIt(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	got := speculate(g, false, true, "Something else entirely.")

	if want := "LLMFullResponseStart|LLMText|LLMFullResponseEnd"; names(got) != want {
		t.Fatalf("emitted %s, want the superseding reply whole", names(got))
	}
	if tx := texts(got); len(tx) != 1 || tx[0] != "Something else entirely." {
		t.Errorf("texts = %v, want only the superseding reply's", tx)
	}
}

// TestAToolResultSurvivesADiscardedSpeculation covers an asynchronous tool
// started in an earlier turn returning while a speculation is held. Its result
// belongs to that earlier work and is guaranteed delivery, so discarding the
// speculation around it keeps it.
func TestAToolResultSurvivesADiscardedSpeculation(t *testing.T) {
	g := newSpeculationGate("Test")

	g.beginSpeculation(true)
	emit(g, reply(false, "Booking.")...)
	emit(g, toolResult("booked"))

	got := emit(g, frames.NewEagerEndOfTurnCancelFrame())
	if want := "EagerEndOfTurnCancel|FunctionCallResult"; names(got) != want {
		t.Fatalf("emitted %s, want %s", names(got), want)
	}
	for _, f := range got {
		if rf, ok := f.(*frames.FunctionCallResultFrame); ok && rf.Result != "booked" {
			t.Errorf("result = %q, want the tool's own", rf.Result)
		}
	}
}

// TestAToolResultIsHeldInOrderWithTheReply covers uninterruptible frames being
// ordered like any other, so one arriving mid-reply is released where it landed.
func TestAToolResultIsHeldInOrderWithTheReply(t *testing.T) {
	g := newSpeculationGate("Test")

	g.beginSpeculation(true)
	emit(g, reply(false, "Booking.")...)
	emit(g, toolResult("booked"))

	got := emit(g, frames.NewUserStoppedSpeakingFrame())
	want := "UserStoppedSpeaking|LLMFullResponseStart|LLMText|FunctionCallResult"
	if names(got) != want {
		t.Errorf("emitted %s, want %s", names(got), want)
	}
}

// TestAToolResultIsNotDroppedWithAReplyBeingDropped covers nothing being held
// back while the tail of a withdrawn reply is dropped, so an uninterruptible
// frame passes on in order rather than going with it.
func TestAToolResultIsNotDroppedWithAReplyBeingDropped(t *testing.T) {
	g := newSpeculationGate("Test")

	speculate(g, true, false, "Booking.")
	if got := emit(g, frames.NewEagerEndOfTurnCancelFrame()); names(got) != "EagerEndOfTurnCancel" {
		t.Fatalf("emitted %s, want only the cancellation", names(got))
	}
	got := emit(g, frames.NewLLMTextFrame(" Done."), toolResult("booked"))
	if names(got) != "FunctionCallResult" {
		t.Errorf("emitted %s, want the tool result kept and the straggler dropped", names(got))
	}
}
