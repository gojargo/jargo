package aggregators

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
)

// Every way an interrupted turn can close, one handler call at a time. The
// frame-driven tests leave it to scheduling which frame reaches the aggregator
// first after an interruption; these pin each path.
//
// Ported from upstream's TestLLMAssistantAggregatorDeferredClose.

type deferredClose struct {
	convo *frames.LLMContext
	a     *AssistantAggregator

	mu    sync.Mutex
	stops []AssistantTurnStopped
}

func newDeferredClose(t *testing.T) *deferredClose {
	t.Helper()
	d := &deferredClose{convo: frames.NewLLMContext("")}
	d.a = New(d.convo).Assistant()
	event.On(d.a.Events(), EventAssistantTurnStopped, func(_ context.Context, s AssistantTurnStopped) {
		d.mu.Lock()
		d.stops = append(d.stops, s)
		d.mu.Unlock()
	})
	return d
}

// settle lets the event handlers run.
func (d *deferredClose) settle(t *testing.T) {
	t.Helper()
	d.a.Events().Cleanup(t.Context())
}

func (d *deferredClose) stopped() []AssistantTurnStopped {
	d.mu.Lock()
	defer d.mu.Unlock()
	return slices.Clone(d.stops)
}

func (d *deferredClose) interrupted() []bool {
	var out []bool
	for _, s := range d.stopped() {
		out = append(out, s.Interrupted)
	}
	return out
}

func (d *deferredClose) contents() []string {
	var out []string
	for _, m := range d.convo.Messages() {
		out = append(out, m.Text)
	}
	return out
}

func (d *deferredClose) turnOpen() bool {
	d.a.mu.Lock()
	defer d.a.mu.Unlock()
	return d.a.turnStartedAt != ""
}

func (d *deferredClose) say(text string) {
	d.a.routeText(frames.NewTTSTextFrame(text, frames.TextTypeWord))
}

func (d *deferredClose) interrupt(t *testing.T) {
	t.Helper()
	if err := d.a.handleInterruptions(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func (d *deferredClose) closeInterrupted(t *testing.T) {
	t.Helper()
	if err := d.a.handleAssistantTurnInterrupted(t.Context()); err != nil {
		t.Fatal(err)
	}
}

// speakAndGetInterrupted opens a turn, speaks, gets interrupted, then receives
// the word the interruption overtook.
func (d *deferredClose) speakAndGetInterrupted(t *testing.T) {
	t.Helper()
	d.a.openTurn(t.Context())
	d.say("Hello")
	d.interrupt(t)
	d.say("there")
}

func TestClosingFrameClosesTheInterruptedTurn(t *testing.T) {
	d := newDeferredClose(t)
	d.speakAndGetInterrupted(t)

	d.closeInterrupted(t)
	d.settle(t)

	if got := d.contents(); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("conversation = %q, want the overtaken word in the turn", got)
	}
	if got := d.interrupted(); !slices.Equal(got, []bool{true}) {
		t.Errorf("turns interrupted = %v, want one interrupted", got)
	}
	if got := d.stopped(); len(got) != 1 || got[0].Content != "Hello there" {
		t.Errorf("turns reported = %+v, want one saying %q", got, "Hello there")
	}
	if d.turnOpen() {
		t.Error("the turn is still open")
	}
}

func TestClosingFrameAfterTheTurnEndedLeavesTheNextOneOpen(t *testing.T) {
	d := newDeferredClose(t)
	d.speakAndGetInterrupted(t)
	// The interruption overtook the turn's own end, which closes it first.
	if err := d.a.commit(t.Context(), false); err != nil {
		t.Fatal(err)
	}
	d.a.openTurn(t.Context())
	d.say("Next")

	d.closeInterrupted(t)
	d.settle(t)

	if got := d.contents(); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("conversation = %q, want the interrupted turn alone", got)
	}
	if got := len(d.stopped()); got != 1 {
		t.Errorf("turns reported = %d, want 1", got)
	}
	if !d.turnOpen() {
		t.Error("the stale closing frame closed the next turn")
	}
}

func TestEndFrameClosesTheInterruptedTurnAsInterrupted(t *testing.T) {
	d := newDeferredClose(t)
	d.speakAndGetInterrupted(t)

	if err := d.a.closeTurn(t.Context(), frames.NewEndFrame()); err != nil {
		t.Fatal(err)
	}
	d.settle(t)

	if got := d.contents(); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("conversation = %q, want the overtaken word in the turn", got)
	}
	if got := d.interrupted(); !slices.Equal(got, []bool{true}) {
		t.Errorf("turns interrupted = %v, want one interrupted", got)
	}

	// The closing frame, arriving later, finds nothing left to close.
	d.closeInterrupted(t)
	d.settle(t)
	if got := len(d.stopped()); got != 1 {
		t.Errorf("turns reported = %d, want 1", got)
	}
}

func TestTwoInterruptionsCloseTheTurnOnce(t *testing.T) {
	d := newDeferredClose(t)
	d.speakAndGetInterrupted(t)
	d.interrupt(t)

	d.closeInterrupted(t)
	d.closeInterrupted(t)
	d.settle(t)

	if got := d.contents(); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("conversation = %q, want the turn written once", got)
	}
	if got := d.interrupted(); !slices.Equal(got, []bool{true}) {
		t.Errorf("turns interrupted = %v, want one interrupted", got)
	}
}

func TestInterruptionWithNoOpenTurnArmsNoClose(t *testing.T) {
	d := newDeferredClose(t)
	d.interrupt(t)

	d.a.mu.Lock()
	armed := d.a.turnInterrupted
	d.a.mu.Unlock()
	if armed {
		t.Fatal("an interruption with no turn open armed a close")
	}

	// The next turn is an ordinary one.
	d.a.openTurn(t.Context())
	d.say("Hello")
	if err := d.a.commit(t.Context(), false); err != nil {
		t.Fatal(err)
	}
	d.settle(t)

	if got := d.contents(); !slices.Equal(got, []string{"Hello"}) {
		t.Errorf("conversation = %q, want the ordinary turn", got)
	}
	if got := d.interrupted(); !slices.Equal(got, []bool{false}) {
		t.Errorf("turns interrupted = %v, want one finished", got)
	}
}
