package processor_test

import (
	"context"
	"sync"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for a processor interrupting the pipeline from inside its own frame
// handling, which is what something other than the user's voice cutting the bot
// off looks like.

// sink records every frame that reaches it, whichever way it came.
type sink struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

func newSink(name string) *sink {
	s := &sink{}
	s.Base = processor.New(name, s)
	return s
}

func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	s.mu.Lock()
	s.got = append(s.got, f)
	s.mu.Unlock()
	return s.PushFrame(ctx, f, dir)
}

// counts returns how many frames matching match reached this sink.
func (s *sink) counts(match func(frames.Frame) bool) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, f := range s.got {
		if match(f) {
			n++
		}
	}
	return n
}

// interrupter interrupts the pipeline when it sees a text frame, then carries on
// and pushes a frame of its own.
type interrupter struct {
	*processor.Base
	mu      sync.Mutex
	carried bool
}

func newInterrupter() *interrupter {
	p := &interrupter{}
	p.Base = processor.New("Interrupter", p)
	return p
}

func (p *interrupter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); !ok {
		return p.PushFrame(ctx, f, dir)
	}
	if err := p.BroadcastInterruption(ctx); err != nil {
		return err
	}
	p.mu.Lock()
	p.carried = true
	p.mu.Unlock()
	return p.PushFrame(ctx, frames.NewOutputTransportMessageUrgentFrame("done"), processor.Downstream)
}

func (p *interrupter) carriedOn() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.carried
}

func isInterruption(f frames.Frame) bool {
	_, ok := f.(*frames.InterruptionFrame)
	return ok
}

// TestBroadcastInterruptionGoesBothWays checks that an interruption raised
// inside a processor reaches the processors on either side of it, so neither
// half of the pipeline is left carrying on with the turn that was cut off.
func TestBroadcastInterruptionGoesBothWays(t *testing.T) {
	ctx := t.Context()
	up, mid, down := newSink("Up"), newInterrupter(), newSink("Down")
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, up, mid, down)

	queue(t, ctx, up, frames.NewStartFrame())
	queue(t, ctx, up, frames.NewTextFrame("cut me off"))

	if !waitFor(t, func() bool {
		return up.counts(isInterruption) == 1 && down.counts(isInterruption) == 1
	}) {
		t.Fatalf("interruptions seen: %d upstream, %d downstream, want one each way",
			up.counts(isInterruption), down.counts(isInterruption))
	}
}

// TestBroadcastInterruptionAllowsSubsequentCode checks that interrupting does
// not cancel the caller. A processor interrupts in order to say something in
// place of what it cut off, so it has to still be running afterwards.
func TestBroadcastInterruptionAllowsSubsequentCode(t *testing.T) {
	ctx := t.Context()
	up, mid, down := newSink("Up"), newInterrupter(), newSink("Down")
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, up, mid, down)

	queue(t, ctx, up, frames.NewStartFrame())
	queue(t, ctx, up, frames.NewTextFrame("cut me off"))

	isUrgent := func(f frames.Frame) bool {
		_, ok := f.(*frames.OutputTransportMessageUrgentFrame)
		return ok
	}
	if !waitFor(t, func() bool { return down.counts(isUrgent) == 1 }) {
		t.Fatal("the frame pushed after the interruption never arrived")
	}
	if !mid.carriedOn() {
		t.Error("the code after the interruption did not run")
	}
}

// TestBroadcastInterruptionDropsQueuedWork checks that the work a processor had
// queued behind the frame it is handling is dropped, which is the point of
// interrupting: what was queued belongs to the turn that was cut off.
func TestBroadcastInterruptionDropsQueuedWork(t *testing.T) {
	ctx := t.Context()
	h := newHolder()
	if err := h.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Cleanup(context.Background()) })

	queue(t, ctx, h, frames.NewStartFrame())
	queue(t, ctx, h, frames.NewTextFrame("held"))
	<-h.entered

	queue(t, ctx, h, frames.NewTextFrame("queued behind it"))
	isText := func(f frames.Frame) bool { _, ok := f.(*frames.TextFrame); return ok }
	if !waitFor(t, func() bool { return h.HasQueuedFrame(isText) }) {
		t.Fatal("the frame never reached the in-order queue")
	}

	if err := h.BroadcastInterruption(ctx); err != nil {
		t.Fatal(err)
	}
	if h.HasQueuedFrame(isText) {
		t.Error("the queued frame survived the interruption")
	}
	close(h.release)
}
