package processor_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// markerFrame is a data frame whose type is uninterruptible by default.
type markerFrame struct {
	frames.BaseDataFrame
	frames.UninterruptibleMixin
	Text string
}

func newMarkerFrame(text string) *markerFrame {
	return &markerFrame{BaseDataFrame: frames.NewBaseDataFrame("markerFrame"), Text: text}
}

// delayer holds every frame that is not a system frame for a while before
// pushing it on, so an interruption lands while it is being processed.
type delayer struct {
	*processor.Base
}

func newDelayer() *delayer {
	d := &delayer{}
	d.Base = processor.New("Delayer", d)
	return d
}

func (d *delayer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := d.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, system := f.(frames.SystemFrame); !system {
		select {
		case <-time.After(400 * time.Millisecond):
		case <-ctx.Done():
			return nil
		}
	}
	return d.PushFrame(ctx, f, dir)
}

// TestInterruptibleFlagDecidesForAFrame checks a plain frame marked
// uninterruptible survives an interruption, and a frame of an uninterruptible
// type marked interruptible does not.
func TestInterruptibleFlagDecidesForAFrame(t *testing.T) {
	ctx := t.Context()
	d, s := newDelayer(), newSink("Sink")
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, d, s)

	kept := frames.NewTextFrame("kept")
	kept.SetInterruptible(false)
	dropped := newMarkerFrame("dropped")
	dropped.SetInterruptible(true)

	queue(t, ctx, d, frames.NewStartFrame())
	queue(t, ctx, d, kept)
	queue(t, ctx, d, dropped)
	time.Sleep(100 * time.Millisecond)
	queue(t, ctx, d, frames.NewInterruptionFrame())

	isText := func(f frames.Frame) bool { _, ok := f.(*frames.TextFrame); return ok }
	isMarker := func(f frames.Frame) bool { _, ok := f.(*markerFrame); return ok }
	if !waitFor(t, func() bool { return s.counts(isText) == 1 }) {
		t.Fatal("the frame marked uninterruptible did not survive the interruption")
	}
	time.Sleep(600 * time.Millisecond)
	if n := s.counts(isMarker); n != 0 {
		t.Fatalf("the frame marked interruptible survived the interruption (%d)", n)
	}
	if n := s.counts(isInterruption); n != 1 {
		t.Fatalf("got %d interruptions downstream, want 1", n)
	}
}
