package processor_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// echo forwards every frame it receives to the next processor.
type echo struct {
	*processor.Base
}

func newEcho() *echo {
	e := &echo{}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echo) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return e.PushFrame(ctx, f, dir)
}

// capture records every frame it receives on a channel. It runs in direct mode
// so it records on the caller's goroutine.
type capture struct {
	*processor.Base
	got chan frames.Frame
}

func newCapture() *capture {
	c := &capture{got: make(chan frames.Frame, 16)}
	c.Base = processor.New("Capture", c, processor.WithDirectMode())
	return c
}

func (c *capture) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	c.got <- f
	return nil
}

func TestEchoRoundTrip(t *testing.T) {
	e := newEcho()
	c := newCapture()
	e.Link(c)

	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	if err := e.Setup(ctx, setup); err != nil {
		t.Fatal(err)
	}
	if err := c.Setup(ctx, setup); err != nil {
		t.Fatal(err)
	}
	defer e.Cleanup(ctx)
	defer c.Cleanup(ctx)

	// The StartFrame primes the processors and must come out first.
	_ = e.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream)
	mustReceive[*frames.StartFrame](t, c.got, "StartFrame")

	tf := frames.NewTextFrame("hello")
	_ = e.QueueFrame(ctx, tf, processor.Downstream)
	got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
	if got.Text != "hello" {
		t.Fatalf("Text = %q, want hello", got.Text)
	}
}

// blocker blocks while processing a TextFrame until its context is canceled,
// recording the context error it observes. It lets a test drive an interruption
// of an in-flight frame.
type blocker struct {
	*processor.Base
	entered chan struct{}
	result  chan error
}

func newBlocker() *blocker {
	b := &blocker{entered: make(chan struct{}), result: make(chan error, 1)}
	b.Base = processor.New("Blocker", b)
	return b
}

func (b *blocker) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		close(b.entered)
		<-ctx.Done()
		b.result <- ctx.Err()
		return nil
	}
	return b.PushFrame(ctx, f, dir)
}

func TestInterruptionCancelsInFlightFrame(t *testing.T) {
	b := newBlocker()
	ctx := context.Background()
	if err := b.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatal(err)
	}
	defer b.Cleanup(ctx)

	_ = b.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream)
	_ = b.QueueFrame(ctx, frames.NewTextFrame("slow"), processor.Downstream)

	select {
	case <-b.entered:
	case <-time.After(time.Second):
		t.Fatal("blocker never began processing the data frame")
	}

	// The interruptible frame is in flight; an interruption must cancel it.
	_ = b.QueueFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)

	select {
	case err := <-b.result:
		if err == nil {
			t.Fatal("expected ProcessFrame to observe a canceled context")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("interruption did not cancel the in-flight frame")
	}
}

func mustReceive[T frames.Frame](t *testing.T, ch <-chan frames.Frame, name string) T {
	t.Helper()
	select {
	case f := <-ch:
		got, ok := f.(T)
		if !ok {
			t.Fatalf("expected %s, got %s", name, f.Name())
		}
		return got
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %s", name)
	}
	var zero T
	return zero
}
