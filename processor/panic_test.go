package processor_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
)

// panicking panics on a text frame reading "boom" and on a UserIdleTimeout
// update (a system frame), and forwards everything else.
type panicking struct {
	*processor.Base
}

func newPanicking(opts ...processor.Option) *panicking {
	p := &panicking{}
	p.Base = processor.New("Panicking", p, opts...)
	return p
}

func (p *panicking) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.TextFrame:
		if fr.Text == "boom" {
			//nolint:forbidigo // the fault this test exists to contain
			panic("boom")
		}
	case *frames.UserIdleTimeoutUpdateFrame:
		//nolint:forbidigo // the fault this test exists to contain
		panic("system boom")
	}
	return p.PushFrame(ctx, f, dir)
}

// checkPanicReported checks an error frame reports the panic as an error of
// unknown cause raised by p.
func checkPanicReported(t *testing.T, ef *frames.ErrorFrame, p processor.Processor, want string) {
	t.Helper()
	if !strings.Contains(ef.Error, want) {
		t.Errorf("error = %q, want it to mention %q", ef.Error, want)
	}
	if ef.Err == nil {
		t.Error("the error frame carries no error")
	}
	if ef.Category != errs.Unknown {
		t.Errorf("category = %v, want unknown", ef.Category)
	}
	if ef.Source != p {
		t.Errorf("source = %v, want %s", ef.Source, p.Name())
	}
}

// TestPanicInProcessFrameIsReportedAndProcessingGoesOn checks a panic while
// handling a data frame becomes an error pushed upstream, and the processor
// goes on handling the frames after it.
func TestPanicInProcessFrameIsReportedAndProcessingGoesOn(t *testing.T) {
	p := newPanicking()
	up, down := linkAndStart(t, p)
	ctx := context.Background()

	queue(t, ctx, p, frames.NewTextFrame("boom"))
	queue(t, ctx, p, frames.NewTextFrame("after"))

	got := waitForErrors(t, up, 1)
	checkPanicReported(t, got[0], p, "Error processing frame: panic: boom")
	if !errors.Is(got[0].Err, processor.ErrPanic) {
		t.Errorf("err = %v, want it to wrap ErrPanic", got[0].Err)
	}
	if tf := mustReceive[*frames.TextFrame](t, down.got, "TextFrame"); tf.Text != "after" {
		t.Errorf("got %q downstream, want the frame after the panic", tf.Text)
	}
}

// TestPanicOnASystemFrameIsReportedAndProcessingGoesOn checks the same for a
// system frame, which is handled on the input goroutine.
func TestPanicOnASystemFrameIsReportedAndProcessingGoesOn(t *testing.T) {
	p := newPanicking()
	up, down := linkAndStart(t, p)
	ctx := context.Background()

	queue(t, ctx, p, frames.NewUserIdleTimeoutUpdateFrame(0))
	queue(t, ctx, p, frames.NewTextFrame("after"))

	got := waitForErrors(t, up, 1)
	checkPanicReported(t, got[0], p, "system boom")
	if tf := mustReceive[*frames.TextFrame](t, down.got, "TextFrame"); tf.Text != "after" {
		t.Errorf("got %q downstream, want the frame after the panic", tf.Text)
	}
}

// TestPanicInDirectModeIsReported checks a processor in direct mode reports
// the panic on the caller's goroutine and returns from QueueFrame normally.
func TestPanicInDirectModeIsReported(t *testing.T) {
	p := newPanicking(processor.WithDirectMode())
	up, _ := linkAndStart(t, p)

	if err := p.QueueFrame(context.Background(), frames.NewTextFrame("boom"), processor.Downstream); err != nil {
		t.Fatalf("QueueFrame returned %v for a panic already reported", err)
	}
	got := waitForErrors(t, up, 1)
	checkPanicReported(t, got[0], p, "boom")
}

// panickingObserver panics on the first text frame it sees pushed.
type panickingObserver struct {
	fired atomic.Bool
}

func (o *panickingObserver) OnPushFrame(data processor.FramePushed) {
	if _, ok := data.Frame.(*frames.TextFrame); ok && o.fired.CompareAndSwap(false, true) {
		//nolint:forbidigo // the fault this test exists to contain
		panic("observer boom")
	}
}

// TestPanicWhilePushingIsReported checks a panic raised while handing a frame
// on (here by an observer) becomes an error from the pushing processor rather
// than a crash.
func TestPanicWhilePushingIsReported(t *testing.T) {
	e := newEcho()
	up, down := newCapture(), newCapture()
	up.Link(e)
	e.Link(down)
	ctx := context.Background()
	s := processor.Setup{Clock: clock.NewSystem(), Observers: []processor.Observer{&panickingObserver{}}}
	for _, p := range []processor.Processor{up, e, down} {
		if err := p.Setup(ctx, s); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = p.Cleanup(context.Background()) })
	}
	queue(t, ctx, e, frames.NewStartFrame())
	mustReceive[*frames.StartFrame](t, down.got, "StartFrame")

	queue(t, ctx, e, frames.NewTextFrame("pushed"))
	got := waitForErrors(t, up, 1)
	checkPanicReported(t, got[0], e, "uncaught panic: panic: observer boom")

	queue(t, ctx, e, frames.NewTextFrame("after"))
	if tf := mustReceive[*frames.TextFrame](t, down.got, "TextFrame"); tf.Text != "after" {
		t.Errorf("got %q downstream, want the frame after the panic", tf.Text)
	}
}

// failing returns an error carrying an HTTP status from ProcessFrame.
type failing struct {
	*processor.Base
	err error
}

func (p *failing) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		return p.err
	}
	return p.PushFrame(ctx, f, dir)
}

// Ported from upstream (test_an_uncaught_exception_is_not_classified). An error
// returned from ProcessFrame carries a 401, but where it came from cannot be
// told, so it is reported unclassified and the processor stays usable.
func TestAnErrorReturnedFromProcessFrameIsNotClassified(t *testing.T) {
	p := &failing{err: errs.NewHTTPStatusError(401, errors.New("unauthorized"))} //nolint:err113 // a one-off cause
	p.Base = processor.New("Failing", p)
	up, _ := linkAndStart(t, p)

	queue(t, context.Background(), p, frames.NewTextFrame("hello"))
	got := waitForErrors(t, up, 1)
	if got[0].Category != errs.Unknown {
		t.Errorf("category = %v, want unknown", got[0].Category)
	}
	if !p.Usable() {
		t.Error("the processor was written off for an error whose cause cannot be told")
	}
}
