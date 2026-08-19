package processor_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for the order a processor takes frames off its queues in, and for what
// happens to frames that arrive before it has been started. Processors are set
// up concurrently, so one that connects during setup can push frames at a
// processor whose StartFrame has not reached it yet.

// namer records the name of every frame it handles, in order.
type namer struct {
	*processor.Base
	mu   sync.Mutex
	seen []string
}

func newNamer() *namer {
	n := &namer{}
	n.Base = processor.New("Namer", n)
	return n
}

func (n *namer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := n.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	n.mu.Lock()
	// The instance number is not what this test is about.
	n.seen = append(n.seen, strings.SplitN(f.Name(), "#", 2)[0])
	n.mu.Unlock()
	return n.PushFrame(ctx, f, dir)
}

func (n *namer) handled() []string {
	n.mu.Lock()
	defer n.mu.Unlock()
	return append([]string(nil), n.seen...)
}

// waitFor polls until cond holds or the deadline passes.
func waitFor(t *testing.T, cond func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(time.Millisecond)
	}
	return cond()
}

// TestStartFrameIsProcessedFirst checks that a processor holds every frame it
// receives until its StartFrame, and that the StartFrame is handled ahead of
// them even when what is waiting is itself a system frame.
func TestStartFrameIsProcessedFirst(t *testing.T) {
	ctx := t.Context()
	n := newNamer()
	if err := n.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Cleanup(context.Background()) })

	queue(t, ctx, n, frames.NewTextFrame("early"))
	queue(t, ctx, n, frames.NewUserStartedSpeakingFrame())

	time.Sleep(100 * time.Millisecond)
	if got := n.handled(); len(got) != 0 {
		t.Fatalf("frames handled before the StartFrame: %v", got)
	}

	queue(t, ctx, n, frames.NewStartFrame())

	want := []string{"StartFrame", "UserStartedSpeakingFrame", "TextFrame"}
	if !waitFor(t, func() bool { return len(n.handled()) == len(want) }) {
		t.Fatalf("handled = %v, want %v", n.handled(), want)
	}
	got := n.handled()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("handled = %v, want %v", got, want)
		}
	}
}

// TestFramesQueuedBeforeTheStartFrameSurvive checks that a data frame pushed at
// a processor that has not started yet is delivered rather than dropped. The
// process goroutine flushes its queue when it is created, so a frame already
// moved there would be lost.
func TestFramesQueuedBeforeTheStartFrameSurvive(t *testing.T) {
	ctx := t.Context()
	n := newNamer()
	if err := n.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Cleanup(context.Background()) })

	for range 3 {
		queue(t, ctx, n, frames.NewTextFrame("early"))
	}
	queue(t, ctx, n, frames.NewStartFrame())

	if !waitFor(t, func() bool { return len(n.handled()) == 4 }) {
		t.Fatalf("handled = %v, want the StartFrame and all three text frames", n.handled())
	}
}

func queue(t *testing.T, ctx context.Context, p processor.Processor, f frames.Frame) {
	t.Helper()
	if err := p.QueueFrame(ctx, f, processor.Downstream); err != nil {
		t.Fatal(err)
	}
}
