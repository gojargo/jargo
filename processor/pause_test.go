package processor_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// pausePair links an echo to a capture, sets both up and starts them, returning
// the pair and a cleanup.
func pausePair(t *testing.T) (*echo, *capture, func()) {
	t.Helper()

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

	_ = e.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream)
	mustReceive[*frames.StartFrame](t, c.got, "StartFrame")

	return e, c, func() {
		_ = e.Cleanup(ctx)
		_ = c.Cleanup(ctx)
	}
}

// silentFor reports whether nothing arrives on got for d.
func silentFor(got chan frames.Frame, d time.Duration) bool {
	select {
	case <-got:
		return false
	case <-time.After(d):
		return true
	}
}

func TestPauseProcessingFramesHoldsDataFrames(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	e.PauseProcessingFrames()
	_ = e.QueueFrame(ctx, frames.NewTextFrame("held"), processor.Downstream)

	if !silentFor(c.got, 200*time.Millisecond) {
		t.Fatal("frame was handled while processing was paused")
	}

	e.ResumeProcessingFrames()
	got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
	if got.Text != "held" {
		t.Errorf("Text = %q, want held", got.Text)
	}
}

func TestPauseProcessingFramesKeepsOrder(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	e.PauseProcessingFrames()
	for _, text := range []string{"one", "two", "three"} {
		_ = e.QueueFrame(ctx, frames.NewTextFrame(text), processor.Downstream)
	}
	if !silentFor(c.got, 100*time.Millisecond) {
		t.Fatal("frame was handled while processing was paused")
	}

	e.ResumeProcessingFrames()
	for _, want := range []string{"one", "two", "three"} {
		got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
		if got.Text != want {
			t.Fatalf("Text = %q, want %q", got.Text, want)
		}
	}
}

func TestPauseProcessingFramesLetsSystemFramesThrough(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	e.PauseProcessingFrames()
	_ = e.QueueFrame(ctx, frames.NewTextFrame("held"), processor.Downstream)
	_ = e.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	// Pausing data and control frames must not hold system frames back.
	mustReceive[*frames.UserStartedSpeakingFrame](t, c.got, "UserStartedSpeakingFrame")

	e.ResumeProcessingFrames()
	mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
}

func TestPauseProcessingSystemFramesHoldsEverything(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	e.PauseProcessingSystemFrames()
	_ = e.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	if !silentFor(c.got, 200*time.Millisecond) {
		t.Fatal("system frame was handled while system-frame processing was paused")
	}

	e.ResumeProcessingSystemFrames()
	mustReceive[*frames.UserStartedSpeakingFrame](t, c.got, "UserStartedSpeakingFrame")
}

func TestPauseIsOneShot(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	e.PauseProcessingFrames()
	_ = e.QueueFrame(ctx, frames.NewTextFrame("first"), processor.Downstream)
	e.ResumeProcessingFrames()
	mustReceive[*frames.TextFrame](t, c.got, "TextFrame")

	// One resume releases the processor for good, until it is paused again.
	_ = e.QueueFrame(ctx, frames.NewTextFrame("second"), processor.Downstream)
	got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
	if got.Text != "second" {
		t.Errorf("Text = %q, want second", got.Text)
	}
}

func TestPauseFramesAddressTheirProcessor(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	_ = e.QueueFrame(ctx, frames.NewFrameProcessorPauseFrame(e), processor.Downstream)
	mustReceive[*frames.FrameProcessorPauseFrame](t, c.got, "FrameProcessorPauseFrame")

	_ = e.QueueFrame(ctx, frames.NewTextFrame("held"), processor.Downstream)
	if !silentFor(c.got, 200*time.Millisecond) {
		t.Fatal("frame was handled after a pause frame addressed to this processor")
	}

	_ = e.QueueFrame(ctx, frames.NewFrameProcessorResumeUrgentFrame(e), processor.Downstream)
	mustReceive[*frames.FrameProcessorResumeUrgentFrame](t, c.got, "FrameProcessorResumeUrgentFrame")
	mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
}

func TestPauseFrameIgnoredWhenAddressedElsewhere(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	other := newEcho()
	_ = e.QueueFrame(ctx, frames.NewFrameProcessorPauseFrame(other), processor.Downstream)
	mustReceive[*frames.FrameProcessorPauseFrame](t, c.got, "FrameProcessorPauseFrame")

	_ = e.QueueFrame(ctx, frames.NewTextFrame("through"), processor.Downstream)
	got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
	if got.Text != "through" {
		t.Errorf("Text = %q, want through", got.Text)
	}
}

func TestCleanupReleasesAPausedProcessor(t *testing.T) {
	e, _, done := pausePair(t)

	ctx := context.Background()
	e.PauseProcessingFrames()
	e.PauseProcessingSystemFrames()
	_ = e.QueueFrame(ctx, frames.NewTextFrame("held"), processor.Downstream)

	// Cleanup must not hang on the goroutines the pause is holding.
	finished := make(chan struct{})
	go func() {
		done()
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Fatal("Cleanup blocked on a paused processor")
	}
}

// A resume reaches the rest of the pipeline before the frames it let through.
// The frame carrying the resume and the frames it releases travel on different
// goroutines, so without ordering the released ones can overtake the resume that
// released them, and a processor downstream sees a paused branch's backlog
// arrive before it is told the branch was resumed.
func TestResumeFramePrecedesTheFramesItReleases(t *testing.T) {
	e, c, done := pausePair(t)
	defer done()

	ctx := context.Background()
	_ = e.QueueFrame(ctx, frames.NewFrameProcessorPauseFrame(e), processor.Downstream)
	mustReceive[*frames.FrameProcessorPauseFrame](t, c.got, "FrameProcessorPauseFrame")

	for _, text := range []string{"one", "two", "three"} {
		_ = e.QueueFrame(ctx, frames.NewTextFrame(text), processor.Downstream)
	}
	if !silentFor(c.got, 200*time.Millisecond) {
		t.Fatal("a frame was handled while the processor was paused")
	}

	_ = e.QueueFrame(ctx, frames.NewFrameProcessorResumeUrgentFrame(e), processor.Downstream)

	// The resume comes out first, then everything it was holding, in order.
	mustReceive[*frames.FrameProcessorResumeUrgentFrame](t, c.got, "FrameProcessorResumeUrgentFrame")
	for _, want := range []string{"one", "two", "three"} {
		got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame")
		if got.Text != want {
			t.Fatalf("Text = %q, want %q: the held frames must keep their order", got.Text, want)
		}
	}
}

// TestPauseUntilReadyHoldsFramesAndReleasesThem covers a processor that cannot
// act until a condition holds: the frames wait and are delivered in order once
// it does, rather than being dropped.
func TestPauseUntilReadyHoldsFramesAndReleasesThem(t *testing.T) {
	e := newEcho()
	up, down := linkAndStart(t, e)
	_ = up

	ready := make(chan struct{})
	e.PauseProcessingAllFramesUntil(func(ctx context.Context) {
		select {
		case <-ready:
		case <-ctx.Done():
		}
	}, 5*time.Second)

	ctx := context.Background()
	for _, text := range []string{"one", "two"} {
		if err := e.QueueFrame(ctx, frames.NewTextFrame(text), processor.Downstream); err != nil {
			t.Fatal(err)
		}
	}

	if got := drain(down); len(got) != 0 {
		t.Fatalf("frames %v got through while the processor was holding them", got)
	}

	close(ready)

	var got []string
	deadline := time.After(3 * time.Second)
	for len(got) < 2 {
		select {
		case f := <-down.got:
			if tf, ok := f.(*frames.TextFrame); ok {
				got = append(got, tf.Text)
			}
		case <-deadline:
			t.Fatalf("held frames = %v, want both released", got)
		}
	}
	if got[0] != "one" || got[1] != "two" {
		t.Errorf("held frames = %v, want them in the order they were queued", got)
	}
}

// TestPauseUntilReadyGivesUp covers the timeout: a processor left holding
// frames could not handle the ones that shut it down, so the hold is always
// lifted.
func TestPauseUntilReadyGivesUp(t *testing.T) {
	e := newEcho()
	_, down := linkAndStart(t, e)

	// A condition that never holds.
	e.PauseProcessingAllFramesUntil(func(ctx context.Context) { <-ctx.Done() }, 100*time.Millisecond)

	if err := e.QueueFrame(context.Background(), frames.NewTextFrame("one"), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-down.got:
			if tf, ok := f.(*frames.TextFrame); ok {
				if tf.Text != "one" {
					t.Errorf("got %q, want the held frame released when the wait gave up", tf.Text)
				}
				return
			}
		case <-deadline:
			t.Fatal("the hold was never lifted")
		}
	}
}

// TestPauseUntilReadyIsLiftedAtCleanup covers the hold not outliving the
// processor.
func TestPauseUntilReadyIsLiftedAtCleanup(t *testing.T) {
	e := newEcho()
	_, _ = linkAndStart(t, e)

	e.PauseProcessingAllFramesUntil(func(ctx context.Context) { <-ctx.Done() }, time.Hour)

	done := make(chan error, 1)
	go func() { done <- e.Cleanup(context.Background()) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("cleanup blocked on a hold that was never lifted")
	}
}

// TestPauseUntilReadyHoldsBothQueues covers the data frames already routed to
// the process queue being held too. The two queues pause independently, so
// frames that have moved on keep draining unless both are held.
func TestPauseUntilReadyHoldsBothQueues(t *testing.T) {
	e := newEcho()
	_, down := linkAndStart(t, e)

	e.PauseProcessingAllFramesUntil(func(ctx context.Context) { <-ctx.Done() }, 5*time.Second)

	ctx := context.Background()
	// A system frame and a data frame: one is held on each queue.
	if err := e.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if err := e.QueueFrame(ctx, frames.NewTextFrame("held"), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	if got := drain(down); len(got) != 0 {
		t.Errorf("frames %v got through while both queues were held", got)
	}
}

// TestPauseUntilReadyIsANoOpInDirectMode covers a processor that bypasses the
// queues the hold acts on: there is nothing to hold, so the call says so.
func TestPauseUntilReadyIsANoOpInDirectMode(t *testing.T) {
	e := newEchoDirect()
	_, down := linkAndStart(t, e)

	e.PauseProcessingAllFramesUntil(func(ctx context.Context) { <-ctx.Done() }, 5*time.Second)

	if err := e.QueueFrame(context.Background(), frames.NewTextFrame("through"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	deadline := time.After(2 * time.Second)
	for {
		select {
		case f := <-down.got:
			if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "through" {
				return
			}
		case <-deadline:
			t.Fatal("a direct-mode processor held frames it has no queue for")
		}
	}
}
