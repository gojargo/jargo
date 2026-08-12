package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/notify"
)

// linkAndStart puts p between two recorders and starts all three. Both sides
// carry a recorder because a frame goes to whichever neighbor lies in its
// direction.
func linkAndStart(t *testing.T, p processor.Processor) (up, down *capture) {
	t.Helper()
	up, down = newCapture(), newCapture()
	up.Link(p)
	p.Link(down)

	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	for _, each := range []processor.Processor{up, p, down} {
		if err := each.Setup(ctx, setup); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = each.Cleanup(context.Background()) })
	}
	// A processor drops frames until it has been started, so the run has to
	// begin before anything else is sent through it.
	if err := p.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	drain(down)
	return up, down
}

// hasType reports whether any of the frames is of type T.
func hasType[T frames.Frame](got []frames.Frame) bool {
	for _, f := range got {
		if _, ok := f.(T); ok {
			return true
		}
	}
	return false
}

// drain returns every frame a recorder holds.
func drain(c *capture) []frames.Frame {
	var got []frames.Frame
	for {
		select {
		case f := <-c.got:
			got = append(got, f)
		default:
			return got
		}
	}
}

// An identity filter changes nothing: what goes in comes out.
func TestIdentityFilterForwardsEverything(t *testing.T) {
	f := processor.NewIdentityFilter("Identity")
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream)
	_ = f.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	got := drain(down)
	if len(got) != 2 {
		t.Fatalf("got %d frames, want both through: %+v", len(got), got)
	}
	if _, ok := got[0].(*frames.TextFrame); !ok {
		t.Fatalf("first = %T, want a TextFrame", got[0])
	}
	if _, ok := got[1].(*frames.UserStartedSpeakingFrame); !ok {
		t.Fatalf("second = %T, want a UserStartedSpeakingFrame", got[1])
	}
}

// A frame filter passes the type it was built for.
func TestFrameFilterPassesTheTypeItWasBuiltFor(t *testing.T) {
	f := processor.NewFrameFilter("Filter", processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream)

	if got := drain(down); !hasType[*frames.TextFrame](got) {
		t.Fatalf("frames = %+v, want the text through", got)
	}
}

// A frame type the filter was not built for is dropped.
func TestFrameFilterDropsOtherTypes(t *testing.T) {
	f := processor.NewFrameFilter("Filter", processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)

	if got := drain(down); len(got) != 0 {
		t.Fatalf("frames = %+v, want nothing through", got)
	}
}

// The end of the run passes whatever the list says, or the processors behind
// the filter would never shut down.
func TestFrameFilterPassesTheEndFrame(t *testing.T) {
	f := processor.NewFrameFilter("Filter", processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewEndFrame(), processor.Downstream)

	if got := drain(down); !hasType[*frames.EndFrame](got) {
		t.Fatalf("frames = %+v, want the end frame through", got)
	}
}

// A system frame passes a filter built for nothing at all.
func TestFrameFilterPassesSystemFrames(t *testing.T) {
	f := processor.NewFrameFilter("Filter")
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	if got := drain(down); !hasType[*frames.UserStartedSpeakingFrame](got) {
		t.Fatalf("frames = %+v, want the system frame through", got)
	}
}

// A null filter drops ordinary frames.
func TestNullFilterDropsOrdinaryFrames(t *testing.T) {
	f := processor.NewNullFilter("Null")
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream)

	if got := drain(down); len(got) != 0 {
		t.Fatalf("frames = %+v, want nothing through", got)
	}
}

// It still passes what the pipeline needs to keep working.
func TestNullFilterPassesTheFramesThePipelineNeeds(t *testing.T) {
	f := processor.NewNullFilter("Null")
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	_ = f.QueueFrame(ctx, frames.NewEndFrame(), processor.Downstream)

	got := drain(down)
	if !hasType[*frames.UserStartedSpeakingFrame](got) || !hasType[*frames.EndFrame](got) {
		t.Fatalf("frames = %+v, want the system and end frames through", got)
	}
}

// A wake notifier signals when a frame it watches satisfies its predicate, and
// forwards the frame either way.
func TestWakeNotifierSignalsOnAMatch(t *testing.T) {
	n := notify.NewEventNotifier()
	f := processor.NewWakeNotifierFilter("Wake", n,
		func(frames.Frame) bool { return true },
		processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream)

	if got := drain(down); !hasType[*frames.TextFrame](got) {
		t.Fatalf("frames = %+v, want the frame forwarded", got)
	}
	if !n.Wait(ctx) {
		t.Fatal("the notifier was not signaled")
	}
}

// A predicate that says no leaves the notifier alone, and the frame still goes
// on: what the filter decides is whether to signal, not whether to forward.
func TestWakeNotifierStaysQuietWhenThePredicateSaysNo(t *testing.T) {
	n := notify.NewEventNotifier()
	f := processor.NewWakeNotifierFilter("Wake", n,
		func(frames.Frame) bool { return false },
		processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream)

	if got := drain(down); !hasType[*frames.TextFrame](got) {
		t.Fatalf("frames = %+v, want the frame forwarded", got)
	}
	quick, cancel := context.WithCancel(ctx)
	cancel()
	if n.Wait(quick) {
		t.Fatal("the notifier was signaled for a frame the predicate rejected")
	}
}

// A frame of a type the filter does not watch never reaches the predicate.
func TestWakeNotifierIgnoresUnwatchedTypes(t *testing.T) {
	n := notify.NewEventNotifier()
	asked := false
	f := processor.NewWakeNotifierFilter("Wake", n,
		func(frames.Frame) bool { asked = true; return true },
		processor.FrameIs[*frames.TextFrame]())
	_, down := linkAndStart(t, f)

	ctx := context.Background()
	_ = f.QueueFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	if asked {
		t.Fatal("the predicate was asked about a type the filter does not watch")
	}
	if got := drain(down); !hasType[*frames.UserStartedSpeakingFrame](got) {
		t.Fatalf("frames = %+v, want the frame forwarded", got)
	}
}
