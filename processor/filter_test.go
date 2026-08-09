package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// filterRig is a filter with a recorder on each side, set up and started. A
// recorder sits on both because a frame the filter forwards goes to whichever
// neighbor lies in its direction.
type filterRig struct {
	filter *processor.FunctionFilter
	down   *capture
	up     *capture
	seen   *[]frames.Frame
}

// newFilterRig builds a filter over allow and links a recorder behind it. Every
// frame the predicate is asked about is recorded too, because upstream consults
// it even for frames that pass regardless.
func newFilterRig(t *testing.T, dir *processor.Direction, allow processor.FilterFunc,
	opts ...processor.FilterOption,
) *filterRig {
	t.Helper()
	var seen []frames.Frame
	wrapped := func(f frames.Frame) bool {
		seen = append(seen, f)
		return allow(f)
	}
	f := processor.NewFunctionFilter("Filter", dir, wrapped, opts...)
	up, down := newCapture(), newCapture()
	up.Link(f)
	f.Link(down)

	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	for _, p := range []interface {
		Setup(context.Context, processor.Setup) error
		Cleanup(context.Context) error
	}{up, f, down} {
		if err := p.Setup(ctx, setup); err != nil {
			t.Fatalf("Setup: %v", err)
		}
		t.Cleanup(func() { _ = p.Cleanup(ctx) })
	}

	// Nothing is forwarded until the processor has been started, so prime it and
	// drop what priming produced.
	if err := f.ProcessFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("priming the filter: %v", err)
	}
	<-down.got
	seen = nil

	return &filterRig{filter: f, down: down, up: up, seen: &seen}
}

// send pushes one frame through the filter and reports whether it came out. The
// filter and the recorder both run in direct mode, so this is synchronous and
// what has not arrived by the time it returns was dropped.
func (r *filterRig) send(t *testing.T, f frames.Frame, dir processor.Direction) bool {
	t.Helper()
	if err := r.filter.ProcessFrame(context.Background(), f, dir); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}
	got := r.down.got
	if dir == processor.Upstream {
		got = r.up.got
	}
	select {
	case got := <-got:
		if got != f {
			t.Errorf("received %T, want the frame that was sent", got)
		}
		return true
	default:
		return false
	}
}

func allowAll(frames.Frame) bool { return true }
func blockAll(frames.Frame) bool { return false }

// TestFunctionFilterPassthrough is upstream's test_passthrough: a predicate that
// accepts everything forwards everything.
func TestFunctionFilterPassthrough(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, allowAll)

	if !r.send(t, frames.NewTextFrame("Hello jargo!"), processor.Downstream) {
		t.Error("an accepted frame did not come out")
	}
}

// TestFunctionFilterNoPassthrough is upstream's test_no_passthrough: a predicate
// that rejects everything drops an ordinary frame.
func TestFunctionFilterNoPassthrough(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, blockAll)

	if r.send(t, frames.NewTextFrame("Hello jargo!"), processor.Downstream) {
		t.Error("a rejected frame came out anyway")
	}
}

// TestFunctionFilterPassesTheOtherDirection is upstream's
// test_downstream_direction_passes_upstream and its mirror: a filter gating one
// direction leaves the other alone, however the predicate answers.
func TestFunctionFilterPassesTheOtherDirection(t *testing.T) {
	down, up := processor.Downstream, processor.Upstream

	r := newFilterRig(t, &down, blockAll)
	if !r.send(t, frames.NewTextFrame("upward"), processor.Upstream) {
		t.Error("a downstream filter dropped an upstream frame")
	}

	r = newFilterRig(t, &up, blockAll)
	if !r.send(t, frames.NewTextFrame("downward"), processor.Downstream) {
		t.Error("an upstream filter dropped a downstream frame")
	}
}

// TestFunctionFilterNoDirectionGatesBoth is upstream's
// test_no_direction_filters_both_directions: with no direction named, the
// predicate decides frames traveling either way.
func TestFunctionFilterNoDirectionGatesBoth(t *testing.T) {
	r := newFilterRig(t, nil, blockAll)

	if r.send(t, frames.NewTextFrame("downward"), processor.Downstream) {
		t.Error("a downstream frame came out of a filter gating both directions")
	}
	if r.send(t, frames.NewTextFrame("upward"), processor.Upstream) {
		t.Error("an upstream frame came out of a filter gating both directions")
	}
}

// TestFunctionFilterAlwaysPassesLifecycleFrames checks the frames that start and
// stop a pipeline reach a processor whose branch is gated off. A branch that
// never starts never runs, and one that never ends never shuts down, so these
// are not the predicate's to reject.
func TestFunctionFilterAlwaysPassesLifecycleFrames(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, blockAll, processor.WithFilterSystemFrames())

	for _, f := range []frames.Frame{
		frames.NewStartFrame(),
		frames.NewEndFrame(),
		frames.NewCancelFrame(),
	} {
		if !r.send(t, f, processor.Downstream) {
			t.Errorf("%T was dropped, want it to reach a gated-off branch", f)
		}
	}
}

// TestFunctionFilterPassesSystemFramesByDefault checks a system frame goes
// through unless the filter was built to decide those too. A filter gating
// ordinary traffic is not meant to swallow an interruption.
func TestFunctionFilterPassesSystemFramesByDefault(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, blockAll)

	if !r.send(t, frames.NewUserStartedSpeakingFrame(), processor.Downstream) {
		t.Error("a system frame was dropped by a filter that does not decide them")
	}
}

// TestFunctionFilterCanGateSystemFrames checks the opt-in: a filter built to
// decide system frames drops the ones its predicate rejects. It is what a
// service switcher needs, so a branch that is gated off stops following the
// conversation rather than hearing all of it in the background.
func TestFunctionFilterCanGateSystemFrames(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, blockAll, processor.WithFilterSystemFrames())

	if r.send(t, frames.NewUserStartedSpeakingFrame(), processor.Downstream) {
		t.Error("a system frame came out of a filter built to decide them")
	}
}

// TestFunctionFilterConsultsThePredicateForEveryFrame checks the predicate is
// asked about frames that pass regardless. A predicate watching the stream to
// decide what comes next has to see all of it, not only the part it can stop.
func TestFunctionFilterConsultsThePredicateForEveryFrame(t *testing.T) {
	down := processor.Downstream
	r := newFilterRig(t, &down, allowAll)

	r.send(t, frames.NewStartFrame(), processor.Downstream)
	r.send(t, frames.NewTextFrame("upward"), processor.Upstream)

	if len(*r.seen) != 2 {
		t.Errorf("the predicate saw %d frames, want the lifecycle frame and the one traveling the other way", len(*r.seen))
	}
}
