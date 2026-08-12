package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for the two endpoints that close a pipeline off. Each hands the frames
// that reach the edge to an external handler and forwards the rest along the
// chain, so a task can watch what leaves the pipeline in either direction.

// handled records the frames an endpoint handler was given.
type handled struct {
	frames []frames.Frame
	dirs   []processor.Direction
}

func (h *handled) handle(_ context.Context, f frames.Frame, dir processor.Direction) error {
	h.frames = append(h.frames, f)
	h.dirs = append(h.dirs, dir)
	return nil
}

// TestSourceRoutesByDirection checks that a source forwards downstream frames
// along the chain and hands upstream ones to its handler, which is how a task
// sees the frames coming back to the top of the pipeline.
func TestSourceRoutesByDirection(t *testing.T) {
	var up handled
	src := processor.NewSource("Source", up.handle)
	c := newCapture()

	ctx := context.Background()
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, src, c)

	// Downstream frames travel on to the next processor.
	if err := src.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, c.got, "StartFrame")

	if err := src.QueueFrame(ctx, frames.NewTextFrame("down"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame"); got.Text != "down" {
		t.Fatalf("Text = %q, want down", got.Text)
	}

	// Upstream frames stop here and go to the handler instead.
	if err := src.QueueFrame(ctx, frames.NewTextFrame("up"), processor.Upstream); err != nil {
		t.Fatal(err)
	}
	if len(up.frames) != 1 {
		t.Fatalf("the handler saw %d frames, want 1", len(up.frames))
	}
	if tf, ok := up.frames[0].(*frames.TextFrame); !ok || tf.Text != "up" {
		t.Fatalf("the handler saw %v, want the upstream TextFrame", up.frames[0])
	}
	if up.dirs[0] != processor.Upstream {
		t.Fatalf("the handler was given direction %s, want upstream", up.dirs[0])
	}
}

// TestSinkRoutesByDirection checks the mirror image: a sink hands downstream
// frames to its handler and pushes upstream ones back along the chain.
func TestSinkRoutesByDirection(t *testing.T) {
	var down handled
	c := newCapture()
	sink := processor.NewSink("Sink", down.handle)

	ctx := context.Background()
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, c, sink)

	// The sink needs its StartFrame before it will push anything.
	if err := sink.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if len(down.frames) != 1 {
		t.Fatalf("the handler saw %d frames, want the StartFrame", len(down.frames))
	}

	if err := sink.QueueFrame(ctx, frames.NewTextFrame("down"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if tf, ok := down.frames[1].(*frames.TextFrame); !ok || tf.Text != "down" {
		t.Fatalf("the handler saw %v, want the downstream TextFrame", down.frames[1])
	}
	if down.dirs[1] != processor.Downstream {
		t.Fatalf("the handler was given direction %s, want downstream", down.dirs[1])
	}

	// Upstream frames travel back to the previous processor.
	if err := sink.QueueFrame(ctx, frames.NewTextFrame("up"), processor.Upstream); err != nil {
		t.Fatal(err)
	}
	if got := mustReceive[*frames.TextFrame](t, c.got, "TextFrame"); got.Text != "up" {
		t.Fatalf("Text = %q, want up", got.Text)
	}
	if len(down.frames) != 2 {
		t.Fatalf("the handler saw %d frames, want the upstream one to have passed it by", len(down.frames))
	}
}
