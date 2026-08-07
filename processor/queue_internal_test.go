package processor

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestQueueSystemPriority(t *testing.T) {
	q := newQueue()
	q.push(item{frame: frames.NewTextFrame("data")})          // data frame
	q.push(item{frame: frames.NewUserStartedSpeakingFrame()}) // system frame

	first, ok := q.get(context.Background())
	if !ok {
		t.Fatal("get returned ok=false")
	}
	if _, isSystem := first.frame.(frames.SystemFrame); !isSystem {
		t.Fatalf("expected the system frame first, got %s", first.frame.Name())
	}

	second, ok := q.get(context.Background())
	if !ok {
		t.Fatal("get returned ok=false")
	}
	if _, isData := second.frame.(frames.DataFrame); !isData {
		t.Fatalf("expected the data frame second, got %s", second.frame.Name())
	}
}

func TestQueueResetKeepsUninterruptible(t *testing.T) {
	q := newQueue()
	q.push(item{frame: frames.NewTextFrame("drop")}) // interruptible
	q.push(item{frame: frames.NewEndFrame()})        // uninterruptible
	q.reset()

	got, ok := q.get(context.Background())
	if !ok {
		t.Fatal("get returned ok=false")
	}
	if _, isEnd := got.frame.(*frames.EndFrame); !isEnd {
		t.Fatalf("expected the EndFrame to survive reset, got %s", got.frame.Name())
	}

	// The interruptible frame should be gone, leaving the queue empty.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, ok := q.get(ctx); ok {
		t.Fatal("expected the queue to be empty after reset")
	}
}

// TestQueueHasFrame checks the introspection a processor uses to tell that more
// of the same work is already queued behind the frame in hand, so it can act
// once on the batch rather than once per frame.
func TestQueueHasFrame(t *testing.T) {
	isText := func(f frames.Frame) bool { _, ok := f.(*frames.TextFrame); return ok }

	q := newQueue()
	if q.hasFrame(isText) {
		t.Error("an empty queue holds nothing")
	}

	q.push(item{frame: frames.NewTextFrame("queued")})
	if !q.hasFrame(isText) {
		t.Error("the queued text frame should be visible")
	}

	// A system frame is handled the moment it is queued, so it never waits and is
	// deliberately not reported.
	q2 := newQueue()
	q2.push(item{frame: frames.NewUserStartedSpeakingFrame()})
	if q2.hasFrame(func(frames.Frame) bool { return true }) {
		t.Error("system frames do not wait in the queue and must not be reported")
	}

	// Taking the frame off the queue leaves nothing behind.
	if _, ok := q.get(context.Background()); !ok {
		t.Fatal("get returned ok=false")
	}
	if q.hasFrame(isText) {
		t.Error("the frame was taken off the queue and should no longer be reported")
	}
}
