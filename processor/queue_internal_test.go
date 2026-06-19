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
