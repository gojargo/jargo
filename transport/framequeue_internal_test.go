package transport

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// The frame queue reads a frame's interruptible flag as it is.

func TestFrameQueueRespectsAFlagSetOnAQueuedFrame(t *testing.T) {
	q := newFrameQueue()
	frame := frames.NewTextFrame("hi")
	q.push(frame)
	if q.hasUninterruptible() {
		t.Fatal("a plain frame counted as uninterruptible")
	}

	frame.SetInterruptible(false)
	if !q.hasUninterruptible() {
		t.Fatal("a frame set uninterruptible after it was queued was not counted")
	}
	q.tryGet()
	if q.hasUninterruptible() {
		t.Fatal("the queue still counts a frame it no longer holds")
	}

	end := frames.NewEndFrame()
	q.push(end)
	if !q.hasUninterruptible() {
		t.Fatal("an EndFrame was not counted as uninterruptible")
	}
	end.SetInterruptible(true)
	if q.hasUninterruptible() {
		t.Fatal("an EndFrame set interruptible after it was queued was still counted")
	}
}

func TestFrameQueueResetKeepsWhatIsUninterruptibleNow(t *testing.T) {
	q := newFrameQueue()
	plain, end := frames.NewTextFrame("hi"), frames.NewEndFrame()
	q.push(plain)
	q.push(end)
	plain.SetInterruptible(false)
	end.SetInterruptible(true)

	q.reset()

	got, ok := q.tryGet()
	if !ok || got != plain {
		t.Fatalf("reset kept %v, want the frame set uninterruptible", got)
	}
	if _, ok := q.tryGet(); ok {
		t.Fatal("reset kept the frame set interruptible")
	}
}
