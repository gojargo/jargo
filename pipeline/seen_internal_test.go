package pipeline

import (
	"runtime"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// TestSeenFramesForgetsCollectedFrames checks the set holds its frames weakly:
// once nothing holds a frame it is collected and leaves the set, so the set
// does not grow with every frame of a session. A frame still held stays seen.
func TestSeenFramesForgetsCollectedFrames(t *testing.T) {
	var s seenFrames
	kept := frames.NewTextFrame("kept")
	if !s.add(kept) {
		t.Fatal("a new frame was reported as seen")
	}
	for range 1000 {
		s.add(frames.NewTextFrame("gone"))
	}

	deadline := time.Now().Add(5 * time.Second)
	for s.len() > 1 && time.Now().Before(deadline) {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	if n := s.len(); n != 1 {
		t.Errorf("the set holds %d frames, want only the one still held", n)
	}
	if s.add(kept) {
		t.Error("a frame still held was forgotten")
	}
	runtime.KeepAlive(kept)
}
