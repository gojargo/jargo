package pipeline

import (
	"runtime"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// countingObserver counts the pushes it is told about.
type countingObserver struct{ pushes chan struct{} }

func (o *countingObserver) OnPushFrame(processor.FramePushed) { o.pushes <- struct{}{} }

// TestAFrameIsForgottenOnceThePipelineLetsGoOfIt checks the proxy tracks the
// frames in flight, not every frame ever pushed.
func TestAFrameIsForgottenOnceThePipelineLetsGoOfIt(t *testing.T) {
	o := &countingObserver{pushes: make(chan struct{}, 16)}
	p := newObserverProxy([]processor.Observer{o})
	p.start()
	t.Cleanup(p.stop)

	push := func(f frames.Frame) {
		p.OnPushFrame(processor.FramePushed{Frame: f, Direction: processor.Downstream})
	}
	for range 10 {
		push(frames.NewTextFrame("hello"))
	}
	if n := p.pushedFrames(); n != 10 {
		t.Fatalf("tracking %d frames, want 10", n)
	}

	// The observer holds on to the last push it handled, so end with a frame
	// the test keeps and wait for every push to be handed over.
	kept := frames.NewTextFrame("kept")
	push(kept)
	for range 11 {
		<-o.pushes
	}

	deadline := time.Now().Add(5 * time.Second)
	for p.pushedFrames() > 1 && time.Now().Before(deadline) {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	if n := p.pushedFrames(); n != 1 {
		t.Fatalf("tracking %d frames after they were let go, want only the one kept", n)
	}
	runtime.KeepAlive(kept)
}
