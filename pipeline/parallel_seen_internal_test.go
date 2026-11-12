package pipeline

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// TestParallelPipelineForgetsTheFramesItPushed checks the frames a parallel
// pipeline remembers to push once are forgotten once collected: a voice call
// sends tens of frames a second through it, and the set must not keep them all
// for the session.
func TestParallelPipelineForgetsTheFramesItPushed(t *testing.T) {
	pp, err := NewParallel(
		[]processor.Processor{processor.NewIdentityFilter("A")},
		[]processor.Processor{processor.NewIdentityFilter("B")},
	)
	if err != nil {
		t.Fatal(err)
	}
	w := NewWorker(New(pp), WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	for range 500 {
		w.QueueFrame(frames.NewTextFrame("hello"))
	}
	w.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	deadline := time.Now().Add(5 * time.Second)
	for pp.seen.len() > 50 && time.Now().Before(deadline) {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	if n := pp.seen.len(); n > 50 {
		t.Errorf("the parallel pipeline still remembers %d frames after 500 went through", n)
	}
}
