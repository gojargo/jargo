package pipeline_test

import (
	"context"
	"errors"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// tagger replaces every downstream TextFrame with a new one whose text is
// prefixed with tag, so each branch of a parallel pipeline produces a distinct
// frame. Other frames pass through unchanged.
type tagger struct {
	*processor.Base
	tag string
}

func newTagger(tag string) *tagger {
	t := &tagger{tag: tag}
	t.Base = processor.New("Tagger", t)
	return t
}

func (t *tagger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := t.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream {
		return t.PushFrame(ctx, frames.NewTextFrame(t.tag+tf.Text), dir)
	}
	return t.PushFrame(ctx, f, dir)
}

// runParallel runs a task wrapping pp, queues frames, stops when done, and
// returns every frame that reached the end of the pipeline.
func runParallel(t *testing.T, pp *pipeline.ParallelPipeline, in []frames.Frame) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(pipeline.New(pp), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrames(in)
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parallel task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

func TestParallelFanOutMerge(t *testing.T) {
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newTagger("A")},
		[]processor.Processor{newTagger("B")},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	got := runParallel(t, pp, []frames.Frame{frames.NewTextFrame("hi")})

	var texts []string
	starts, ends := 0, 0
	for _, f := range got {
		switch fr := f.(type) {
		case *frames.TextFrame:
			texts = append(texts, fr.Text)
		case *frames.StartFrame:
			starts++
		case *frames.EndFrame:
			ends++
		}
	}
	sort.Strings(texts)

	if len(texts) != 2 || texts[0] != "Ahi" || texts[1] != "Bhi" {
		t.Errorf("downstream texts = %v, want [Ahi Bhi]", texts)
	}
	// Each branch forwards the same lifecycle frame; the merge must release one.
	if starts != 1 {
		t.Errorf("StartFrame reached downstream %d times, want 1", starts)
	}
	if ends != 1 {
		t.Errorf("EndFrame reached downstream %d times, want 1", ends)
	}
}

func TestParallelPassthroughDeduplicated(t *testing.T) {
	// Both branches forward the input frame unchanged, so it shares a frame id at
	// both sinks and must escape exactly once.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEcho()},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	got := runParallel(t, pp, []frames.Frame{frames.NewTextFrame("once")})

	n := 0
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "once" {
			n++
		}
	}
	if n != 1 {
		t.Errorf("passthrough TextFrame reached downstream %d times, want 1", n)
	}
}

func TestParallelNoBranches(t *testing.T) {
	if _, err := pipeline.NewParallel(); err == nil {
		t.Fatal("NewParallel() with no branches: want error, got nil")
	}
}

func TestParallelLifecycleEndToEnd(t *testing.T) {
	// A parallel pipeline with branches of differing length still completes its
	// run: the EndFrame is released only after both branches have flushed.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEcho()},
		[]processor.Processor{newEcho(), newEcho(), newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	task := pipeline.NewTask(pipeline.New(pp), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTextFrame("x"))
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parallel task did not finish")
	}
	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
}
