package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// taggedFrame is a distinguishable data frame, so a test can tell which branch
// of a sync parallel pipeline produced which output.
type taggedFrame struct {
	frames.BaseDataFrame
	tag string
}

func newTaggedFrame(tag string) *taggedFrame {
	return &taggedFrame{BaseDataFrame: frames.NewBaseDataFrame("TaggedFrame"), tag: tag}
}

// emitTagged replaces every TextFrame with a taggedFrame, after an optional
// delay that makes one branch slower than another.
type emitTagged struct {
	*processor.Base
	tag   string
	delay time.Duration
}

func newEmitTagged(tag string, delay time.Duration) *emitTagged {
	e := &emitTagged{tag: tag, delay: delay}
	e.Base = processor.New("EmitTagged", e)
	return e
}

func (e *emitTagged) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); !ok {
		return e.PushFrame(ctx, f, dir)
	}
	if e.delay > 0 {
		select {
		case <-time.After(e.delay):
		case <-ctx.Done():
			return nil
		}
	}
	return e.PushFrame(ctx, newTaggedFrame(e.tag), processor.Downstream)
}

// runSyncParallel runs a task wrapping spp, queues frames, stops when done and
// returns every frame that reached the end of the pipeline.
func runSyncParallel(t *testing.T, spp *pipeline.SyncParallelPipeline, in []frames.Frame) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(spp), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
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
	case <-time.After(5 * time.Second):
		t.Fatal("sync parallel task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

// tags returns the tag of every taggedFrame in fs, in order.
func tags(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*taggedFrame); ok {
			out = append(out, tf.tag)
		}
	}
	return out
}

func TestSyncParallelNoBranches(t *testing.T) {
	if _, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival); err == nil {
		t.Fatal("NewSyncParallel() with no branches: want error, got nil")
	}
}

func TestSyncParallelDedupMultipleFrames(t *testing.T) {
	// The same frame comes out of both branches unchanged, so it must escape once.
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival,
		[]processor.Processor{newEcho()},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewSyncParallel: %v", err)
	}

	got := runSyncParallel(t, spp, []frames.Frame{
		frames.NewTextFrame("one"), frames.NewTextFrame("two"),
	})

	var texts []string
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok {
			texts = append(texts, tf.Text)
		}
	}
	if len(texts) != 2 || texts[0] != "one" || texts[1] != "two" {
		t.Errorf("downstream texts = %v, want [one two]", texts)
	}
}

func TestSyncParallelArrivalOrder(t *testing.T) {
	// In arrival order a slow first branch's output lands after a fast second
	// branch's, in each batch.
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival,
		[]processor.Processor{newEmitTagged("slow", 50*time.Millisecond)},
		[]processor.Processor{newEmitTagged("fast", 0)},
	)
	if err != nil {
		t.Fatalf("NewSyncParallel: %v", err)
	}

	got := tags(runSyncParallel(t, spp, []frames.Frame{
		frames.NewTextFrame("one"), frames.NewTextFrame("two"),
	}))

	want := []string{"fast", "slow", "fast", "slow"}
	if !equalStrings(got, want) {
		t.Errorf("tags = %v, want %v", got, want)
	}
}

func TestSyncParallelPipelineOrder(t *testing.T) {
	// In branch order each batch follows the order the branches were declared in,
	// however fast each one is.
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderPipeline,
		[]processor.Processor{newEmitTagged("slow", 50*time.Millisecond)},
		[]processor.Processor{newEmitTagged("fast", 0)},
	)
	if err != nil {
		t.Fatalf("NewSyncParallel: %v", err)
	}

	got := tags(runSyncParallel(t, spp, []frames.Frame{
		frames.NewTextFrame("one"), frames.NewTextFrame("two"),
	}))

	want := []string{"slow", "fast", "slow", "fast"}
	if !equalStrings(got, want) {
		t.Errorf("tags = %v, want %v", got, want)
	}
}

func TestSyncParallelDefaultOrderIsArrival(t *testing.T) {
	var order pipeline.FrameOrder
	if order != pipeline.FrameOrderArrival {
		t.Errorf("zero FrameOrder = %v, want FrameOrderArrival", order)
	}
}

func TestSyncParallelReleasesOutputBeforeEnd(t *testing.T) {
	// A branch that produces output as the EndFrame passes still gets that output
	// out, and the EndFrame itself still reaches the end of the pipeline.
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival,
		[]processor.Processor{newEmitOnEnd("before-end")},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewSyncParallel: %v", err)
	}

	got := runSyncParallel(t, spp, []frames.Frame{frames.NewTextFrame("hi")})

	var sawTagged, sawEnd bool
	for _, f := range got {
		switch fr := f.(type) {
		case *taggedFrame:
			if fr.tag == "before-end" {
				sawTagged = true
			}
		case *frames.EndFrame:
			sawEnd = true
		}
	}
	if !sawTagged {
		t.Error("output produced as the EndFrame passed never left the pipeline")
	}
	if !sawEnd {
		t.Error("EndFrame never reached the end of the pipeline")
	}
}

// emitOnEnd pushes a taggedFrame just before it forwards the EndFrame.
type emitOnEnd struct {
	*processor.Base
	tag string
}

func newEmitOnEnd(tag string) *emitOnEnd {
	e := &emitOnEnd{tag: tag}
	e.Base = processor.New("EmitOnEnd", e)
	return e
}

func (e *emitOnEnd) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.EndFrame); ok {
		if err := e.PushFrame(ctx, newTaggedFrame(e.tag), processor.Downstream); err != nil {
			return err
		}
	}
	return e.PushFrame(ctx, f, dir)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
