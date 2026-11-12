package pipeline_test

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// holdUntilEnd holds text until the EndFrame, then pushes it ahead of the
// EndFrame: what a sentence aggregator does with text that never ends a
// sentence.
type holdUntilEnd struct {
	*processor.Base
	held []frames.Frame
}

func newHoldUntilEnd() *holdUntilEnd {
	h := &holdUntilEnd{}
	h.Base = processor.New("HoldUntilEnd", h)
	return h
}

func (h *holdUntilEnd) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := h.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.TextFrame:
		h.held = append(h.held, frames.NewTextFrame(fr.Text))
		return nil
	case *frames.EndFrame:
		for _, held := range h.held {
			if err := h.PushFrame(ctx, held, dir); err != nil {
				return err
			}
		}
		h.held = nil
	}
	return h.PushFrame(ctx, f, dir)
}

// runSyncParallelPaced is runSyncParallel with a pause after each frame.
func runSyncParallelPaced(t *testing.T, spp *pipeline.SyncParallelPipeline, in []frames.Frame) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(spp), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range in {
		task.QueueFrame(f)
		time.Sleep(100 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("sync parallel task did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return got
}

// outputTypes are the types of the frames that reached the end, the StartFrame
// left out.
func outputTypes(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if _, ok := f.(*frames.StartFrame); ok {
			continue
		}
		out = append(out, fmt.Sprintf("%T", f))
	}
	return out
}

// Ported from upstream (test_end_frame_after_pipeline_output). An EndFrame
// comes out after the output a branch pushes when it ends.
func TestSyncParallelEndFrameAfterBranchOutput(t *testing.T) {
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival, []processor.Processor{newHoldUntilEnd()})
	if err != nil {
		t.Fatal(err)
	}
	got := outputTypes(runSyncParallel(t, spp, []frames.Frame{frames.NewTextFrame("Hello")}))
	if want := []string{"*frames.TextFrame", "*frames.EndFrame"}; !slices.Equal(got, want) {
		t.Errorf("output %v, want %v", got, want)
	}
}

// Ported from upstream (test_end_frame_after_every_pipeline_output). An EndFrame
// comes out once, after the output of every branch, in either frame order.
func TestSyncParallelEndFrameAfterEveryBranchOutput(t *testing.T) {
	for _, order := range []pipeline.FrameOrder{pipeline.FrameOrderArrival, pipeline.FrameOrderPipeline} {
		spp, err := pipeline.NewSyncParallel(order,
			[]processor.Processor{processor.NewIdentityFilter("Identity")},
			[]processor.Processor{newHoldUntilEnd()},
		)
		if err != nil {
			t.Fatal(err)
		}
		got := outputTypes(runSyncParallel(t, spp, []frames.Frame{frames.NewTextFrame("Hello")}))
		if want := []string{"*frames.TextFrame", "*frames.TextFrame", "*frames.EndFrame"}; !slices.Equal(got, want) {
			t.Errorf("order %d: output %v, want %v", order, got, want)
		}
	}
}

// Ported from upstream (test_system_frame_pushed_once). A system frame comes out
// once, though every branch passes it on, in either frame order.
func TestSyncParallelSystemFramePushedOnce(t *testing.T) {
	for _, order := range []pipeline.FrameOrder{pipeline.FrameOrderArrival, pipeline.FrameOrderPipeline} {
		spp, err := pipeline.NewSyncParallel(order,
			[]processor.Processor{processor.NewIdentityFilter("Identity")},
			[]processor.Processor{processor.NewIdentityFilter("Identity")},
		)
		if err != nil {
			t.Fatal(err)
		}
		// Paced as upstream paces it: a system frame queued with data frames
		// jumps ahead of them.
		got := outputTypes(runSyncParallelPaced(t, spp, []frames.Frame{
			frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1),
			frames.NewTextFrame("one"),
			frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1),
		}))
		want := []string{
			"*frames.InputAudioRawFrame", "*frames.TextFrame", "*frames.InputAudioRawFrame", "*frames.EndFrame",
		}
		if !slices.Equal(got, want) {
			t.Errorf("order %d: output %v, want %v", order, got, want)
		}
	}
}
