// Package tonetest runs a tone processor in a pipeline for its tests.
package tonetest

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

// Run runs p alone in a pipeline started with params, sends it each frame in
// order (a time.Duration sleeps that long instead), lets it finish, and returns
// the frames that came out of it downstream and upstream, without the StartFrame
// and EndFrame.
func Run(t *testing.T, p processor.Processor, params pipeline.Params, send ...any) (down, up []frames.Frame) {
	t.Helper()
	return run(t, p, params, processor.Downstream, send)
}

// RunUpstream is Run sending each frame upstream, from the end of the pipeline,
// as a processor after p would.
func RunUpstream(t *testing.T, p processor.Processor, params pipeline.Params, send ...any) (down, up []frames.Frame) {
	t.Helper()
	return run(t, p, params, processor.Upstream, send)
}

func run(
	t *testing.T, p processor.Processor, params pipeline.Params, dir processor.Direction, send []any,
) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	noRTVI := false
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		Params:                  params,
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame:
			return
		}
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range send {
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v, dir)
		default:
			t.Fatalf("tonetest: cannot send %T", s)
		}
	}
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	return down, up
}

// Of returns the frames of type T, in order.
func Of[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}
