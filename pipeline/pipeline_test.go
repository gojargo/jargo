package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// echo forwards every frame it receives downstream.
type echo struct {
	*processor.Base
}

func newEcho() *echo {
	e := &echo{}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echo) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return e.PushFrame(ctx, f, dir)
}

func TestTaskEchoEndToEnd(t *testing.T) {
	pipe := pipeline.New(newEcho())

	var mu sync.Mutex
	var texts []string
	params := pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if tf, ok := f.(*frames.TextFrame); ok {
				mu.Lock()
				texts = append(texts, tf.Text)
				mu.Unlock()
			}
		},
	}
	task := pipeline.NewTask(pipe, params)

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(texts) != 1 || texts[0] != "hello" {
		t.Fatalf("frames reaching downstream = %v, want [hello]", texts)
	}
}

func TestTaskCancelStops(t *testing.T) {
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{})

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTextFrame("hi"))
	task.Cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("task did not stop after cancel")
	}
	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
}
