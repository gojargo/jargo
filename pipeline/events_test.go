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

// Tests for the lifecycle events the Task reports and for the filters selecting
// which frames reaching either end of the pipeline are reported.

// raiser pushes a preloaded frame upstream when it sees a TextFrame, standing in
// for a processor reporting a problem to the task.
type raiser struct {
	*processor.Base
	send frames.Frame
	once sync.Once
}

func newRaiser(send frames.Frame) *raiser {
	r := &raiser{send: send}
	r.Base = processor.New("Raiser", r)
	return r
}

func (r *raiser) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		r.once.Do(func() { _ = r.PushFrame(ctx, r.send, processor.Upstream) })
	}
	return r.PushFrame(ctx, f, dir)
}

// TestPipelineStartedReportsTheStartFrame checks the started handler runs once
// the StartFrame has crossed the whole pipeline, which is what "ready" means.
func TestPipelineStartedReportsTheStartFrame(t *testing.T) {
	var (
		mu    sync.Mutex
		calls int
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		OnPipelineStarted: func(*frames.StartFrame) {
			mu.Lock()
			calls++
			mu.Unlock()
		},
	})

	done := runTask(t, task)
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if calls != 1 {
		t.Errorf("OnPipelineStarted called %d times, want 1", calls)
	}
}

// TestPipelineErrorReportsBothKinds checks every error frame reaching the start
// of the pipeline is reported, and that only a fatal one cancels the run.
//
// A FatalErrorFrame is a distinct type embedding ErrorFrame, so matching the
// concrete type alone would miss it and an unrecoverable failure reported by
// type would go unnoticed.
func TestPipelineErrorReportsBothKinds(t *testing.T) {
	tests := []struct {
		name       string
		send       func() frames.Frame
		wantCancel bool
	}{
		{
			name:       "a non-fatal error is reported and the run continues",
			send:       func() frames.Frame { return frames.NewErrorFrame("just a warning") },
			wantCancel: false,
		},
		{
			name:       "a fatal error cancels the run",
			send:       func() frames.Frame { return frames.NewFatalErrorFrame("unrecoverable") },
			wantCancel: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				mu        sync.Mutex
				reported  []*frames.ErrorFrame
				sawCancel bool
			)
			task := pipeline.NewTask(pipeline.New(newRaiser(tt.send())), pipeline.TaskParams{
				OnPipelineError: func(ef *frames.ErrorFrame) {
					mu.Lock()
					reported = append(reported, ef)
					mu.Unlock()
				},
				OnPipelineFinished: func(f frames.Frame) {
					mu.Lock()
					_, sawCancel = f.(*frames.CancelFrame)
					mu.Unlock()
				},
			})

			done := runTask(t, task)
			task.QueueFrame(frames.NewTextFrame("go"))

			if !tt.wantCancel {
				// Nothing will end the run on its own, so end it here.
				time.Sleep(100 * time.Millisecond)
				task.StopWhenDone()
			}
			waitDone(t, done)

			mu.Lock()
			defer mu.Unlock()
			if len(reported) != 1 {
				t.Fatalf("OnPipelineError called %d times, want 1", len(reported))
			}
			if got := reported[0].Fatal; got != tt.wantCancel {
				t.Errorf("reported error Fatal = %v, want %v", got, tt.wantCancel)
			}
			if sawCancel != tt.wantCancel {
				t.Errorf("run canceled = %v, want %v", sawCancel, tt.wantCancel)
			}
		})
	}
}

// TestReachedFilterSelectsFrames checks the filter decides what the reached
// handler hears, and that an unset filter reports nothing at all.
func TestReachedFilterSelectsFrames(t *testing.T) {
	collect := func(filter pipeline.FrameFilter) []frames.Frame {
		var (
			mu   sync.Mutex
			seen []frames.Frame
		)
		task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
			ReachedDownstreamFilter: filter,
			OnReachedDownstream: func(f frames.Frame) {
				mu.Lock()
				seen = append(seen, f)
				mu.Unlock()
			},
		})
		done := runTask(t, task)
		task.QueueFrame(frames.NewTextFrame("hello"))
		task.StopWhenDone()
		waitDone(t, done)

		mu.Lock()
		defer mu.Unlock()
		return append([]frames.Frame(nil), seen...)
	}

	t.Run("no filter reports nothing", func(t *testing.T) {
		if got := collect(nil); len(got) != 0 {
			t.Errorf("frames reported = %v, want none without a filter", got)
		}
	})

	t.Run("a type filter reports only those types", func(t *testing.T) {
		got := collect(pipeline.FrameTypes(&frames.TextFrame{}))
		if len(got) != 1 {
			t.Fatalf("frames reported = %v, want just the TextFrame", got)
		}
		if tf, ok := got[0].(*frames.TextFrame); !ok || tf.Text != "hello" {
			t.Errorf("frame reported = %v, want the TextFrame carrying hello", got[0])
		}
	})

	t.Run("AnyFrame reports the whole stream", func(t *testing.T) {
		got := collect(pipeline.AnyFrame)
		var sawStart, sawText, sawEnd bool
		for _, f := range got {
			switch f.(type) {
			case *frames.StartFrame:
				sawStart = true
			case *frames.TextFrame:
				sawText = true
			case *frames.EndFrame:
				sawEnd = true
			}
		}
		if !sawStart || !sawText || !sawEnd {
			t.Errorf("frames reported = %v, want the start, the text and the end", got)
		}
	})
}

// TestAddReachedFilterWidensTheSelection checks two consumers of one task can
// each ask for their own frames without knowing about each other.
func TestAddReachedFilterWidensTheSelection(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []frames.Frame
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			seen = append(seen, f)
			mu.Unlock()
		},
	})
	task.AddReachedDownstreamFilter(pipeline.FrameTypes(&frames.EndFrame{}))

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	var sawText, sawEnd, sawStart bool
	for _, f := range seen {
		switch f.(type) {
		case *frames.TextFrame:
			sawText = true
		case *frames.EndFrame:
			sawEnd = true
		case *frames.StartFrame:
			sawStart = true
		}
	}
	if !sawText || !sawEnd {
		t.Errorf("frames reported = %v, want both the text and the end", seen)
	}
	if sawStart {
		t.Errorf("frames reported = %v, want the StartFrame left out", seen)
	}
}
