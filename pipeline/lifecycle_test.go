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

// Tests for the worker frames a processor pushes upstream to ask the Task to end,
// cancel, stop or interrupt the run, and for the flush probe that reports when
// the pipeline has drained.

// upstreamOnce forwards every frame downstream and, the first time it sees a
// TextFrame, pushes a preloaded frame upstream toward the Task. It stands in for
// a processor deep in the pipeline that has no Task handle.
type upstreamOnce struct {
	*processor.Base
	send frames.Frame
	once sync.Once
}

func newUpstreamOnce(send frames.Frame) *upstreamOnce {
	u := &upstreamOnce{send: send}
	u.Base = processor.New("UpstreamOnce", u)
	return u
}

func (u *upstreamOnce) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := u.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		u.once.Do(func() { _ = u.PushFrame(ctx, u.send, processor.Upstream) })
	}
	return u.PushFrame(ctx, f, dir)
}

// runTask starts a task and returns a channel carrying its run error.
func runTask(t *testing.T, task *pipeline.Task) chan error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return done
}

// waitDone fails the test if the task does not finish promptly.
func waitDone(t *testing.T, done chan error) {
	t.Helper()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// TestWorkerFramesEndTheRun checks each worker frame a processor pushes upstream
// drives the Task to the matching pipeline-wide frame, so a processor with no
// Task handle can still end, cancel or stop the run.
func TestWorkerFramesEndTheRun(t *testing.T) {
	tests := []struct {
		name    string
		send    func() frames.Frame
		wantEnd func(frames.Frame) bool
	}{
		{
			name: "EndWorkerFrame yields an EndFrame",
			send: func() frames.Frame { return frames.NewEndWorkerFrame() },
			wantEnd: func(f frames.Frame) bool {
				_, ok := f.(*frames.EndFrame)
				return ok
			},
		},
		{
			name: "CancelWorkerFrame yields a CancelFrame",
			send: func() frames.Frame { return frames.NewCancelWorkerFrame() },
			wantEnd: func(f frames.Frame) bool {
				_, ok := f.(*frames.CancelFrame)
				return ok
			},
		},
		{
			name: "StopWorkerFrame yields a StopFrame",
			send: func() frames.Frame { return frames.NewStopWorkerFrame() },
			wantEnd: func(f frames.Frame) bool {
				_, ok := f.(*frames.StopFrame)
				return ok
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mu sync.Mutex
			var saw bool

			pipe := pipeline.New(newUpstreamOnce(tt.send()))
			task := pipeline.NewTask(pipe, pipeline.TaskParams{
				OnReachedDownstream: func(f frames.Frame) {
					if tt.wantEnd(f) {
						mu.Lock()
						saw = true
						mu.Unlock()
					}
				},
			})

			done := runTask(t, task)
			task.QueueFrame(frames.NewTextFrame("go"))
			waitDone(t, done)

			mu.Lock()
			defer mu.Unlock()
			if !saw {
				t.Error("the worker frame did not produce its pipeline-wide counterpart")
			}
		})
	}
}

// TestCancelWorkerFrameCarriesReason checks a deliberate cancellation keeps its
// reason, so stopping a healthy run is not reported as an error.
func TestCancelWorkerFrameCarriesReason(t *testing.T) {
	var mu sync.Mutex
	var reason string
	var sawError bool

	cancel := frames.NewCancelWorkerFrame()
	cancel.Reason = "caller hung up"

	pipe := pipeline.New(newUpstreamOnce(cancel))
	task := pipeline.NewTask(pipe, pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			if cf, ok := f.(*frames.CancelFrame); ok {
				reason = cf.Reason
			}
			if _, ok := f.(*frames.ErrorFrame); ok {
				sawError = true
			}
		},
	})

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("go"))
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if reason != "caller hung up" {
		t.Errorf("CancelFrame.Reason = %q, want the reason carried from the worker frame", reason)
	}
	if sawError {
		t.Error("a deliberate cancellation should not surface as an ErrorFrame")
	}
}

// TestFlushProbeRoundTrip checks Flush returns only after the frames queued ahead
// of the probe have been processed — the guarantee a caller relies on to let the
// pipeline settle before injecting new work.
func TestFlushProbeRoundTrip(t *testing.T) {
	var mu sync.Mutex
	var seen []string

	pipe := pipeline.New(newEcho())
	task := pipeline.NewTask(pipe, pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if tf, ok := f.(*frames.TextFrame); ok {
				mu.Lock()
				seen = append(seen, tf.Text)
				mu.Unlock()
			}
		},
	})

	done := runTask(t, task)

	task.QueueFrame(frames.NewTextFrame("first"))
	task.QueueFrame(frames.NewTextFrame("second"))

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("Flush() = %v, want the probe to complete its round trip", err)
	}

	mu.Lock()
	got := append([]string(nil), seen...)
	mu.Unlock()
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Errorf("frames processed at flush = %v, want both queued frames done", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// TestFlushHonorsContext checks Flush gives up when its context is canceled
// rather than blocking forever on a pipeline that never drains.
func TestFlushHonorsContext(t *testing.T) {
	pipe := pipeline.New(newEcho())
	task := pipeline.NewTask(pipe, pipeline.TaskParams{})

	done := runTask(t, task)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := task.Flush(ctx); err == nil {
		t.Error("Flush() = nil, want the canceled context's error")
	}

	task.StopWhenDone()
	waitDone(t, done)
}
