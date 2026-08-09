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

// started blocks until the StartFrame has traveled the whole pipeline, so a test
// injecting a frame directly does not race the pipeline coming up. Pair it with
// a TaskParams whose OnReachedDownstream closes ch on the StartFrame.
func started(t *testing.T, ch chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline never started")
	}
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

// pacedEcho forwards frames after a short pause, so a test can be sure work is
// still in flight in the pipeline when it injects something behind it.
type pacedEcho struct {
	*processor.Base
}

func newPacedEcho() *pacedEcho {
	p := &pacedEcho{}
	p.Base = processor.New("PacedEcho", p)
	return p
}

func (p *pacedEcho) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		time.Sleep(100 * time.Millisecond)
	}
	return p.PushFrame(ctx, f, dir)
}

// TestFlushProbeRoundTrip checks Flush returns only after the work already in
// the pipeline ahead of the probe has been processed, which is the guarantee a
// caller relies on to let the pipeline settle before injecting new work.
//
// The wait for the first frame is what makes it deterministic: once that has
// come out the far end, the second is inside the pipeline, so the probe injected
// behind it has to wait for it.
func TestFlushProbeRoundTrip(t *testing.T) {
	var mu sync.Mutex
	var seen []string

	up := make(chan struct{})
	first := make(chan struct{})
	var startOnce, firstOnce sync.Once
	pipe := pipeline.New(newPacedEcho())
	task := pipeline.NewTask(pipe, pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.StartFrame); ok {
				startOnce.Do(func() { close(up) })
			}
			if tf, ok := f.(*frames.TextFrame); ok {
				mu.Lock()
				seen = append(seen, tf.Text)
				mu.Unlock()
				if tf.Text == "first" {
					firstOnce.Do(func() { close(first) })
				}
			}
		},
	})

	done := runTask(t, task)
	started(t, up)

	task.QueueFrame(frames.NewTextFrame("first"))
	task.QueueFrame(frames.NewTextFrame("second"))
	started(t, first)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("Flush() = %v, want the probe to complete its round trip", err)
	}

	mu.Lock()
	got := append([]string(nil), seen...)
	mu.Unlock()
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Errorf("frames processed at flush = %v, want the in-flight frames done", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// flushSpy records the flush probes that reach it, so a test can tell whether a
// probe entered the pipeline at all.
type flushSpy struct {
	*processor.Base
	mu   sync.Mutex
	seen int
}

func newFlushSpy() *flushSpy {
	p := &flushSpy{}
	p.Base = processor.New("FlushSpy", p)
	return p
}

func (p *flushSpy) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.PipelineFlushFrame); ok {
		p.mu.Lock()
		p.seen++
		p.mu.Unlock()
	}
	return p.PushFrame(ctx, f, dir)
}

func (p *flushSpy) probes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.seen
}

// TestFlushAfterPipelineEndQueued checks a flush probe still reaches the
// pipeline once a pipeline-ending frame has been queued ahead of it.
//
// The task stops draining the push queue as soon as one goes in, since it then
// waits for that frame to travel through. A probe put through the queue behind
// it would never enter the pipeline at all, and the caller would wait out its
// whole timeout for a pipeline that was never asked anything. A tool handler
// that ends the session leaves exactly that behind.
//
// It asserts the probe arrived rather than that the flush settled: once the end
// frame is released the pipeline tears down, so whether the probe finishes its
// round trip before that is a race by nature. Entering at all is the part the
// caller was being denied.
func TestFlushAfterPipelineEndQueued(t *testing.T) {
	up := make(chan struct{})
	var once sync.Once
	spy := newFlushSpy()
	task := pipeline.NewTask(pipeline.New(spy, newSlowEnd(300*time.Millisecond)), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.StartFrame); ok {
				once.Do(func() { close(up) })
			}
		},
	})
	done := runTask(t, task)
	started(t, up)

	task.StopWhenDone()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = task.Flush(ctx)

	if spy.probes() == 0 {
		t.Error("the flush probe never entered the pipeline behind the end frame")
	}
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
