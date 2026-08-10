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

// heartbeatEater drops every heartbeat, so none reaches the end of the
// pipeline. It stands in for a processor that has stopped moving frames.
type heartbeatEater struct {
	*processor.Base
}

func newHeartbeatEater() *heartbeatEater {
	h := &heartbeatEater{}
	h.Base = processor.New("HeartbeatEater", h)
	return h
}

func (h *heartbeatEater) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := h.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.HeartbeatFrame); ok {
		return nil
	}
	return h.PushFrame(ctx, f, dir)
}

// TestHeartbeatsCrossThePipeline checks heartbeats are sent once the pipeline is
// up and arrive at the far end, and that they stop when the run does.
func TestHeartbeatsCrossThePipeline(t *testing.T) {
	var (
		mu    sync.Mutex
		beats int
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		EnableHeartbeats:        true,
		HeartbeatPeriod:         20 * time.Millisecond,
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.HeartbeatFrame{}),
		OnReachedDownstream: func(frames.Frame) {
			mu.Lock()
			beats++
			mu.Unlock()
		},
	})

	done := runTask(t, task)
	time.Sleep(150 * time.Millisecond)
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	got := beats
	mu.Unlock()
	if got < 3 {
		t.Errorf("heartbeats reaching the end = %d, want at least 3", got)
	}

	// The run is over, so the pusher must have stopped with it.
	time.Sleep(80 * time.Millisecond)
	mu.Lock()
	after := beats
	mu.Unlock()
	if after != got {
		t.Errorf("heartbeats kept arriving after the run ended: %d then %d", got, after)
	}
}

// TestHeartbeatTimeoutKeepsReporting checks a stalled pipeline is reported over
// and over rather than once: one still stuck a while later is still worth
// hearing about.
func TestHeartbeatTimeoutKeepsReporting(t *testing.T) {
	var (
		mu      sync.Mutex
		reports int
	)
	enough := make(chan struct{})
	var once sync.Once

	task := pipeline.NewTask(pipeline.New(newHeartbeatEater()), pipeline.TaskParams{
		EnableHeartbeats:        true,
		HeartbeatPeriod:         10 * time.Millisecond,
		HeartbeatMonitorTimeout: 40 * time.Millisecond,
		OnHeartbeatTimeout: func() {
			mu.Lock()
			reports++
			done := reports >= 2
			mu.Unlock()
			if done {
				once.Do(func() { close(enough) })
			}
		},
	})

	runDone := runTask(t, task)
	select {
	case <-enough:
	case <-time.After(3 * time.Second):
		mu.Lock()
		got := reports
		mu.Unlock()
		t.Fatalf("heartbeat timeout reported %d times, want it to keep reporting", got)
	}

	task.StopWhenDone()
	waitDone(t, runDone)
}

// TestIdlePipelineIsCanceled checks a pipeline that goes quiet is reported and
// canceled, which is what keeps an abandoned session from running forever.
func TestIdlePipelineIsCanceled(t *testing.T) {
	var (
		mu       sync.Mutex
		reports  int
		finished frames.Frame
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		IdleTimeout: 80 * time.Millisecond,
		OnIdleTimeout: func() {
			mu.Lock()
			reports++
			mu.Unlock()
		},
		OnPipelineFinished: func(f frames.Frame) {
			mu.Lock()
			finished = f
			mu.Unlock()
		},
	})

	done := runTask(t, task)
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if reports != 1 {
		t.Errorf("OnIdleTimeout called %d times, want 1", reports)
	}
	if _, ok := finished.(*frames.CancelFrame); !ok {
		t.Errorf("run ended with %v, want a CancelFrame", finished)
	}
}

// TestIdleTimeoutCanLeaveTheRunAlone checks a caller can hear about the quiet
// and decide for itself, and that it keeps hearing about it.
func TestIdleTimeoutCanLeaveTheRunAlone(t *testing.T) {
	var (
		mu      sync.Mutex
		reports int
	)
	enough := make(chan struct{})
	var once sync.Once

	no := false
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		IdleTimeout:         50 * time.Millisecond,
		CancelOnIdleTimeout: &no,
		OnIdleTimeout: func() {
			mu.Lock()
			reports++
			done := reports >= 2
			mu.Unlock()
			if done {
				once.Do(func() { close(enough) })
			}
		},
	})

	done := runTask(t, task)
	select {
	case <-enough:
	case <-time.After(3 * time.Second):
		t.Fatal("the idle pipeline was reported once and then went unwatched")
	}
	if task.HasFinished() {
		t.Error("the task was canceled despite CancelOnIdleTimeout being false")
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// TestActivityKeepsThePipelineAlive checks the frames that count as activity
// hold the idle timeout off.
func TestActivityKeepsThePipelineAlive(t *testing.T) {
	var (
		mu      sync.Mutex
		reports int
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		IdleTimeout: 100 * time.Millisecond,
		OnIdleTimeout: func() {
			mu.Lock()
			reports++
			mu.Unlock()
		},
	})

	done := runTask(t, task)
	// Speak often enough that the pipeline is never quiet for a whole interval.
	for range 8 {
		task.QueueFrame(&frames.BotSpeakingFrame{
			BaseSystemFrame: frames.NewBaseSystemFrame("BotSpeakingFrame"),
		})
		time.Sleep(30 * time.Millisecond)
	}

	mu.Lock()
	got := reports
	mu.Unlock()
	if got != 0 {
		t.Errorf("OnIdleTimeout called %d times, want 0 while the bot was speaking", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// TestStartMetadataRidesTheStartFrame checks the values a session is stamped
// with reach every processor, which they do by traveling on the StartFrame.
func TestStartMetadataRidesTheStartFrame(t *testing.T) {
	var (
		mu   sync.Mutex
		seen map[string]any
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		StartMetadata:           map[string]any{"call_id": "abc-123"},
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.StartFrame{}),
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			seen = f.Base().Metadata()
			mu.Unlock()
		},
	})

	done := runTask(t, task)
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if seen["call_id"] != "abc-123" {
		t.Errorf("StartFrame metadata = %v, want it carrying call_id", seen)
	}
}

// TestQueueFrameUpstreamEntersAtTheEnd checks a frame can be queued into the far
// end of the pipeline, which is how a caller answers something the pipeline sent
// it rather than starting something new.
func TestQueueFrameUpstreamEntersAtTheEnd(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []string
	)
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		ReachedUpstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
		OnReachedUpstream: func(f frames.Frame) {
			tf, ok := f.(*frames.TextFrame)
			if !ok {
				return
			}
			mu.Lock()
			seen = append(seen, tf.Text)
			mu.Unlock()
		},
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.StartFrame{}),
		OnReachedDownstream:     func(frames.Frame) {},
	})

	done := runTask(t, task)
	// Give the pipeline a moment to come up: a frame pushed before the
	// StartFrame has crossed is dropped.
	time.Sleep(100 * time.Millisecond)
	task.QueueFrames([]frames.Frame{
		frames.NewTextFrame("first"),
		frames.NewTextFrame("second"),
	}, processor.Upstream)
	time.Sleep(100 * time.Millisecond)

	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 || seen[0] != "first" || seen[1] != "second" {
		t.Errorf("frames reaching the start = %v, want [first second]", seen)
	}
}

// TestTurnTrackingRunsWithoutTracing checks the turns are followed in a session
// that is not traced, since where the turns fell is worth knowing either way.
func TestTurnTrackingRunsWithoutTracing(t *testing.T) {
	task := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{})
	if task.TurnTracking() == nil {
		t.Error("TurnTracking() = nil, want the turns followed in an untraced task")
	}
	if task.TurnTrace() != nil {
		t.Error("TurnTrace() is set on a task that does not trace")
	}

	off := false
	quiet := pipeline.NewTask(pipeline.New(newEcho()), pipeline.TaskParams{
		EnableTurnTracking: &off,
	})
	if quiet.TurnTracking() != nil {
		t.Error("TurnTracking() is set on a task that turned it off")
	}
}
