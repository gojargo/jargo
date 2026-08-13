package pipeline_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{})
	events.On(&task.Registry, pipeline.EventPipelineStarted, func(_ context.Context, _ *frames.StartFrame) {
		mu.Lock()
		calls++
		mu.Unlock()
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
			task := pipeline.NewWorker(pipeline.New(newRaiser(tt.send())), pipeline.WorkerConfig{})
			events.On(&task.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
				mu.Lock()
				reported = append(reported, ef)
				mu.Unlock()
			})
			events.On(&task.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
				mu.Lock()
				_, sawCancel = f.(*frames.CancelFrame)
				mu.Unlock()
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
		task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
			ReachedDownstreamFilter: filter,
		})
		events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
			mu.Lock()
			seen = append(seen, f)
			mu.Unlock()
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		seen = append(seen, f)
		mu.Unlock()
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.HeartbeatFrame{}),
		Params: pipeline.Params{
			EnableHeartbeats: true,
			HeartbeatPeriod:  20 * time.Millisecond,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, _ frames.Frame) {
		mu.Lock()
		beats++
		mu.Unlock()
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

	task := pipeline.NewWorker(pipeline.New(newHeartbeatEater()), pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableHeartbeats:        true,
			HeartbeatPeriod:         10 * time.Millisecond,
			HeartbeatMonitorTimeout: 40 * time.Millisecond,
		},
	})
	events.OnSignal(&task.Registry, pipeline.EventHeartbeatTimeout, func(context.Context) {
		mu.Lock()
		reports++
		done := reports >= 2
		mu.Unlock()
		if done {
			once.Do(func() { close(enough) })
		}
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		IdleTimeout: 80 * time.Millisecond,
	})
	events.OnSignal(&task.Registry, pipeline.EventIdleTimeout, func(context.Context) {
		mu.Lock()
		reports++
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		finished = f
		mu.Unlock()
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		IdleTimeout:         50 * time.Millisecond,
		CancelOnIdleTimeout: &no,
	})
	events.OnSignal(&task.Registry, pipeline.EventIdleTimeout, func(context.Context) {
		mu.Lock()
		reports++
		done := reports >= 2
		mu.Unlock()
		if done {
			once.Do(func() { close(enough) })
		}
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		IdleTimeout: 100 * time.Millisecond,
	})
	events.OnSignal(&task.Registry, pipeline.EventIdleTimeout, func(context.Context) {
		mu.Lock()
		reports++
		mu.Unlock()
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.StartFrame{}),
		Params: pipeline.Params{
			StartMetadata: map[string]any{"call_id": "abc-123"},
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		seen = f.Base().Metadata()
		mu.Unlock()
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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedUpstreamFilter:   pipeline.FrameTypes(&frames.TextFrame{}),
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.StartFrame{}),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		tf, ok := f.(*frames.TextFrame)
		if !ok {
			return
		}
		mu.Lock()
		seen = append(seen, tf.Text)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, _ frames.Frame) {})

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
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{})
	if task.TurnTracking() == nil {
		t.Error("TurnTracking() = nil, want the turns followed in an untraced task")
	}
	if task.TurnTrace() != nil {
		t.Error("TurnTrace() is set on a task that does not trace")
	}

	off := false
	quiet := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		EnableTurnTracking: &off,
	})
	if quiet.TurnTracking() != nil {
		t.Error("TurnTracking() is set on a task that turned it off")
	}
}

// slowObserver takes its time over every frame, standing in for one that logs
// to disk or ships events over the network.
type slowObserver struct {
	delay time.Duration
	mu    sync.Mutex
	seen  int
}

func (o *slowObserver) OnPushFrame(processor.FramePushed) {
	time.Sleep(o.delay)
	o.mu.Lock()
	o.seen++
	o.mu.Unlock()
}

func (o *slowObserver) count() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.seen
}

// TestASlowObserverDoesNotHoldUpThePipeline checks watching the pipeline does
// not change how it runs. An observer is reported to for every handover, so one
// that is slow would otherwise put its own delay between every pair of
// processors.
func TestASlowObserverDoesNotHoldUpThePipeline(t *testing.T) {
	slow := &slowObserver{delay: 50 * time.Millisecond}

	arrived := make(chan struct{})
	var once sync.Once
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{slow},
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, _ frames.Frame) {
		once.Do(func() { close(arrived) })
	})

	done := runTask(t, task)
	start := time.Now()
	task.QueueFrame(frames.NewTextFrame("hello"))

	select {
	case <-arrived:
	case <-time.After(3 * time.Second):
		t.Fatal("the frame never crossed the pipeline")
	}

	// The frame crosses several handovers. Reported inline, the slow observer
	// would add its delay to each of them.
	if elapsed := time.Since(start); elapsed > 40*time.Millisecond {
		t.Errorf("the frame took %s to cross, want the slow observer kept off the path", elapsed)
	}

	// Kept off the frame path, but still reported to: it works through its queue
	// on its own time while the pipeline runs.
	waitFor(t, func() bool { return slow.count() > 0 }, "the slow observer to be reported to")

	task.StopWhenDone()
	waitDone(t, done)
}

// stalledObserver holds up its first report until it is released, so everything
// reported meanwhile piles up behind it.
type stalledObserver struct {
	release chan struct{}
	once    sync.Once
	mu      sync.Mutex
	seen    int
}

func newStalledObserver() *stalledObserver {
	return &stalledObserver{release: make(chan struct{})}
}

func (o *stalledObserver) OnPushFrame(processor.FramePushed) {
	<-o.release
	o.mu.Lock()
	o.seen++
	o.mu.Unlock()
}

func (o *stalledObserver) let() { o.once.Do(func() { close(o.release) }) }

func (o *stalledObserver) count() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.seen
}

// TestAnObserverFarBehindLosesNothing checks nothing waiting for an observer is
// dropped while the pipeline runs, however far behind it falls, so watching
// slowly shows the same run as watching quickly. The stateful observers are
// counting, and one that lost the start of a turn or the close of a tool call
// would report a conversation that never happened.
func TestAnObserverFarBehindLosesNothing(t *testing.T) {
	stalled := newStalledObserver()
	keeping := &slowObserver{}

	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{stalled, keeping},
	})

	done := runTask(t, task)
	// Enough frames that a queue with any bound on it would have to drop some.
	const sent = 600
	for i := range sent {
		task.QueueFrame(frames.NewTextFrame(fmt.Sprintf("frame %d", i)))
	}

	// Released while the run is still going, so it catches up on the reports
	// that piled up rather than on whatever survived the end of the run.
	stalled.let()
	waitFor(t, func() bool {
		kept := keeping.count()
		return kept > sent && stalled.count() == kept
	}, "the observer that fell behind to catch up with the one that kept up")

	task.StopWhenDone()
	waitDone(t, done)
}

// TestAddObserverWhileRunning checks something built after the task can still
// watch the frames going by.
func TestAddObserverWhileRunning(t *testing.T) {
	late := &slowObserver{}
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{})

	done := runTask(t, task)
	task.AddObserver(late)
	task.QueueFrame(frames.NewTextFrame("hello"))

	waitFor(t, func() bool { return late.count() > 0 },
		"the observer added while running to be reported to")

	task.StopWhenDone()
	waitDone(t, done)
}

// TestRemoveObserverStopsReporting checks an observer dropped while the pipeline
// runs hears nothing more, so a caller may release what it holds.
func TestRemoveObserverStopsReporting(t *testing.T) {
	watching := &slowObserver{}
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{watching},
	})

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("before"))
	waitFor(t, func() bool { return watching.count() > 0 }, "the observer to be reported to")

	task.RemoveObserver(watching)
	settled := watching.count()

	task.QueueFrame(frames.NewTextFrame("after"))
	task.StopWhenDone()
	waitDone(t, done)

	if got := watching.count(); got != settled {
		t.Errorf("the removed observer saw %d reports, want the %d it had already", got, settled)
	}
}

// TestRemoveObserverAfterTheRun checks dropping an observer once the run is over
// is harmless, so a caller tidying up does not have to know whether the pipeline
// stopped first.
func TestRemoveObserverAfterTheRun(t *testing.T) {
	watching := &slowObserver{}
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{watching},
	})

	done := runTask(t, task)
	task.StopWhenDone()
	waitDone(t, done)

	task.RemoveObserver(watching)
	task.RemoveObserver(watching)
}

// startedObserver records the pipeline having started, and whether anything
// queued for the conversation was reported before it.
type startedObserver struct {
	mu     sync.Mutex
	starts int
	texts  int
	early  int
}

func (o *startedObserver) OnPushFrame(data processor.FramePushed) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if _, ok := data.Frame.(*frames.TextFrame); ok {
		o.texts++
		if o.starts == 0 {
			o.early++
		}
	}
}

func (o *startedObserver) OnPipelineStarted() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.starts++
}

func (o *startedObserver) counts() (starts, texts, early int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.starts, o.texts, o.early
}

// TestPipelineStartedReachesObservers checks an observer hears that the pipeline
// has started, once, and in order with the frames: it arrives before anything
// the conversation queued behind the StartFrame.
func TestPipelineStartedReachesObservers(t *testing.T) {
	watching := &startedObserver{}
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{watching},
	})

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("hello"))
	waitFor(t, func() bool {
		_, texts, _ := watching.counts()
		return texts > 0
	}, "the text frame to be reported")

	task.StopWhenDone()
	waitDone(t, done)

	starts, _, early := watching.counts()
	if starts != 1 {
		t.Errorf("the pipeline started %d times, want 1", starts)
	}
	if early != 0 {
		t.Errorf("%d frames were reported before the pipeline started", early)
	}
}

// TestOnReachedDownstreamRegistersLateHandlers checks handlers can be added
// after the task is built. Something that queues frames needs the task to queue
// them on, so it cannot also have been passed to NewTask.
func TestOnReachedDownstreamRegistersLateHandlers(t *testing.T) {
	var (
		mu    sync.Mutex
		first []frames.Frame
		later []frames.Frame
	)
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		first = append(first, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		later = append(later, f)
		mu.Unlock()
	})

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if len(first) != 1 || len(later) != 1 {
		t.Fatalf("handlers saw %d and %d frames, want one each", len(first), len(later))
	}
	if first[0] != later[0] {
		t.Error("the two handlers were given different frames")
	}
}

// TestSetReachedFilterReplacesTheSelection checks a filter set after the task
// was built replaces the one it was built with, rather than widening it.
func TestSetReachedFilterReplacesTheSelection(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []frames.Frame
	)
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		seen = append(seen, f)
		mu.Unlock()
	})
	task.SetReachedDownstreamFilter(pipeline.FrameTypes(&frames.TextFrame{}))

	done := runTask(t, task)
	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 {
		t.Fatalf("frames reported = %v, want only the text frame", seen)
	}
	if _, ok := seen[0].(*frames.TextFrame); !ok {
		t.Errorf("frames reported = %v, want only the text frame", seen)
	}
}

// TestUpstreamReachedFiltersSelectWhatIsReported checks the same two controls on
// the upstream side, where a processor deep in the pipeline reports back to the
// task.
func TestUpstreamReachedFiltersSelectWhatIsReported(t *testing.T) {
	tests := []struct {
		name  string
		apply func(*pipeline.Worker)
		want  bool
	}{
		{
			name:  "a filter that selects the frame",
			apply: func(task *pipeline.Worker) { task.SetReachedUpstreamFilter(pipeline.AnyFrame) },
			want:  true,
		},
		{
			name: "a filter that does not",
			apply: func(task *pipeline.Worker) {
				task.SetReachedUpstreamFilter(pipeline.FrameTypes(&frames.TextFrame{}))
			},
		},
		{
			name: "a filter widened to select it",
			apply: func(task *pipeline.Worker) {
				task.SetReachedUpstreamFilter(pipeline.FrameTypes(&frames.TextFrame{}))
				task.AddReachedUpstreamFilter(pipeline.FrameTypes(&frames.ErrorFrame{}))
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				mu      sync.Mutex
				sawErr  bool
				errFrom = frames.NewErrorFrame("upstream report")
			)
			task := pipeline.NewWorker(pipeline.New(newUpstreamOnce(errFrom)), pipeline.WorkerConfig{})
			events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
				if _, ok := f.(*frames.ErrorFrame); ok {
					mu.Lock()
					sawErr = true
					mu.Unlock()
				}
			})
			tt.apply(task)

			done := runTask(t, task)
			task.QueueFrame(frames.NewTextFrame("go"))
			task.StopWhenDone()
			waitDone(t, done)

			mu.Lock()
			defer mu.Unlock()
			if sawErr != tt.want {
				t.Errorf("the error frame was reported = %v, want %v", sawErr, tt.want)
			}
		})
	}
}
