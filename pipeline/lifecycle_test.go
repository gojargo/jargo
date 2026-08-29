package pipeline_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
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
func runTask(t *testing.T, task *pipeline.Worker) chan error {
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
	waitDoneWithin(t, done, 3*time.Second)
}

// waitDoneWithin is waitDone with an explicit deadline, for the tests that wedge
// a processor on purpose and so have to wait out a teardown bound.
func waitDoneWithin(t *testing.T, done chan error, d time.Duration) {
	t.Helper()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(d):
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
			task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
				ReachedDownstreamFilter: pipeline.AnyFrame,
			})
			events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
				if tt.wantEnd(f) {
					mu.Lock()
					saw = true
					mu.Unlock()
				}
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
	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		if cf, ok := f.(*frames.CancelFrame); ok {
			reason = cf.Reason
		}
		if _, ok := f.(*frames.ErrorFrame); ok {
			sawError = true
		}
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
	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
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
	// The frames are through the pipeline, but reporting them is an event and
	// so runs off the frame path; wait for what has been reported to be handled
	// before reading what the handler collected.
	task.Events().Cleanup(ctx)

	mu.Lock()
	got := append([]string(nil), seen...)
	mu.Unlock()
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Errorf("frames processed at flush = %v, want the in-flight frames done", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// TestFlushProbeMakesTheTripTwice checks the probe passes a processor in the
// middle of the pipeline three times: down to the sink, back up to the source,
// and down again. Only the second arrival at the sink settles it.
func TestFlushProbeMakesTheTripTwice(t *testing.T) {
	up := make(chan struct{})
	var once sync.Once
	spy := newFlushSpy()
	task := pipeline.NewWorker(pipeline.New(spy), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(up) })
		}
	})
	done := runTask(t, task)
	started(t, up)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if got := spy.probes(); got != 3 {
		t.Errorf("probe passes = %d, want 3 (down, up, down)", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// TestFlushWaitsForWorkStartedAtTheTurnaround checks the probe waits for work
// that was still on its way down when it turned around at the source.
//
// That is what the extra leg is for: an LLM run triggered by a function call
// result is pushed up and its response only comes back down afterwards. A probe
// that settled at the source would return while the response was still in
// flight.
func TestFlushWaitsForWorkStartedAtTheTurnaround(t *testing.T) {
	up := make(chan struct{})
	var once sync.Once

	// Recorded by a processor at the end of the pipeline rather than by an
	// event, so what the flush is asserted against is what has actually been
	// through rather than what has been reported.
	rec := newTextRecorder()
	task := pipeline.NewWorker(pipeline.New(newTurnaround(), rec), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(up) })
		}
	})
	done := runTask(t, task)
	started(t, up)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	if arrived := slices.Contains(rec.texts(), "late"); !arrived {
		t.Error("the flush settled while work started at the turnaround was still on its way down")
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// textRecorder records the text frames that reach it, in order.
type textRecorder struct {
	*processor.Base
	mu   sync.Mutex
	seen []string
}

func newTextRecorder() *textRecorder {
	p := &textRecorder{}
	p.Base = processor.New("TextRecorder", p)
	return p
}

func (p *textRecorder) ProcessFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok {
		p.mu.Lock()
		p.seen = append(p.seen, tf.Text)
		p.mu.Unlock()
	}
	return p.PushFrame(ctx, f, dir)
}

func (p *textRecorder) texts() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.seen...)
}

// turnaround starts work on its way down as the probe passes it going up, which
// is when a processor answering something pushed upstream starts its own.
type turnaround struct {
	*processor.Base
	once sync.Once
}

func newTurnaround() *turnaround {
	p := &turnaround{}
	p.Base = processor.New("Turnaround", p)
	return p
}

func (p *turnaround) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.PipelineFlushFrame); ok && dir == processor.Upstream {
		p.once.Do(func() {
			_ = p.PushFrame(ctx, frames.NewTextFrame("late"), processor.Downstream)
		})
	}
	return p.PushFrame(ctx, f, dir)
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

// TestFlushWaitsForFramesAlreadyQueued checks the probe does not overtake the
// frames waiting on the worker's own queue.
//
// The probe goes on that queue like any other frame, so everything handed to
// QueueFrame before the call is covered by the flush. Injected straight into the
// pipeline it would jump the queue and report a drain that had not happened.
func TestFlushWaitsForFramesAlreadyQueued(t *testing.T) {
	up := make(chan struct{})
	var once sync.Once

	rec := newTextRecorder()
	task := pipeline.NewWorker(
		pipeline.New(newHoldText(150*time.Millisecond), rec), pipeline.WorkerConfig{
			ReachedDownstreamFilter: pipeline.AnyFrame,
		})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(up) })
		}
	})
	done := runTask(t, task)
	started(t, up)

	task.QueueFrame(frames.NewTextFrame("one"))
	task.QueueFrame(frames.NewTextFrame("two"))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	if got := rec.texts(); len(got) != 2 {
		t.Errorf("frames through by the time the flush settled = %v, want both", got)
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// holdText delays each text frame, so a flush has something slow to wait for.
type holdText struct {
	*processor.Base
	delay time.Duration
}

func newHoldText(d time.Duration) *holdText {
	p := &holdText{delay: d}
	p.Base = processor.New("HoldText", p)
	return p
}

func (p *holdText) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		select {
		case <-time.After(p.delay):
		case <-ctx.Done():
		}
	}
	return p.PushFrame(ctx, f, dir)
}

// TestFlushHonorsContext checks Flush gives up when its context is canceled
// rather than blocking forever on a pipeline that never drains.
func TestFlushHonorsContext(t *testing.T) {
	pipe := pipeline.New(newEcho())
	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{})

	done := runTask(t, task)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := task.Flush(ctx); err == nil {
		t.Error("Flush() = nil, want the canceled context's error")
	}

	task.StopWhenDone()
	waitDone(t, done)
}

// startBlocker holds the StartFrame so it never reaches the sink, standing in
// for a processor wedged during startup.
type startBlocker struct {
	*processor.Base
	reached chan struct{}
	once    sync.Once
	release chan struct{}
}

func newStartBlocker() *startBlocker {
	b := &startBlocker{reached: make(chan struct{}), release: make(chan struct{})}
	b.Base = processor.New("StartBlocker", b)
	return b
}

func (b *startBlocker) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		b.once.Do(func() { close(b.reached) })
		<-b.release
	}
	return b.PushFrame(ctx, f, dir)
}

// TestCancelBeforeStartReachesSink checks canceling while the pipeline is
// still coming up ends the run. The run loop waits for the StartFrame to reach
// the sink before it drains the queue the CancelFrame goes into, so canceling
// has to release that wait or the frame would never be pushed at all.
func TestCancelBeforeStartReachesSink(t *testing.T) {
	blocker := newStartBlocker()
	task := pipeline.NewWorker(pipeline.New(blocker), pipeline.WorkerConfig{
		CancelTimeout: 100 * time.Millisecond,
	})

	timedOut := make(chan frames.Frame, 4)
	events.On(&task.Registry, pipeline.EventPipelineTimeout, func(_ context.Context, f frames.Frame) {
		timedOut <- f
	})

	done := runTask(t, task)

	<-blocker.reached
	task.Cancel(t.Context(), "")

	// The blocker is still wedged inside ProcessFrame, which no context
	// cancellation can reach, so teardown has to wait out its bound before the
	// run can return.
	waitDoneWithin(t, done, 10*time.Second)
	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}

	// The blocked processor never lets the CancelFrame drain, so the worker
	// gives up waiting for it and reports the timeout.
	select {
	case f := <-timedOut:
		if _, ok := f.(*frames.CancelFrame); !ok {
			t.Errorf("the timeout reported %T, want the CancelFrame", f)
		}
	default:
		t.Error("the worker never reported that the cancel frame did not arrive")
	}
	close(blocker.release)
}

// cancelSwallower drops the CancelFrame, so it never reaches the sink.
type cancelSwallower struct {
	*processor.Base
}

func newCancelSwallower() *cancelSwallower {
	c := &cancelSwallower{}
	c.Base = processor.New("CancelSwallower", c)
	return c
}

func (c *cancelSwallower) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.CancelFrame); ok {
		return nil
	}
	return c.PushFrame(ctx, f, dir)
}

// TestCancelTimesOutOnASwallowedFrame checks a CancelFrame that never reaches
// the sink still ends the run, and still reports the pipeline as finished.
// Canceling is what a caller reaches for when something has already gone
// wrong, so it must not be the thing that hangs.
func TestCancelTimesOutOnASwallowedFrame(t *testing.T) {
	startedCh := make(chan struct{})
	var (
		mu       sync.Mutex
		finished frames.Frame
	)
	task := pipeline.NewWorker(pipeline.New(newCancelSwallower()), pipeline.WorkerConfig{
		CancelTimeout:           100 * time.Millisecond,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		finished = f
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			close(startedCh)
		}
	})

	done := runTask(t, task)
	started(t, startedCh)
	task.Cancel(t.Context(), "")

	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if _, ok := finished.(*frames.CancelFrame); !ok {
		t.Errorf("OnPipelineFinished frame = %v, want a CancelFrame", finished)
	}
}

// TestPipelineFinishedReportsTheEndFrame checks the finished handler reports a
// graceful shutdown too, and reports it once.
func TestPipelineFinishedReportsTheEndFrame(t *testing.T) {
	var (
		mu    sync.Mutex
		calls []frames.Frame
	)
	task := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{})
	events.On(&task.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		calls = append(calls, f)
		mu.Unlock()
	})

	done := runTask(t, task)
	task.StopWhenDone()
	waitDone(t, done)

	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 1 {
		t.Fatalf("OnPipelineFinished called %d times, want 1: %v", len(calls), calls)
	}
	if _, ok := calls[0].(*frames.EndFrame); !ok {
		t.Errorf("OnPipelineFinished frame = %v, want an EndFrame", calls[0])
	}
}

// cleanupCounter records whether the pipeline tore it down.
type cleanupCounter struct {
	*processor.Base
	mu      sync.Mutex
	cleaned int
}

func newCleanupCounter() *cleanupCounter {
	c := &cleanupCounter{}
	c.Base = processor.New("CleanupCounter", c)
	return c
}

func (c *cleanupCounter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return c.PushFrame(ctx, f, dir)
}

func (c *cleanupCounter) Cleanup(ctx context.Context) error {
	c.mu.Lock()
	c.cleaned++
	c.mu.Unlock()
	return c.Base.Cleanup(ctx)
}

func (c *cleanupCounter) cleanups() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cleaned
}

// TestStopFrameLeavesTheProcessorsUp checks a StopFrame ends the run without
// tearing the pipeline down, which is the whole difference between it and an
// EndFrame: the processors keep their connections open, ready for another run.
func TestStopFrameLeavesTheProcessorsUp(t *testing.T) {
	tests := []struct {
		name        string
		end         func() frames.Frame
		wantCleanup int
	}{
		{"EndFrame tears the pipeline down", func() frames.Frame { return frames.NewEndFrame() }, 1},
		{"StopFrame leaves it up", func() frames.Frame { return frames.NewStopFrame() }, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter := newCleanupCounter()
			task := pipeline.NewWorker(pipeline.New(counter), pipeline.WorkerConfig{})

			done := runTask(t, task)
			task.QueueFrame(tt.end())
			waitDone(t, done)

			if got := counter.cleanups(); got != tt.wantCleanup {
				t.Errorf("processor cleanups = %d, want %d", got, tt.wantCleanup)
			}
		})
	}
}

// frameRecorder records the frames it handles and whether it was cleaned up, so
// a test can tell an orderly shutdown from processors simply stopping.
type frameRecorder struct {
	*processor.Base
	mu      sync.Mutex
	seen    []frames.Frame
	cleaned bool
}

func newFrameRecorder() *frameRecorder {
	r := &frameRecorder{}
	r.Base = processor.New("FrameRecorder", r)
	return r
}

func (r *frameRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	r.seen = append(r.seen, f)
	r.mu.Unlock()
	return r.PushFrame(ctx, f, dir)
}

func (r *frameRecorder) Cleanup(ctx context.Context) error {
	r.mu.Lock()
	r.cleaned = true
	r.mu.Unlock()
	return r.Base.Cleanup(ctx)
}

func (r *frameRecorder) sawCancel() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, f := range r.seen {
		if _, ok := f.(*frames.CancelFrame); ok {
			return true
		}
	}
	return false
}

func (r *frameRecorder) wasCleaned() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cleaned
}

// TestCancelingTheContextShutsThePipelineDown checks that ending the run by
// canceling its context still takes the pipeline down in order: a CancelFrame
// travels the whole of it, so each processor is told the call is over and can
// close what it had open, rather than being stopped where it stands and left to
// its own timeouts.
func TestCancelingTheContextShutsThePipelineDown(t *testing.T) {
	rec := newFrameRecorder()
	startedCh := make(chan struct{})
	var once sync.Once
	task := pipeline.NewWorker(pipeline.New(rec), pipeline.WorkerConfig{
		CancelTimeout:           time.Second,
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.StartFrame{}),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream,
		func(_ context.Context, _ frames.Frame) { once.Do(func() { close(startedCh) }) })

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- task.Run(ctx) }()

	started(t, startedCh)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Run() = %v, want the context's error reported to the caller", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the run did not finish after its context was canceled")
	}

	if !rec.sawCancel() {
		t.Error("the processor never saw a CancelFrame, so it was stopped rather than told")
	}
	if !rec.wasCleaned() {
		t.Error("the processor was not cleaned up")
	}
	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
}
