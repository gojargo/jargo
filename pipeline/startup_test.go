package pipeline_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// Each processor holds the StartFrame only for its own setup.

var errCannotConnect = errors.New("cannot connect")

// startRecorder takes setup to set up, like a service connecting, and records
// when it gets the StartFrame (since t0), whether it gets a CancelFrame and
// whether it is cleaned up. preSetup is spent before Base.Setup, like a
// processor that connects first.
type startRecorder struct {
	*processor.Base
	t0        time.Time
	preSetup  time.Duration
	setup     time.Duration
	failSetup bool

	mu        sync.Mutex
	startAt   time.Duration
	started   bool
	gotCancel bool
	cleanedUp bool
}

func newStartRecorder(t0 time.Time, setup time.Duration) *startRecorder {
	r := &startRecorder{t0: t0, setup: setup}
	r.Base = processor.New("StartRecorder", r)
	return r
}

func (r *startRecorder) Setup(ctx context.Context, s processor.Setup) error {
	if !sleepCtx(ctx, r.preSetup) {
		return ctx.Err()
	}
	if err := r.Base.Setup(ctx, s); err != nil {
		return err
	}
	if !sleepCtx(ctx, r.setup) {
		return ctx.Err()
	}
	if r.failSetup {
		return errCannotConnect
	}
	return nil
}

func (r *startRecorder) Cleanup(ctx context.Context) error {
	err := r.Base.Cleanup(ctx)
	r.mu.Lock()
	r.cleanedUp = true
	r.mu.Unlock()
	return err
}

func (r *startRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	switch f.(type) {
	case *frames.StartFrame:
		r.startAt, r.started = time.Since(r.t0), true
	case *frames.CancelFrame:
		r.gotCancel = true
	}
	r.mu.Unlock()
	return r.PushFrame(ctx, f, dir)
}

// startedAt is when the recorder got the StartFrame, and whether it did.
func (r *startRecorder) startedAt() (time.Duration, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.startAt, r.started
}

func (r *startRecorder) canceled() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.gotCancel
}

func (r *startRecorder) cleaned() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cleanedUp
}

// sleepCtx sleeps for d, and reports false when ctx ends first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return true
	}
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}

// mustStart reports when the recorder got the StartFrame, failing the test
// when it never did.
func mustStart(t *testing.T, r *startRecorder, name string) time.Duration {
	t.Helper()
	at, ok := r.startedAt()
	if !ok {
		t.Fatalf("%s never got the StartFrame", name)
	}
	return at
}

// runToEnd queues an EndFrame and runs the worker to the end.
func runStartupToEnd(t *testing.T, w *pipeline.Worker) {
	t.Helper()
	w.QueueFrame(frames.NewEndFrame())
	waitDoneWithin(t, runTask(t, w), 5*time.Second)
}

// onStarted records when the pipeline counted as started, since t0.
func onStarted(w *pipeline.Worker, t0 time.Time) func() []time.Duration {
	var mu sync.Mutex
	var at []time.Duration
	events.On(&w.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		mu.Lock()
		at = append(at, time.Since(t0))
		mu.Unlock()
	})
	return func() []time.Duration {
		mu.Lock()
		defer mu.Unlock()
		return append([]time.Duration(nil), at...)
	}
}

// A slow service setup (an LLM connecting, say) must not delay the StartFrame
// for the processors before it. The input transport only starts reading audio
// once it gets the StartFrame, so a caller's first audio would wait on the
// slowest connection.
func TestStartFrameNotHeldByUnrelatedSetup(t *testing.T) {
	const sttSetup, llmSetup = 200 * time.Millisecond, time.Second
	t0 := time.Now()
	input := newStartRecorder(t0, 0)
	stt := newStartRecorder(t0, sttSetup)
	llm := newStartRecorder(t0, llmSetup)
	w := pipeline.NewWorker(pipeline.New(input, stt, llm), pipeline.WorkerConfig{})
	startedAt := onStarted(w, t0)

	runStartupToEnd(t, w)

	// The input transport has nothing to connect, so it starts right away.
	if at := mustStart(t, input, "input"); at >= sttSetup {
		t.Errorf("input started at %s, want before %s", at, sttSetup)
	}
	// Speech-to-text starts once its own connection is up, not the LLM's.
	if at := mustStart(t, stt, "stt"); at < sttSetup || at >= llmSetup {
		t.Errorf("stt started at %s, want between %s and %s", at, sttSetup, llmSetup)
	}
	// The pipeline still only counts as started once everything is set up.
	if at := mustStart(t, llm, "llm"); at < llmSetup {
		t.Errorf("llm started at %s, want after %s", at, llmSetup)
	}
	if got := startedAt(); len(got) != 1 || got[0] < llmSetup {
		t.Errorf("pipeline started at %v, want once, after %s", got, llmSetup)
	}
}

// A parallel pipeline does not hold the StartFrame for its branches.
func TestStartFrameNotHeldBySlowParallelBranch(t *testing.T) {
	t0 := time.Now()
	first := newStartRecorder(t0, 0)
	fast := newStartRecorder(t0, 100*time.Millisecond)
	slow := newStartRecorder(t0, time.Second)
	par, err := pipeline.NewParallel([]processor.Processor{fast}, []processor.Processor{slow})
	if err != nil {
		t.Fatal(err)
	}
	w := pipeline.NewWorker(pipeline.New(first, par), pipeline.WorkerConfig{})

	runStartupToEnd(t, w)

	if at := mustStart(t, first, "first"); at >= 100*time.Millisecond {
		t.Errorf("first started at %s, want before 100ms", at)
	}
	if at := mustStart(t, slow, "slow"); at < time.Second {
		t.Errorf("slow started at %s, want after 1s", at)
	}
}

// A nested pipeline does not hold the StartFrame for its processors.
func TestStartFrameNotHeldBySlowNestedProcessor(t *testing.T) {
	t0 := time.Now()
	first := newStartRecorder(t0, 0)
	innerFast := newStartRecorder(t0, 100*time.Millisecond)
	innerSlow := newStartRecorder(t0, time.Second)
	w := pipeline.NewWorker(pipeline.New(first, pipeline.New(innerFast, innerSlow)), pipeline.WorkerConfig{})

	runStartupToEnd(t, w)

	if at := mustStart(t, first, "first"); at >= 100*time.Millisecond {
		t.Errorf("first started at %s, want before 100ms", at)
	}
	if at := mustStart(t, innerFast, "inner fast"); at < 100*time.Millisecond || at >= time.Second {
		t.Errorf("inner fast started at %s, want between 100ms and 1s", at)
	}
	if at := mustStart(t, innerSlow, "inner slow"); at < time.Second {
		t.Errorf("inner slow started at %s, want after 1s", at)
	}
}

// A processor that fails to set up still lets the StartFrame through, so the
// rest of the pipeline starts.
func TestFailedSetupStillStarts(t *testing.T) {
	t0 := time.Now()
	failing := newStartRecorder(t0, 100*time.Millisecond)
	failing.failSetup = true
	after := newStartRecorder(t0, 0)
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(t0, 0), failing, after), pipeline.WorkerConfig{})
	startedAt := onStarted(w, t0)

	runStartupToEnd(t, w)

	if got := startedAt(); len(got) != 1 {
		t.Errorf("pipeline started %d times, want once", len(got))
	}
	mustStart(t, after, "the processor after the failed one")
}

// Setting up has its own timeout, so a setup that takes longer than the start
// timeout does not make the StartFrame time out.
func TestStartTimeoutCountsFromSetupDone(t *testing.T) {
	t0 := time.Now()
	llm := newStartRecorder(t0, 500*time.Millisecond)
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(t0, 0), llm), pipeline.WorkerConfig{
		StartTimeout: 200 * time.Millisecond,
		SetupTimeout: 5 * time.Second,
	})
	timedOut := onTimeout(w)

	runStartupToEnd(t, w)

	if got := timedOut(); len(got) != 0 {
		t.Errorf("the pipeline timed out on %v, want no timeout", got)
	}
	if at := mustStart(t, llm, "llm"); at < 500*time.Millisecond {
		t.Errorf("llm started at %s, want after 500ms", at)
	}
}

// onTimeout records the frames EventPipelineTimeout reports.
func onTimeout(w *pipeline.Worker) func() []frames.Frame {
	var mu sync.Mutex
	var got []frames.Frame
	events.On(&w.Registry, pipeline.EventPipelineTimeout, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	return func() []frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return append([]frames.Frame(nil), got...)
	}
}

// Canceling does not wait for a processor that is still connecting, as when the
// caller hangs up first.
func TestCancelWhileSettingUp(t *testing.T) {
	t0 := time.Now()
	first := newStartRecorder(t0, 0)
	slow := newStartRecorder(t0, 30*time.Second)
	w := pipeline.NewWorker(pipeline.New(first, slow), pipeline.WorkerConfig{CancelTimeout: time.Second})

	done := runTask(t, w)
	time.Sleep(200 * time.Millisecond)
	w.Cancel(t.Context(), "")
	waitDoneWithin(t, done, 5*time.Second)

	if !w.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
	if took := time.Since(t0); took >= time.Second {
		t.Errorf("the run took %s, want under 1s", took)
	}
	// The canceled setup is abandoned, so that processor never starts. The
	// pipeline never started either, so it is cleaned up, not canceled.
	mustStart(t, first, "first")
	if _, ok := slow.startedAt(); ok {
		t.Error("the processor still setting up got the StartFrame")
	}
	if first.canceled() {
		t.Error("the pipeline that never started got a CancelFrame")
	}
	if !first.cleaned() || !slow.cleaned() {
		t.Errorf("cleaned up: first %t, slow %t, want both", first.cleaned(), slow.cleaned())
	}
}

// A cancel can land at any point while the worker starts, even before its
// processors begin setting up, and the worker still finishes.
func TestCancelAtAnyPointOfStartingFinishes(t *testing.T) {
	for wait := range 25 {
		w := pipeline.NewWorker(pipeline.New(newStartRecorder(time.Now(), 0)), pipeline.WorkerConfig{})
		done := runTask(t, w)
		time.Sleep(time.Duration(wait) * 50 * time.Microsecond)
		w.Cancel(t.Context(), "")
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("the worker hung when canceled after %s", time.Duration(wait)*50*time.Microsecond)
		}
	}
}

// A setup that times out is abandoned: its processor and the ones after it
// never get the StartFrame, and the worker only cleans up.
func TestSetupTimeoutNeverStartsThePipeline(t *testing.T) {
	t0 := time.Now()
	first := newStartRecorder(t0, 0)
	blocked := newStartRecorder(t0, 30*time.Second)
	after := newStartRecorder(t0, 0)
	w := pipeline.NewWorker(pipeline.New(first, blocked, after), pipeline.WorkerConfig{
		SetupTimeout:  200 * time.Millisecond,
		CancelTimeout: 3 * time.Second,
	})

	var mu sync.Mutex
	var seen []string
	record := func(name string) {
		mu.Lock()
		seen = append(seen, name)
		mu.Unlock()
	}
	events.OnSignal(&w.Registry, pipeline.EventSetupTimeout, func(context.Context) { record("setup_timeout") })
	events.On(&w.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		record("pipeline_started")
	})
	events.On(&w.Registry, pipeline.EventPipelineTimeout, func(context.Context, frames.Frame) {
		record("pipeline_timeout")
	})

	waitDoneWithin(t, runTask(t, w), 5*time.Second)

	mu.Lock()
	got := append([]string(nil), seen...)
	mu.Unlock()
	if len(got) != 1 || got[0] != "setup_timeout" {
		t.Errorf("events = %v, want only the setup timeout", got)
	}
	mustStart(t, first, "first")
	if _, ok := blocked.startedAt(); ok {
		t.Error("the processor whose setup timed out got the StartFrame")
	}
	if _, ok := after.startedAt(); ok {
		t.Error("the processor after the one whose setup timed out got the StartFrame")
	}
	if first.canceled() {
		t.Error("the pipeline that never started got a CancelFrame")
	}
	if !first.cleaned() || !blocked.cleaned() || !after.cleaned() {
		t.Errorf("cleaned up: first %t, blocked %t, after %t, want all",
			first.cleaned(), blocked.cleaned(), after.cleaned())
	}
	// Nothing waits for a CancelFrame to drain.
	if took := time.Since(t0); took >= 3*time.Second {
		t.Errorf("the run took %s, want under 3s", took)
	}
}

// A processor that times out before calling Base.Setup has nothing to start
// with, and does not hold up the teardown.
func TestSetupTimeoutBeforeBaseSetup(t *testing.T) {
	t0 := time.Now()
	blocked := newStartRecorder(t0, 0)
	blocked.preSetup = 30 * time.Second
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(t0, 0), blocked, newStartRecorder(t0, 0)),
		pipeline.WorkerConfig{SetupTimeout: 200 * time.Millisecond, CancelTimeout: 3 * time.Second})
	timedOut := onTimeout(w)

	waitDoneWithin(t, runTask(t, w), 5*time.Second)

	if got := timedOut(); len(got) != 0 {
		t.Errorf("the pipeline timed out on %v, want no timeout", got)
	}
	if _, ok := blocked.startedAt(); ok {
		t.Error("the processor that never called Base.Setup got the StartFrame")
	}
	if took := time.Since(t0); took >= 3*time.Second {
		t.Errorf("the run took %s, want under 3s", took)
	}
}

// Setting up is not idle time: the idle timeout only counts once the pipeline
// has started.
func TestIdleTimeoutCountsFromStart(t *testing.T) {
	t0 := time.Now()
	llm := newStartRecorder(t0, 500*time.Millisecond)
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(t0, 0), llm), pipeline.WorkerConfig{
		IdleTimeout: 200 * time.Millisecond,
	})
	var mu sync.Mutex
	var idleAt []time.Duration
	events.OnSignal(&w.Registry, pipeline.EventIdleTimeout, func(context.Context) {
		mu.Lock()
		idleAt = append(idleAt, time.Since(t0))
		mu.Unlock()
	})

	waitDoneWithin(t, runTask(t, w), 5*time.Second)

	at := mustStart(t, llm, "llm")
	mu.Lock()
	defer mu.Unlock()
	if len(idleAt) != 1 || idleAt[0] < at {
		t.Errorf("idle timeout at %v, want once, after the llm started at %s", idleAt, at)
	}
}

// A sync parallel pipeline passes the StartFrame on only once it is through
// all its branches, so the pipeline counts as started only once a slow branch
// is set up.
func TestSyncParallelStartsAfterItsBranches(t *testing.T) {
	t0 := time.Now()
	first := newStartRecorder(t0, 0)
	slow := newStartRecorder(t0, 500*time.Millisecond)
	after := newStartRecorder(t0, 0)
	sync, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival, []processor.Processor{slow})
	if err != nil {
		t.Fatal(err)
	}
	w := pipeline.NewWorker(pipeline.New(first, sync, after), pipeline.WorkerConfig{})
	startedAt := onStarted(w, t0)

	runStartupToEnd(t, w)

	if at := mustStart(t, first, "first"); at >= 500*time.Millisecond {
		t.Errorf("first started at %s, want before 500ms", at)
	}
	slowAt := mustStart(t, slow, "slow")
	if slowAt < 500*time.Millisecond {
		t.Errorf("slow started at %s, want after 500ms", slowAt)
	}
	if at := mustStart(t, after, "after"); at < slowAt {
		t.Errorf("after started at %s, before the slow branch at %s", at, slowAt)
	}
	if got := startedAt(); len(got) != 1 || got[0] < slowAt {
		t.Errorf("pipeline started at %v, want once, after %s", got, slowAt)
	}
}

// slowSyncParallel takes a while before its own setup.
type slowSyncParallel struct {
	*pipeline.SyncParallelPipeline
}

func (p *slowSyncParallel) Setup(ctx context.Context, s processor.Setup) error {
	if !sleepCtx(ctx, 200*time.Millisecond) {
		return ctx.Err()
	}
	return p.SyncParallelPipeline.Setup(ctx, s)
}

// The StartFrame can reach a sync parallel pipeline before its branches begin
// setting up; their processors still hold it until they are set up, so they
// start.
func TestSyncParallelSetUpLateStartsItsBranches(t *testing.T) {
	t0 := time.Now()
	inner := newStartRecorder(t0, 0)
	after := newStartRecorder(t0, 0)
	sp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival, []processor.Processor{inner})
	if err != nil {
		t.Fatal(err)
	}
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(t0, 0), &slowSyncParallel{sp}, after),
		pipeline.WorkerConfig{})

	runStartupToEnd(t, w)

	mustStart(t, inner, "the branch's processor")
	mustStart(t, after, "the processor after the sync parallel pipeline")
}

// A cancel that arrives before the processors are set up still stops a
// processor that never finishes setting up.
func TestCancelBeforeASetupThatNeverFinishes(t *testing.T) {
	blocked := newStartRecorder(time.Now(), time.Hour)
	w := pipeline.NewWorker(pipeline.New(blocked), pipeline.WorkerConfig{SetupTimeout: 100 * time.Millisecond})
	timedOut := onTimeout(w)

	w.Cancel(t.Context(), "")
	waitDoneWithin(t, runTask(t, w), 2*time.Second)

	if !w.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
	if got := timedOut(); len(got) != 0 {
		t.Errorf("the pipeline timed out on %v, want no timeout", got)
	}
}

// setupCanceler cancels its worker from its own Setup, as a processor that
// cannot connect might.
type setupCanceler struct {
	*processor.Base
	worker *pipeline.Worker
}

func (c *setupCanceler) Setup(ctx context.Context, s processor.Setup) error {
	if err := c.Base.Setup(ctx, s); err != nil {
		return err
	}
	c.worker.Cancel(ctx, "cannot connect")
	return nil
}

// A processor can cancel the worker from its own Setup, as when it cannot
// connect.
func TestCancelFromSetup(t *testing.T) {
	canceler := &setupCanceler{}
	canceler.Base = processor.New("SetupCanceler", canceler)
	w := pipeline.NewWorker(pipeline.New(newStartRecorder(time.Now(), 0), canceler), pipeline.WorkerConfig{})
	canceler.worker = w
	timedOut := onTimeout(w)

	waitDoneWithin(t, runTask(t, w), 2*time.Second)

	if !w.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
	if got := timedOut(); len(got) != 0 {
		t.Errorf("the pipeline timed out on %v, want no timeout", got)
	}
}
