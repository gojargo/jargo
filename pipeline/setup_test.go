package pipeline_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

//nolint:gochecknoglobals // sentinel errors for the tests below
var (
	errFirstSetup    = errors.New("first failed")
	errSecondSetup   = errors.New("second failed")
	errCannotCleanUp = errors.New("cannot clean up")
)

// failingSetup fails while it is set up, the way a service whose connection
// attempt is refused does.
type failingSetup struct {
	*processor.Base
	reason error
}

func newFailingSetup(name string, reason error) *failingSetup {
	f := &failingSetup{reason: reason}
	f.Base = processor.New(name, f)
	return f
}

func (f *failingSetup) Setup(ctx context.Context, s processor.Setup) error {
	if err := f.Base.Setup(ctx, s); err != nil {
		return err
	}
	return f.reason
}

func (f *failingSetup) ProcessFrame(ctx context.Context, fr frames.Frame, dir processor.Direction) error {
	if err := f.Base.ProcessFrame(ctx, fr, dir); err != nil {
		return err
	}
	return f.PushFrame(ctx, fr, dir)
}

// errorWatcher records every error frame that passes it.
type errorWatcher struct {
	*processor.Base
	mu   sync.Mutex
	seen []string
}

func newErrorWatcher() *errorWatcher {
	w := &errorWatcher{}
	w.Base = processor.New("ErrorWatcher", w)
	return w
}

func (w *errorWatcher) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := w.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if ef, ok := f.(*frames.ErrorFrame); ok {
		w.mu.Lock()
		w.seen = append(w.seen, ef.Error)
		w.mu.Unlock()
	}
	return w.PushFrame(ctx, f, dir)
}

func (w *errorWatcher) errors() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.seen...)
}

// Ported from upstream. A processor that fails to set up reports it as an error
// frame. Services connect during setup, so a failure there is as worth
// reporting as one while handling a frame. Each failing processor reports its
// own, so one failure never hides another.
func TestSetupFailuresAreReported(t *testing.T) {
	first := newFailingSetup("First", errFirstSetup)
	second := newFailingSetup("Second", errSecondSetup)
	watcher := newErrorWatcher()

	task := pipeline.NewWorker(pipeline.New(watcher, first, second), pipeline.WorkerConfig{})
	done := runTask(t, task)
	task.QueueFrame(frames.NewEndFrame())
	waitDoneWithin(t, done, 5*time.Second)

	got := watcher.errors()
	if len(got) != 2 {
		t.Fatalf("saw %d error frames, want both setup failures: %q", len(got), got)
	}
	var sawFirst, sawSecond bool
	for _, msg := range got {
		if strings.Contains(msg, "first failed") {
			sawFirst = true
		}
		if strings.Contains(msg, "second failed") {
			sawSecond = true
		}
	}
	if !sawFirst || !sawSecond {
		t.Fatalf("errors = %q, want one naming each failure", got)
	}

	// Setting up is not attempted again, so a processor that failed it can no
	// longer do its job and a switcher can move off it.
	if first.Usable() {
		t.Error("the first processor is still usable after failing to set up")
	}
	if second.Usable() {
		t.Error("the second processor is still usable after failing to set up")
	}
}

// setupBlocker never finishes being set up, the way a service that never
// connects does not.
type setupBlocker struct {
	*processor.Base
	release chan struct{}
}

func newSetupBlocker() *setupBlocker {
	b := &setupBlocker{release: make(chan struct{})}
	b.Base = processor.New("SetupBlocker", b)
	return b
}

func (b *setupBlocker) Setup(ctx context.Context, s processor.Setup) error {
	if err := b.Base.Setup(ctx, s); err != nil {
		return err
	}
	<-b.release
	return nil
}

// cleanupFailer fails to clean up.
type cleanupFailer struct {
	*processor.Base
}

func newCleanupFailer() *cleanupFailer {
	c := &cleanupFailer{}
	c.Base = processor.New("CleanupFailer", c)
	return c
}

func (c *cleanupFailer) Cleanup(ctx context.Context) error {
	_ = c.Base.Cleanup(ctx)
	return errCannotCleanUp
}

// cleanupRecorder records that it was cleaned up.
type cleanupRecorder struct {
	*processor.Base
	mu      sync.Mutex
	cleaned bool
}

func newCleanupRecorder() *cleanupRecorder {
	c := &cleanupRecorder{}
	c.Base = processor.New("CleanupRecorder", c)
	return c
}

func (c *cleanupRecorder) Cleanup(ctx context.Context) error {
	_ = c.Base.Cleanup(ctx)
	c.mu.Lock()
	c.cleaned = true
	c.mu.Unlock()
	return nil
}

func (c *cleanupRecorder) wasCleaned() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cleaned
}

// Ported from upstream. Processors connect while they are set up, so one that
// never connects would otherwise leave the run waiting on it with nothing to
// time it out. Nothing else tells the application its pipeline never came up.
func TestSetupNeverFinishes(t *testing.T) {
	blocker := newSetupBlocker()
	defer close(blocker.release)

	task := pipeline.NewWorker(pipeline.New(blocker), pipeline.WorkerConfig{
		SetupTimeout:  100 * time.Millisecond,
		CancelTimeout: 100 * time.Millisecond,
	})

	var timeouts atomic.Int64
	events.OnSignal(&task.Registry, pipeline.EventSetupTimeout, func(context.Context) {
		timeouts.Add(1)
	})

	done := runTask(t, task)
	waitDoneWithin(t, done, 5*time.Second)

	if !task.HasFinished() {
		t.Error("the run returned without the worker reporting it had finished")
	}
	if got := timeouts.Load(); got != 1 {
		t.Errorf("the setup timeout was reported %d times, want once", got)
	}
}

// Ported from upstream. Setting up is abandoned part way, so processors are
// cleaned up from states they never finished reaching, and a cleanup that fails
// must not cost the others theirs.
func TestSetupTimeoutStillCleansTheRestUp(t *testing.T) {
	blocker := newSetupBlocker()
	defer close(blocker.release)
	recorder := newCleanupRecorder()

	task := pipeline.NewWorker(
		pipeline.New(newCleanupFailer(), blocker, recorder),
		pipeline.WorkerConfig{
			SetupTimeout:  100 * time.Millisecond,
			CancelTimeout: 100 * time.Millisecond,
		},
	)

	done := runTask(t, task)
	waitDoneWithin(t, done, 5*time.Second)

	if !recorder.wasCleaned() {
		t.Error("a processor was left uncleaned by another's failing cleanup")
	}
}

// startHolder holds the StartFrame and never passes it on, so the frame never
// reaches the end of the pipeline.
type startHolder struct {
	*processor.Base
	release chan struct{}
}

func newStartHolder() *startHolder {
	h := &startHolder{release: make(chan struct{})}
	h.Base = processor.New("StartHolder", h)
	return h
}

func (h *startHolder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := h.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		<-h.release
	}
	return h.PushFrame(ctx, f, dir)
}

// Ported from upstream. A processor that blocks while handling the StartFrame
// leaves the pipeline half-started, and the worker would wait on it forever.
// Nothing else tells the application its pipeline never came up.
func TestStartFrameNeverReachesSink(t *testing.T) {
	holder := newStartHolder()
	defer close(holder.release)

	task := pipeline.NewWorker(pipeline.New(holder), pipeline.WorkerConfig{
		StartTimeout:  100 * time.Millisecond,
		CancelTimeout: 100 * time.Millisecond,
	})

	timedOut := make(chan frames.Frame, 4)
	events.On(&task.Registry, pipeline.EventPipelineTimeout, func(_ context.Context, f frames.Frame) {
		timedOut <- f
	})

	done := runTask(t, task)
	waitDoneWithin(t, done, 5*time.Second)

	if !task.HasFinished() {
		t.Error("the run returned without the worker reporting it had finished")
	}
	select {
	case f := <-timedOut:
		if _, ok := f.(*frames.StartFrame); !ok {
			t.Errorf("the timeout reported %T, want the StartFrame", f)
		}
	default:
		t.Fatal("the pipeline never reported that it failed to start")
	}
	if len(timedOut) != 0 {
		t.Errorf("the timeout was reported %d times, want once", len(timedOut)+1)
	}
}
