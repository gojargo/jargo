package processor

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// event is a manual-reset event. Waiters block until it is set; clearing it
// makes later waiters block again. It is the primitive behind pausing a
// processor, where a loop has to be held before its next frame and released
// again later, possibly several times over a session.
type event struct {
	mu  sync.Mutex
	ch  chan struct{}
	set bool
}

func newEvent() *event { return &event{ch: make(chan struct{})} }

// Set releases every waiter, and lets a later wait pass straight through.
func (e *event) Set() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.set {
		e.set = true
		close(e.ch)
	}
}

// Clear makes a later wait block again.
func (e *event) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.set {
		e.set = false
		e.ch = make(chan struct{})
	}
}

// Wait blocks until the event is set, or ctx is done. It reports whether the
// event was set; false means ctx ended the wait.
func (e *event) Wait(ctx context.Context) bool {
	e.mu.Lock()
	ch, set := e.ch, e.set
	e.mu.Unlock()
	if set {
		return true
	}
	select {
	case <-ch:
		return true
	case <-ctx.Done():
		return false
	}
}

// PauseProcessingFrames holds the processor before it handles its next data or
// control frame. Held frames stay queued, in order, and are handled once
// ResumeProcessingFrames is called. System frames are unaffected; pause those
// with PauseProcessingSystemFrames.
//
// The pause is one-shot: a resume releases the processor, which then keeps
// handling frames until it is paused again.
func (b *Base) PauseProcessingFrames() {
	b.pauseMu.Lock()
	b.blockFrames = true
	ev := b.processEvent
	b.pauseMu.Unlock()
	if ev != nil {
		ev.Clear()
	}
}

// PauseProcessingSystemFrames holds the processor before it handles its next
// system frame. Because the same goroutine feeds the queue that data and control
// frames are handled from, holding it also stops those frames being handed on.
func (b *Base) PauseProcessingSystemFrames() {
	b.pauseMu.Lock()
	b.blockSystemFrames = true
	ev := b.inputEvent
	b.pauseMu.Unlock()
	if ev != nil {
		ev.Clear()
	}
}

// ResumeProcessingFrames releases a processor paused with
// PauseProcessingFrames.
func (b *Base) ResumeProcessingFrames() {
	b.pauseMu.Lock()
	ev := b.processEvent
	b.pauseMu.Unlock()
	if ev != nil {
		ev.Set()
	}
}

// ResumeProcessingSystemFrames releases a processor paused with
// PauseProcessingSystemFrames.
func (b *Base) ResumeProcessingSystemFrames() {
	b.pauseMu.Lock()
	ev := b.inputEvent
	b.pauseMu.Unlock()
	if ev != nil {
		ev.Set()
	}
}

// waitWhilePaused holds a processor loop that has just taken a frame off its
// queue, while the matching pause is in effect. It reports whether the frame may
// now be handled; false means ctx ended the wait and the loop must stop.
func (b *Base) waitWhilePaused(ctx context.Context, systemLoop bool) bool {
	b.pauseMu.Lock()
	blocked, ev := b.blockFrames, b.processEvent
	if systemLoop {
		blocked, ev = b.blockSystemFrames, b.inputEvent
	}
	b.pauseMu.Unlock()

	if !blocked || ev == nil {
		return true
	}
	if !ev.Wait(ctx) {
		return false
	}
	ev.Clear()

	b.pauseMu.Lock()
	if systemLoop {
		b.blockSystemFrames = false
	} else {
		b.blockFrames = false
	}
	b.pauseMu.Unlock()
	return true
}

// pauseIfAddressed pauses this processor when a pause frame names it.
func (b *Base) pauseIfAddressed(target frames.ProcessorTarget) {
	if target != nil && target.Name() == b.name {
		b.PauseProcessingFrames()
	}
}

// resumeIfAddressed arranges for this processor to resume when a resume frame
// names it. The release is held until the frame carrying it has been handled and
// forwarded, so the resume reaches the rest of the pipeline ahead of the frames
// it let through. The two travel on different goroutines here, and without this
// the released frames can overtake the resume that released them.
func (b *Base) resumeIfAddressed(target frames.ProcessorTarget) {
	if target == nil || target.Name() != b.name {
		return
	}
	b.pauseMu.Lock()
	b.pendingResume = true
	b.pauseMu.Unlock()
}

// applyPendingResume releases a processor whose resume frame has now been
// forwarded. It runs after every frame, so a resume is never left pending.
func (b *Base) applyPendingResume() {
	b.pauseMu.Lock()
	pending := b.pendingResume
	b.pendingResume = false
	b.pauseMu.Unlock()
	if pending {
		b.ResumeProcessingFrames()
	}
}

// DefaultPauseUntilReadyTimeout is how long a processor holds frames waiting for
// a readiness condition before giving up and resuming. See
// PauseProcessingAllFramesUntil.
const DefaultPauseUntilReadyTimeout = 5 * time.Second

// PauseProcessingAllFramesUntil holds the frames arriving at this processor
// until ready returns.
//
// It is for a processor that cannot act on frames until some condition holds,
// such as one opening a connection in the background. The frames wait in the
// processor's queues and are handled in order once the condition resolves, so
// nothing is lost.
//
// The frame being handled when this is called is unaffected: the pause takes
// hold from the next frame on. A processor pausing while it handles its
// StartFrame still passes that frame downstream, so starting the pipeline is not
// delayed.
//
// Both queues are held, so a processor left paused could not handle the frames
// that shut it down. The pause is therefore always lifted: when ready returns,
// when timeout elapses, or at cleanup, whichever comes first. A timeout of zero
// takes DefaultPauseUntilReadyTimeout.
//
// ready is called on a goroutine of its own and should return once the processor
// can work, or when ctx ends.
func (b *Base) PauseProcessingAllFramesUntil(ready func(ctx context.Context), timeout time.Duration) {
	if b.directMode {
		slog.Warn("cannot hold frames: this processor runs in direct mode", "processor", b.name)
		return
	}
	if timeout <= 0 {
		timeout = DefaultPauseUntilReadyTimeout
	}

	b.cancelPauseWatcher()
	b.PauseProcessingSystemFrames()
	b.PauseProcessingFrames()

	ctx, cancel := context.WithCancel(b.baseCtx)
	b.pauseMu.Lock()
	b.pauseWatcher = cancel
	b.pauseMu.Unlock()

	b.pauseWG.Go(func() {
		defer cancel()
		b.watchUntilReady(ctx, ready, timeout)
	})
}

// watchUntilReady waits for ready, the timeout or cancellation, then lifts the
// pause whichever it was.
func (b *Base) watchUntilReady(ctx context.Context, ready func(ctx context.Context), timeout time.Duration) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		ready(ctx)
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
		slog.Warn("still not ready, resuming frame processing anyway",
			"processor", b.name, "timeout", timeout)
	case <-ctx.Done():
	}
	b.resumeProcessingAllFrames()
}

// cancelPauseWatcher stops a watcher already running and lifts the pause it was
// going to lift.
func (b *Base) cancelPauseWatcher() {
	b.pauseMu.Lock()
	cancel := b.pauseWatcher
	b.pauseWatcher = nil
	b.pauseMu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	b.pauseWG.Wait()
	b.resumeProcessingAllFrames()
}

// resumeProcessingAllFrames releases both queues.
func (b *Base) resumeProcessingAllFrames() {
	b.ResumeProcessingSystemFrames()
	b.ResumeProcessingFrames()
}
