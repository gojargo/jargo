package processor

import (
	"context"
	"sync"

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
