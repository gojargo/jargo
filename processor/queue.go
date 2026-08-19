package processor

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// item is a frame traveling through a processor's internal queues together
// with the direction it is flowing.
type item struct {
	frame frames.Frame
	dir   Direction
}

// queue is an unbounded, concurrency-safe frame queue with a single consumer.
//
// It serves both internal queues of a processor. Frames come off it in three
// tiers: the StartFrame first, then the other system frames, then data and
// control frames. Frames of the same tier keep their arrival order.
//
// The StartFrame outranking everything is what lets a processor be handed
// frames before it has started. Nothing drains the queue until the StartFrame
// arrives, so whatever was pushed at the processor meanwhile waits, and the
// StartFrame is still handled first when the queue finally opens.
//
// Producers never block, which keeps neighboring processors from deadlocking
// when they push to each other.
type queue struct {
	mu     sync.Mutex
	start  []item // the StartFrame, ahead of everything
	system []item // the other system frames
	other  []item // data and control frames
	notify chan struct{}
}

func newQueue() *queue {
	return &queue{notify: make(chan struct{}, 1)}
}

// push appends an item, classifying it by tier so it is served in the right
// order. It never blocks.
func (q *queue) push(it item) {
	q.mu.Lock()
	switch {
	case isStartFrame(it.frame):
		q.start = append(q.start, it)
	case isSystemFrame(it.frame):
		q.system = append(q.system, it)
	default:
		q.other = append(q.other, it)
	}
	q.mu.Unlock()

	// Wake a waiting get. The buffer of one means a signal is never lost: if no
	// one is waiting the pending wake is coalesced and drained on the next get.
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// get returns the next item, blocking until one is available or ctx is done. It
// reports ok=false only when ctx is canceled. The tiers are served in order:
// the StartFrame, then the other system frames, then the rest.
func (q *queue) get(ctx context.Context) (item, bool) {
	for {
		q.mu.Lock()
		switch {
		case len(q.start) > 0:
			it := q.start[0]
			q.start = q.start[1:]
			q.mu.Unlock()
			return it, true
		case len(q.system) > 0:
			it := q.system[0]
			q.system = q.system[1:]
			q.mu.Unlock()
			return it, true
		case len(q.other) > 0:
			it := q.other[0]
			q.other = q.other[1:]
			q.mu.Unlock()
			return it, true
		}
		q.mu.Unlock()

		select {
		case <-ctx.Done():
			return item{}, false
		case <-q.notify:
		}
	}
}

// hasFrame reports whether a queued frame satisfies match. Only the data and
// control frames are considered: a system frame never waits here, it is handled
// on the input goroutine as it arrives.
func (q *queue) hasFrame(match func(frames.Frame) bool) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, it := range q.other {
		if match(it.frame) {
			return true
		}
	}
	return false
}

// reset drops every data and control frame that is not uninterruptible, keeping
// uninterruptible frames so they are still delivered after an interruption.
// System frames are untouched. It is used when an interruption flushes the
// process queue.
func (q *queue) reset() {
	q.mu.Lock()
	defer q.mu.Unlock()
	kept := q.other[:0:0]
	for _, it := range q.other {
		if _, ok := it.frame.(frames.Uninterruptible); ok {
			kept = append(kept, it)
		}
	}
	q.other = kept
}

// isStartFrame reports whether f is the frame that starts a processor.
func isStartFrame(f frames.Frame) bool {
	_, ok := f.(*frames.StartFrame)
	return ok
}
