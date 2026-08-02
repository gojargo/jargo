package transport

import (
	"sync"

	"github.com/gojargo/jargo/frames"
)

// frameQueue holds the frames a sender has yet to pace out. It is a queue rather
// than a channel because an interruption has to drop what will no longer be
// played while keeping the frames that must still be delivered, and a channel
// cannot express that: it can only be drained wholesale.
//
// It counts the uninterruptible frames it holds as they go in and out, so asking
// whether it holds any costs nothing and does not walk the queue.
type frameQueue struct {
	mu    sync.Mutex
	items []frames.Frame
	// uninterruptible is how many of the queued frames must survive an
	// interruption.
	uninterruptible int
	// notify carries one wake-up for a waiting reader.
	notify chan struct{}
}

func newFrameQueue() *frameQueue {
	return &frameQueue{notify: make(chan struct{}, 1)}
}

// push adds a frame to the back of the queue and wakes a waiting reader.
func (q *frameQueue) push(f frames.Frame) {
	q.mu.Lock()
	q.items = append(q.items, f)
	if isUninterruptible(f) {
		q.uninterruptible++
	}
	q.mu.Unlock()

	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// tryGet takes the frame at the front, reporting false when the queue is empty.
func (q *frameQueue) tryGet() (frames.Frame, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	f := q.items[0]
	q.items = q.items[1:]
	if isUninterruptible(f) {
		q.uninterruptible--
	}
	return f, true
}

// wait returns the channel a reader blocks on while the queue is empty.
func (q *frameQueue) wait() <-chan struct{} { return q.notify }

// hasUninterruptible reports whether the queue holds a frame that must survive
// an interruption.
func (q *frameQueue) hasUninterruptible() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.uninterruptible > 0
}

// reset drops every queued frame that is not uninterruptible, keeping the
// uninterruptible ones in the order they were queued so they are still
// delivered after an interruption.
func (q *frameQueue) reset() {
	q.mu.Lock()
	defer q.mu.Unlock()
	kept := q.items[:0:0]
	for _, f := range q.items {
		if isUninterruptible(f) {
			kept = append(kept, f)
		}
	}
	q.items = kept
	q.uninterruptible = len(kept)
}

// isUninterruptible reports whether f must survive an interruption.
func isUninterruptible(f frames.Frame) bool {
	_, ok := f.(frames.Uninterruptible)
	return ok
}
