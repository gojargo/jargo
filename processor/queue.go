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
// It serves both internal queues of a processor. System frames take priority:
// get returns any queued system frame before any data or control frame, so a
// processor always handles system frames first. Producers never block, which
// keeps neighboring processors from deadlocking when they push to each other.
type queue struct {
	mu     sync.Mutex
	system []item // system frames, highest priority
	other  []item // data and control frames
	notify chan struct{}
}

func newQueue() *queue {
	return &queue{notify: make(chan struct{}, 1)}
}

// push appends an item, classifying it by category so system frames are served
// first. It never blocks.
func (q *queue) push(it item) {
	q.mu.Lock()
	if _, ok := it.frame.(frames.SystemFrame); ok {
		q.system = append(q.system, it)
	} else {
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
// reports ok=false only when ctx is canceled. System frames are returned ahead
// of all other frames.
func (q *queue) get(ctx context.Context) (item, bool) {
	for {
		q.mu.Lock()
		if len(q.system) > 0 {
			it := q.system[0]
			q.system = q.system[1:]
			q.mu.Unlock()
			return it, true
		}
		if len(q.other) > 0 {
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
