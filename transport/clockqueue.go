package transport

import (
	"container/heap"
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// timedFrame is one frame waiting for its presentation timestamp, with the
// sequence it was queued in so frames sharing a timestamp keep their order.
type timedFrame struct {
	pts   time.Duration
	seq   uint64
	frame frames.Frame
}

// timedHeap orders frames by presentation timestamp, then by arrival.
type timedHeap []timedFrame

func (h timedHeap) Len() int { return len(h) }

func (h timedHeap) Less(i, j int) bool {
	if h[i].pts != h[j].pts {
		return h[i].pts < h[j].pts
	}
	return h[i].seq < h[j].seq
}

func (h timedHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

//nolint:forcetypeassert // only timedFrame is ever pushed
func (h *timedHeap) Push(x any) { *h = append(*h, x.(timedFrame)) }

func (h *timedHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// clockQueue holds frames until the pipeline clock reaches the presentation
// timestamp they carry, then pushes them downstream.
//
// It is the second of the output's two paths. Audio goes out through the audio
// queue, paced by playout, and a frame that has to land at a point inside that
// audio (a word as it is spoken, say) goes here instead: its timestamp is a
// wall-clock instant, so it is delivered when the audio around it is heard
// rather than when it happened to be produced. Sending it down the audio path
// would put it wherever the buffering left it.
type clockQueue struct {
	out *BaseOutput

	mu     sync.Mutex
	heap   timedHeap
	seq    uint64
	notify chan struct{}

	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newClockQueue(out *BaseOutput) *clockQueue {
	return &clockQueue{out: out, notify: make(chan struct{}, 1)}
}

// start brings the delivery goroutine up.
func (q *clockQueue) start(ctx context.Context) {
	q.stop()
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	q.cancel = cancel
	q.wg.Add(1)
	go q.run(runCtx)
}

// stop tears the delivery goroutine down and drops what it was holding.
func (q *clockQueue) stop() {
	cancel := q.cancel
	q.cancel = nil
	if cancel == nil {
		return
	}
	cancel()
	q.wg.Wait()
	q.mu.Lock()
	q.heap = nil
	q.mu.Unlock()
}

// drop discards everything waiting, leaving the delivery goroutine running.
func (q *clockQueue) drop() {
	q.mu.Lock()
	q.heap = nil
	q.mu.Unlock()
}

// push queues a frame for delivery at pts, a presentation timestamp in
// nanoseconds on the pipeline clock.
func (q *clockQueue) push(pts int64, f frames.Frame) {
	q.mu.Lock()
	heap.Push(&q.heap, timedFrame{pts: time.Duration(pts), seq: q.seq, frame: f})
	q.seq++
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// next reports the earliest frame waiting, without removing it.
func (q *clockQueue) next() (timedFrame, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.heap) == 0 {
		return timedFrame{}, false
	}
	return q.heap[0], true
}

// take removes the earliest frame waiting, if it is still the one seen.
func (q *clockQueue) take(seq uint64) (timedFrame, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.heap) == 0 || q.heap[0].seq != seq {
		return timedFrame{}, false
	}
	it, ok := heap.Pop(&q.heap).(timedFrame)
	return it, ok
}

// run delivers each frame when the clock reaches its timestamp. A frame whose
// timestamp has already passed goes out at once.
func (q *clockQueue) run(ctx context.Context) {
	defer q.wg.Done()
	timer := time.NewTimer(time.Hour)
	timer.Stop()
	defer timer.Stop()
	for {
		it, ok := q.next()
		if !ok {
			select {
			case <-ctx.Done():
				return
			case <-q.notify:
			}
			continue
		}
		if wait := it.pts - q.now(); wait > 0 {
			timer.Reset(wait)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-q.notify:
				// An earlier frame may have arrived; look again.
				timer.Stop()
				continue
			case <-timer.C:
			}
		}
		if due, taken := q.take(it.seq); taken {
			_ = q.out.PushFrame(ctx, due.frame, processor.Downstream)
		}
	}
}

// now reads the pipeline clock, or zero before the pipeline has one.
func (q *clockQueue) now() time.Duration {
	clk := q.out.Clock()
	if clk == nil {
		return 0
	}
	return clk.Time()
}
