package pipeline

import (
	"sync"

	"github.com/gojargo/jargo/processor"
)

// observerProxy stands in for the observers a task runs with. It is the one
// observer handed to the processors; every frame reported to it is passed on to
// each of the real observers.
//
// The point of it is that watching does not slow the pipeline down. Observers
// are reported to on the goroutine the frame is traveling on, so anything one of
// them does slowly is done in the middle of the conversation. Each observer here
// gets a queue and a goroutine of its own instead, so a slow one falls behind on
// its own rather than holding up the frames or the other observers.
//
// Delivery to a single observer stays in order, which is what the stateful ones
// need: a turn cannot end before it starts.
type observerProxy struct {
	mu        sync.Mutex
	observers []*observerWorker
	started   bool
	stopped   bool
}

// observerWorker is one observer with the queue of reports waiting for it.
//
// Each has a quit of its own rather than sharing the proxy's, so a single
// observer can be dropped while the pipeline runs without disturbing the rest.
type observerWorker struct {
	observer processor.Observer
	queue    *reportQueue
	quit     chan struct{}
	done     chan struct{}
	// stopping makes cancel idempotent, so an observer removed after the
	// pipeline stopped is not canceled twice.
	stopping sync.Once
}

// pipelineStarted is what a worker queues to itself to report the pipeline
// having started, so the report reaches the observer in order with the frames
// rather than overtaking them.
type pipelineStarted struct{}

func newObserverWorker(o processor.Observer) *observerWorker {
	return &observerWorker{
		observer: o,
		queue:    newReportQueue(),
		quit:     make(chan struct{}),
		done:     make(chan struct{}),
	}
}

// newObserverProxy builds a proxy over the given observers.
func newObserverProxy(observers []processor.Observer) *observerProxy {
	p := &observerProxy{}
	for _, o := range observers {
		p.observers = append(p.observers, newObserverWorker(o))
	}
	return p
}

// start runs a goroutine per observer. Reports made before this are dropped, as
// there is nothing yet to deliver them.
func (p *observerProxy) start() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.started {
		return
	}
	p.started = true
	for _, w := range p.observers {
		go w.run()
	}
}

// stop ends the observer goroutines and waits for them to finish. A report still
// waiting for an observer is not delivered: watching stops when the pipeline
// does.
func (p *observerProxy) stop() {
	p.mu.Lock()
	if p.stopped || !p.started {
		p.stopped = true
		p.mu.Unlock()
		return
	}
	p.stopped = true
	workers := p.observers
	p.mu.Unlock()

	for _, w := range workers {
		w.cancel()
	}
}

// add registers another observer while the pipeline runs, starting its goroutine
// if the others are already running.
func (p *observerProxy) add(o processor.Observer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return
	}
	w := newObserverWorker(o)
	p.observers = append(p.observers, w)
	if !p.started {
		return
	}
	go w.run()
}

// remove drops an observer while the pipeline runs. It stops reporting to it
// before returning, so a caller may release whatever the observer holds.
func (p *observerProxy) remove(o processor.Observer) {
	p.mu.Lock()
	var found *observerWorker
	kept := p.observers[:0:0]
	for _, w := range p.observers {
		if found == nil && w.observer == o {
			found = w
			continue
		}
		kept = append(kept, w)
	}
	p.observers = kept
	started := p.started
	p.mu.Unlock()

	// Nothing is running before the pipeline starts, so there is no goroutine to
	// end: dropping it from the list is the whole of it.
	if found != nil && started {
		found.cancel()
	}
}

// pipelineStarted reports the pipeline having started to every observer. It goes
// through the same queues as the frames, so an observer hears it in order.
func (p *observerProxy) pipelineStarted() { p.send(pipelineStarted{}) }

// OnPushFrame implements processor.Observer.
func (p *observerProxy) OnPushFrame(data processor.FramePushed) { p.send(data) }

// OnProcessFrame implements processor.ProcessObserver.
func (p *observerProxy) OnProcessFrame(data processor.FrameProcessed) { p.send(data) }

// send hands a report to every observer's queue without blocking the pipeline.
func (p *observerProxy) send(data any) {
	p.mu.Lock()
	workers := p.observers
	started := p.started
	p.mu.Unlock()
	if !started {
		return
	}
	for _, w := range workers {
		w.queue.push(data)
	}
}

// run delivers this observer's reports, in order, until the worker is canceled.
func (w *observerWorker) run() {
	defer close(w.done)
	for {
		data, ok := w.queue.get(w.quit)
		if !ok {
			return
		}
		w.deliver(data)
	}
}

// cancel ends this worker's goroutine and waits for it to finish. It is safe to
// call more than once, and from more than one goroutine.
func (w *observerWorker) cancel() {
	w.stopping.Do(func() { close(w.quit) })
	<-w.done
}

// deliver passes one report to the observer.
func (w *observerWorker) deliver(data any) {
	switch d := data.(type) {
	case processor.FramePushed:
		w.observer.OnPushFrame(d)
	case processor.FrameProcessed:
		if po, ok := w.observer.(processor.ProcessObserver); ok {
			po.OnProcessFrame(d)
		}
	case pipelineStarted:
		if ps, ok := w.observer.(processor.PipelineStartedObserver); ok {
			ps.OnPipelineStarted()
		}
	}
}

// reportQueue is an unbounded, concurrency-safe FIFO of reports with a single
// consumer, holding what is waiting for one observer.
//
// It is unbounded because nothing is dropped while the pipeline runs, however
// far behind an observer falls. The stateful ones are counting, and one that
// lost the start of a turn or the close of a tool call would report a
// conversation that never happened. Producers never block, so an observer that
// cannot keep up grows its queue rather than holding up the frames.
type reportQueue struct {
	mu     sync.Mutex
	items  []any
	notify chan struct{}
}

func newReportQueue() *reportQueue {
	return &reportQueue{notify: make(chan struct{}, 1)}
}

// push appends a report. It never blocks and never drops.
func (q *reportQueue) push(data any) {
	q.mu.Lock()
	q.items = append(q.items, data)
	q.mu.Unlock()

	// Wake a waiting get. The buffer of one means a signal is never lost: if no
	// one is waiting the pending wake is coalesced and drained on the next get.
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// tryGet pops the next report without blocking, reporting false when none is
// waiting.
func (q *reportQueue) tryGet() (any, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	data := q.items[0]
	q.items = q.items[1:]
	return data, true
}

// get returns the next report, blocking until one is waiting or quit is closed.
// It reports ok=false as soon as quit closes, whether or not reports are still
// waiting: watching stops when the pipeline does.
func (q *reportQueue) get(quit <-chan struct{}) (any, bool) {
	for {
		select {
		case <-quit:
			return nil, false
		default:
		}
		if data, ok := q.tryGet(); ok {
			return data, true
		}
		select {
		case <-quit:
			return nil, false
		case <-q.notify:
		}
	}
}

// Compile-time interface checks.
var (
	_ processor.Observer        = (*observerProxy)(nil)
	_ processor.ProcessObserver = (*observerProxy)(nil)
)
