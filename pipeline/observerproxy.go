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
	quit      chan struct{}
	wg        sync.WaitGroup
}

// observerWorker is one observer with the queue of reports waiting for it.
type observerWorker struct {
	observer processor.Observer
	queue    chan any
	// dropped counts the reports discarded because the observer fell too far
	// behind, so the pipeline is never blocked by one that cannot keep up.
	dropped int
}

// observerQueueSize is how many reports may be waiting for one observer before
// the oldest are dropped. An observer that falls this far behind is not going to
// catch up, and holding frames for it would make watching the pipeline change
// how the pipeline runs.
const observerQueueSize = 512

// newObserverProxy builds a proxy over the given observers.
func newObserverProxy(observers []processor.Observer) *observerProxy {
	p := &observerProxy{quit: make(chan struct{})}
	for _, o := range observers {
		p.observers = append(p.observers, &observerWorker{
			observer: o,
			queue:    make(chan any, observerQueueSize),
		})
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
		p.wg.Go(func() { w.run(p.quit) })
	}
}

// stop delivers what is still queued and waits for the observer goroutines to
// finish.
//
// Everything queued is handed over rather than dropped. The end of a call is
// exactly when the reports that complete a picture arrive, the last of a turn or
// the close of a tool call, and an observer that lost them would be reporting a
// conversation that never finished.
func (p *observerProxy) stop() {
	p.mu.Lock()
	if p.stopped || !p.started {
		p.stopped = true
		p.mu.Unlock()
		return
	}
	p.stopped = true
	p.mu.Unlock()

	close(p.quit)
	p.wg.Wait()
}

// add registers another observer while the pipeline runs, starting its goroutine
// if the others are already running.
func (p *observerProxy) add(o processor.Observer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return
	}
	w := &observerWorker{observer: o, queue: make(chan any, observerQueueSize)}
	p.observers = append(p.observers, w)
	if !p.started {
		return
	}
	p.wg.Go(func() { w.run(p.quit) })
}

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
		w.offer(data)
	}
}

// offer queues a report, dropping the oldest waiting one when the observer has
// fallen behind. Frames matter more than watching them.
func (w *observerWorker) offer(data any) {
	select {
	case w.queue <- data:
		return
	default:
	}
	select {
	case <-w.queue:
		w.dropped++
	default:
	}
	select {
	case w.queue <- data:
	default:
	}
}

// run delivers this observer's reports, in order, until the proxy stops, then
// hands over whatever is still waiting.
func (w *observerWorker) run(quit <-chan struct{}) {
	for {
		select {
		case data := <-w.queue:
			w.deliver(data)
		case <-quit:
			w.drain()
			return
		}
	}
}

// drain delivers the reports already queued and returns.
func (w *observerWorker) drain() {
	for {
		select {
		case data := <-w.queue:
			w.deliver(data)
		default:
			return
		}
	}
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
	}
}

// Compile-time interface checks.
var (
	_ processor.Observer        = (*observerProxy)(nil)
	_ processor.ProcessObserver = (*observerProxy)(nil)
)
