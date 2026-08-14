// Package events is the event-handler registry an object exposes to report
// what it did, for the things a caller wants to observe but the frame path does
// not carry: a summary being applied, a socket failing to open, a worker
// becoming ready.
//
// An object declares the events it raises with Register, once, at construction.
// A caller then attaches handlers to a declared name with Add. Attaching to a
// name the object never declared is a mistake the registry reports rather than
// accepts, so a typo in an event name is caught instead of silently never
// firing.
//
// An event is either synchronous or asynchronous, fixed when it is declared. A
// synchronous event runs its handlers inline, in the order they were added, and
// Call returns only once they have all finished; the object raising it is
// therefore waiting on them. An asynchronous event queues its handlers and Call
// returns straight away, leaving the object free to carry on. Cleanup waits for
// the handlers still queued or running.
//
// The queue behind an asynchronous event is ordered, and one handler runs at a
// time. That matters: a handler is very often accumulating something, and it
// would otherwise see two firings of the same event in either order, or see
// them at once and have to lock against itself. So a handler observes the
// events of one object in the order that object raised them, and never runs
// beside another handler of the same object.
//
// It is stricter in one place: a handler that blocks holds up the handlers
// queued behind it. A blocking handler is worth avoiding for that reason.
//
// A handler that panics is reported and does not bring down the object that
// raised the event, nor stop the other handlers for it.
package events

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"runtime/debug"
	"sync"
)

// Handler is called when the event it is attached to fires. source is the
// object that raised the event, and args are the event's own arguments, whose
// number and types belong to that event. See On for a typed single-argument
// handler, which is the shape of every event that carries one value.
type Handler func(ctx context.Context, source any, args ...any)

// HandlerID identifies a handler that was added, so it can be removed again. Go
// functions are not comparable, so a handler cannot be removed by passing the
// same function back; Add returns this instead.
type HandlerID uint64

// event is one declared event: the handlers attached to it and whether they run
// inline.
type event struct {
	name     string
	sync     bool
	handlers []registered
}

// registered is one attached handler and the id that removes it.
type registered struct {
	id HandlerID
	fn Handler
}

// Registry holds the events an object declares and the handlers attached to
// them. Embed it in the object that raises the events:
//
//	type Summarizer struct {
//	    events.Registry
//	    …
//	}
//
//	func New() *Summarizer {
//	    s := &Summarizer{}
//	    s.Register("on_summary_applied", false)
//	    return s
//	}
//
// The zero value is ready to use. It is safe for concurrent use.
type Registry struct {
	mu     sync.Mutex
	events map[string]*event
	nextID HandlerID

	// queue holds the asynchronous handler calls waiting to run, in the order
	// they were raised, and dispatching reports whether a goroutine is working
	// through them. The goroutine runs only while there is something queued.
	queue       []call
	dispatching bool
	// inFlight counts the queued call being run, if any, so Cleanup can tell an
	// empty queue from a finished one.
	inFlight int
	// idle is closed once the queue has drained, and exists only while
	// something is waiting for that.
	idle chan struct{}
}

// call is one handler waiting to be run, with everything the firing carried.
type call struct {
	//nolint:containedctx // the context of the firing, carried to the handler
	ctx    context.Context
	name   string
	fn     Handler
	source any
	args   []any
}

// Register declares an event this object raises. sync fixes how its handlers
// run: inline and awaited when true, each on its own goroutine when false.
//
// Call it once per event, at construction. Declaring the same name twice is
// reported and leaves the first declaration in place.
func (r *Registry) Register(name string, sync bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.events == nil {
		r.events = make(map[string]*event)
	}
	if _, ok := r.events[name]; ok {
		slog.Warn("event handler already registered", "event", name)
		return
	}
	r.events[name] = &event{name: name, sync: sync}
}

// Add attaches h to the event named name and returns the id that removes it
// again. Handlers run in the order they were added.
//
// Adding to a name that was never declared with Register attaches nothing and
// is reported: the event would otherwise never fire and the caller would have no
// way to tell. The returned id is then zero, which Remove ignores.
func (r *Registry) Add(name string, h Handler) HandlerID {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.events[name]
	if !ok {
		slog.Warn("event handler not registered", "event", name)
		return 0
	}
	r.nextID++
	id := r.nextID
	e.handlers = append(e.handlers, registered{id: id, fn: h})
	return id
}

// On attaches a typed handler to an event carrying a single value, which is the
// shape of most of them, and returns the id that removes it again.
//
// It is Add with the value already read out of the arguments, so a handler is
// written against what the event carries rather than against a slice of any. A
// firing that carries something else is reported and skipped, since a handler
// that cannot read its own event would otherwise do so silently.
//
//	events.On(&worker.Registry, workers.EventJobCompleted,
//	    func(ctx context.Context, result jobcontext.GroupResponse) { … })
func On[T any](r *Registry, name string, fn func(ctx context.Context, value T)) HandlerID {
	return r.Add(name, func(ctx context.Context, _ any, args ...any) {
		if len(args) == 0 {
			slog.Warn("event carries no value for a typed handler", "event", name)
			return
		}
		value, ok := args[0].(T)
		if !ok {
			slog.Warn("event carries a value of another type than the handler reads",
				"event", name,
				"carried", fmt.Sprintf("%T", args[0]),
				"wanted", fmt.Sprintf("%T", value))
			return
		}
		fn(ctx, value)
	})
}

// OnSignal attaches a handler to an event that carries nothing, and returns the
// id that removes it again. Such an event says only that something happened: a
// pipeline went idle, a heartbeat stopped arriving.
func OnSignal(r *Registry, name string, fn func(ctx context.Context)) HandlerID {
	return r.Add(name, func(ctx context.Context, _ any, _ ...any) { fn(ctx) })
}

// Remove detaches the handler Add returned id for. It is a no-op if the event
// was never declared, or the handler was already removed, or id is zero.
func (r *Registry) Remove(name string, id HandlerID) {
	if id == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.events[name]
	if !ok {
		return
	}
	for i, h := range e.handlers {
		if h.id == id {
			e.handlers = append(e.handlers[:i:i], e.handlers[i+1:]...)
			return
		}
	}
}

// Call fires the event named name, passing source and args to every handler
// attached to it. Firing an event that was never declared does nothing, which is
// what lets an object raise an event unconditionally whether or not anything
// declared interest in it.
//
// For a synchronous event Call runs the handlers inline and returns once they
// have all finished. For an asynchronous one it queues them and returns straight
// away; they run one at a time, in the order they were queued, and Cleanup waits
// for them.
//
// Call never blocks on an asynchronous event, so a handler is free to raise
// another event on the same object.
func (r *Registry) Call(ctx context.Context, name string, source any, args ...any) {
	r.mu.Lock()
	e, ok := r.events[name]
	if !ok {
		r.mu.Unlock()
		return
	}
	// Copy under the lock: a handler is free to add or remove handlers, and the
	// slice would otherwise be rewritten while it is being walked.
	handlers := append([]registered(nil), e.handlers...)
	isSync := e.sync

	if isSync {
		r.mu.Unlock()
		for _, h := range handlers {
			run(ctx, name, h.fn, source, args)
		}
		return
	}

	// Queued under the lock, which is what puts the handlers of two firings in
	// the order the object raised them.
	for _, h := range handlers {
		r.queue = append(r.queue, call{ctx: ctx, name: name, fn: h.fn, source: source, args: args})
	}
	if !r.dispatching && len(r.queue) > 0 {
		r.dispatching = true
		go r.dispatch()
	}
	r.mu.Unlock()
}

// dispatch runs the queued handlers one at a time until the queue is empty, and
// then returns. The next Call starts it again.
func (r *Registry) dispatch() {
	for {
		r.mu.Lock()
		if len(r.queue) == 0 {
			r.dispatching = false
			r.signalIdleLocked()
			r.mu.Unlock()
			return
		}
		next := r.queue[0]
		r.queue = r.queue[1:]
		r.inFlight++
		r.mu.Unlock()

		run(next.ctx, next.name, next.fn, next.source, next.args)

		r.mu.Lock()
		r.inFlight--
		r.mu.Unlock()
	}
}

// signalIdleLocked tells whoever is waiting that the queue has drained. The
// caller holds r.mu.
func (r *Registry) signalIdleLocked() {
	if len(r.queue) == 0 && r.inFlight == 0 && r.idle != nil {
		close(r.idle)
		r.idle = nil
	}
}

// Cleanup waits for the handlers of asynchronous events that are still queued or
// running. Call it when the object raising the events is done with, so a handler
// is not left running past the thing it was observing.
//
// Do not call it from a handler: it would be waiting for itself.
func (r *Registry) Cleanup(ctx context.Context) {
	r.mu.Lock()
	if len(r.queue) == 0 && r.inFlight == 0 {
		r.mu.Unlock()
		return
	}
	if r.idle == nil {
		r.idle = make(chan struct{})
	}
	idle := r.idle
	r.mu.Unlock()

	select {
	case <-idle:
	case <-ctx.Done():
	}
}

// run calls one handler, turning a panic into a report. A handler belongs to the
// caller, so a fault in it must not propagate into the object that raised the
// event or stop the other handlers for it.
func run(ctx context.Context, name string, fn Handler, source any, args []any) {
	defer func() {
		if v := recover(); v != nil {
			slog.Error("uncaught panic in event handler",
				"event", name,
				"source", sourceName(source),
				"panic", v,
				"stack", string(debug.Stack()))
		}
	}()
	fn(ctx, source, args...)
}

// sourceName identifies the object that raised an event for a log line: its
// name when it has one ("OpenAILLM#3"), and the name of its type otherwise.
//
// The object itself is never formatted. It is the thing that raised the event,
// which is very often a service holding an API key, and writing out its fields
// would put that key in the log.
func sourceName(source any) string {
	switch s := source.(type) {
	case nil:
		return "<nil>"
	case interface{ Name() string }:
		return s.Name()
	default:
		return reflect.TypeOf(source).String()
	}
}
