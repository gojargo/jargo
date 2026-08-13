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
// therefore waiting on them. An asynchronous event runs each handler on its own
// goroutine and Call returns immediately. Cleanup waits for the handlers still
// running.
//
// A handler that panics is reported and does not bring down the object that
// raised the event, nor stop the other handlers for it.
package events

import (
	"context"
	"fmt"
	"log/slog"
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

	// running counts the goroutines started for asynchronous handlers, so
	// Cleanup can wait for them.
	running sync.WaitGroup
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
// have all finished. For an asynchronous one it starts a goroutine per handler
// and returns straight away; Cleanup waits for those.
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
	r.mu.Unlock()

	for _, h := range handlers {
		if isSync {
			run(ctx, name, h.fn, source, args)
			continue
		}
		r.running.Add(1)
		go func(fn Handler) {
			defer r.running.Done()
			run(ctx, name, fn, source, args)
		}(h.fn)
	}
}

// Cleanup waits for the handlers of asynchronous events that are still running.
// Call it when the object raising the events is done with, so a handler is not
// left running past the thing it was observing.
func (r *Registry) Cleanup(ctx context.Context) {
	done := make(chan struct{})
	go func() {
		r.running.Wait()
		close(done)
	}()
	select {
	case <-done:
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
				"source", fmt.Sprint(source),
				"panic", v,
				"stack", string(debug.Stack()))
		}
	}()
	fn(ctx, source, args...)
}
