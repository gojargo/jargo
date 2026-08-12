// Package notify carries the signaling primitives processors use to coordinate
// with one another without being wired together in the pipeline.
package notify

import "context"

// Notifier signals that something has happened, to whoever is waiting on it.
//
// It is what lets one processor release another: a filter watching the stream
// notifies, and a gate somewhere else in the pipeline wakes and lets its held
// frame through. Neither has to know where the other sits.
type Notifier interface {
	// Notify signals whoever is waiting.
	Notify()
	// Wait blocks until Notify is called, or until ctx ends. It reports false
	// when the context ended first, so a caller in a loop knows to stop rather
	// than treating it as a signal.
	Wait(ctx context.Context) bool
}

// EventNotifier is a Notifier for a single waiter.
//
// A notification with nobody waiting is remembered, so a waiter arriving after
// it does not block; it is remembered once, not counted, so several
// notifications with nobody waiting release one wait. Waiting from more than one
// goroutine is not supported: a notification releases one of them and the others
// keep waiting.
type EventNotifier struct {
	// ch carries the one remembered notification. It is buffered so Notify
	// never blocks, and a full buffer is what "already notified" looks like.
	ch chan struct{}
}

// NewEventNotifier builds an EventNotifier with nothing signaled yet.
func NewEventNotifier() *EventNotifier {
	return &EventNotifier{ch: make(chan struct{}, 1)}
}

// Notify implements Notifier. It never blocks: with a notification already
// remembered there is nothing to add.
func (n *EventNotifier) Notify() {
	select {
	case n.ch <- struct{}{}:
	default:
	}
}

// Wait implements Notifier.
func (n *EventNotifier) Wait(ctx context.Context) bool {
	select {
	case <-n.ch:
		return true
	case <-ctx.Done():
		return false
	}
}

var _ Notifier = (*EventNotifier)(nil)
