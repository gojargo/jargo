package notify_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/utils/notify"
)

// A waiter already waiting is released by a signal.
func TestNotifierReleasesAWaiter(t *testing.T) {
	n := notify.NewEventNotifier()

	released := make(chan bool, 1)
	go func() { released <- n.Wait(context.Background()) }()

	// Give the waiter a moment to be waiting rather than about to.
	time.Sleep(20 * time.Millisecond)
	n.Notify()

	select {
	case ok := <-released:
		if !ok {
			t.Fatal("the wait reported no signal")
		}
	case <-time.After(time.Second):
		t.Fatal("the waiter was never released")
	}
}

// A signal raised with nobody waiting is remembered, so a waiter arriving after
// it does not block.
func TestNotifierRemembersASignal(t *testing.T) {
	n := notify.NewEventNotifier()
	n.Notify()

	if !n.Wait(context.Background()) {
		t.Fatal("a signal raised before the wait was lost")
	}
}

// It is remembered once rather than counted, so several signals with nobody
// waiting release one wait.
func TestNotifierRemembersOneSignalOnly(t *testing.T) {
	n := notify.NewEventNotifier()
	n.Notify()
	n.Notify()
	n.Notify()

	if !n.Wait(context.Background()) {
		t.Fatal("the remembered signal was lost")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if n.Wait(ctx) {
		t.Fatal("a second wait was released, so the signals were counted")
	}
}

// Signaling never blocks, whether or not anyone is waiting.
func TestNotifierNeverBlocks(t *testing.T) {
	n := notify.NewEventNotifier()

	done := make(chan struct{})
	go func() {
		for range 100 {
			n.Notify()
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("signaling blocked")
	}
}

// A wait whose context ends reports that it was not a signal, so a caller in a
// loop stops rather than acting on nothing.
func TestNotifierReportsAContextThatEnded(t *testing.T) {
	n := notify.NewEventNotifier()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if n.Wait(ctx) {
		t.Fatal("a context that ended was reported as a signal")
	}
}
