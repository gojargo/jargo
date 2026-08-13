package events

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"
)

// Upstream ships no test for the object this package ports, so these are ours.

type source struct{ Registry }

func TestHandlersRunInTheOrderTheyWereAdded(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	var order []string
	s.Add("on_thing", func(context.Context, any, ...any) { order = append(order, "first") })
	s.Add("on_thing", func(context.Context, any, ...any) { order = append(order, "second") })

	s.Call(t.Context(), "on_thing", s)

	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Errorf("handlers ran as %v, want [first second]", order)
	}
}

func TestSourceAndArgumentsReachTheHandler(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	var gotSource any
	var gotArgs []any
	s.Add("on_thing", func(_ context.Context, src any, args ...any) {
		gotSource, gotArgs = src, args
	})

	s.Call(t.Context(), "on_thing", s, 42, "text")

	if gotSource != any(s) {
		t.Error("the handler did not receive the object that raised the event")
	}
	if len(gotArgs) != 2 || gotArgs[0] != 42 || gotArgs[1] != "text" {
		t.Errorf("the handler received %v, want [42 text]", gotArgs)
	}
}

// Attaching to an event the object never declared attaches nothing: the event
// would never fire and the caller would have no way to tell.
func TestAddingToAnUndeclaredEventAttachesNothing(t *testing.T) {
	s := &source{}
	s.Register("on_declared", true)

	called := false
	id := s.Add("on_undeclared", func(context.Context, any, ...any) { called = true })

	if id != 0 {
		t.Errorf("Add returned id %d for an undeclared event, want 0", id)
	}
	s.Call(t.Context(), "on_undeclared", s)
	if called {
		t.Error("a handler attached to an undeclared event ran")
	}
}

// Raising an event nothing is attached to, or that was never declared, is what
// lets an object report unconditionally.
func TestCallingAnEventWithNoHandlersDoesNothing(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)
	s.Call(t.Context(), "on_thing", s)
	s.Call(t.Context(), "never_declared", s)
}

func TestRemoveDetachesTheHandler(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	calls := 0
	id := s.Add("on_thing", func(context.Context, any, ...any) { calls++ })

	s.Call(t.Context(), "on_thing", s)
	s.Remove("on_thing", id)
	s.Call(t.Context(), "on_thing", s)

	if calls != 1 {
		t.Errorf("the handler ran %d times, want 1", calls)
	}

	// Removing again, and removing a zero id, are both no-ops.
	s.Remove("on_thing", id)
	s.Remove("on_thing", 0)
	s.Remove("never_declared", id)
}

// A synchronous event is awaited: the object raising it is waiting on its
// handlers, so Call must not return before they finish.
func TestSynchronousEventIsAwaited(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	done := false
	s.Add("on_thing", func(context.Context, any, ...any) {
		time.Sleep(10 * time.Millisecond)
		done = true
	})

	s.Call(t.Context(), "on_thing", s)

	if !done {
		t.Error("Call returned before a synchronous handler finished")
	}
}

// An asynchronous event queues its handlers rather than running them inline,
// and Cleanup is what waits for them.
func TestAsynchronousEventIsAwaitedByCleanup(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	var mu sync.Mutex
	done := 0
	for range 3 {
		s.Add("on_thing", func(context.Context, any, ...any) {
			time.Sleep(10 * time.Millisecond)
			mu.Lock()
			done++
			mu.Unlock()
		})
	}

	s.Call(t.Context(), "on_thing", s)
	s.Cleanup(t.Context())

	mu.Lock()
	defer mu.Unlock()
	if done != 3 {
		t.Errorf("Cleanup returned with %d of 3 handlers finished", done)
	}
}

// A handler belongs to the caller, so a fault in it must not propagate into the
// object that raised the event, nor stop the other handlers.
func TestAPanickingHandlerDoesNotStopTheOthers(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	ran := false
	//nolint:forbidigo // the fault this test exists to contain
	s.Add("on_thing", func(context.Context, any, ...any) { panic("handler fault") })
	s.Add("on_thing", func(context.Context, any, ...any) { ran = true })

	s.Call(t.Context(), "on_thing", s)

	if !ran {
		t.Error("a panic in one handler stopped the next one")
	}
}

// Declaring the same event twice leaves the first declaration, and its
// handlers, in place.
func TestDeclaringAnEventTwiceKeepsTheFirstDeclaration(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	called := false
	s.Add("on_thing", func(context.Context, any, ...any) { called = true })
	s.Register("on_thing", false)

	s.Call(t.Context(), "on_thing", s)

	if !called {
		t.Error("redeclaring the event dropped the handler attached to it")
	}
}

// A handler is free to attach or detach handlers while the event is firing: the
// registry walks a copy, so the slice is not rewritten underneath it.
func TestAHandlerMayChangeTheHandlersWhileTheEventFires(t *testing.T) {
	s := &source{}
	s.Register("on_thing", true)

	s.Add("on_thing", func(ctx context.Context, src any, _ ...any) {
		s.Add("on_thing", func(context.Context, any, ...any) {})
	})

	s.Call(t.Context(), "on_thing", s)
}

// A handler is very often accumulating something, so it must see the firings of
// one object in the order that object raised them.
func TestAsynchronousFiringsReachAHandlerInOrder(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	var mu sync.Mutex
	var got []int
	s.Add("on_thing", func(_ context.Context, _ any, args ...any) {
		mu.Lock()
		got = append(got, args[0].(int)) //nolint:forcetypeassert // the event carries what this test passed
		mu.Unlock()
	})

	const firings = 50
	for i := range firings {
		s.Call(t.Context(), "on_thing", s, i)
	}
	s.Cleanup(t.Context())

	mu.Lock()
	defer mu.Unlock()
	if len(got) != firings {
		t.Fatalf("the handler saw %d of %d firings", len(got), firings)
	}
	for i, n := range got {
		if n != i {
			t.Fatalf("firing %d arrived at position %d, want them in the order they were raised: %v", n, i, got)
		}
	}
}

// The handlers of one object run one at a time, so a handler need not lock
// against another handler of the same object.
func TestAsynchronousHandlersDoNotRunAtOnce(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	var mu sync.Mutex
	running, maxRunning := 0, 0
	for range 4 {
		s.Add("on_thing", func(context.Context, any, ...any) {
			mu.Lock()
			running++
			maxRunning = max(maxRunning, running)
			mu.Unlock()

			time.Sleep(5 * time.Millisecond)

			mu.Lock()
			running--
			mu.Unlock()
		})
	}

	for range 3 {
		s.Call(t.Context(), "on_thing", s)
	}
	s.Cleanup(t.Context())

	mu.Lock()
	defer mu.Unlock()
	if maxRunning != 1 {
		t.Errorf("%d handlers ran at once, want them one at a time", maxRunning)
	}
}

// Firing an asynchronous event never waits for the handlers, so a handler is
// free to raise another event on the object it is handling one for.
func TestAHandlerMayRaiseAnotherEvent(t *testing.T) {
	s := &source{}
	s.Register("on_first", false)
	s.Register("on_second", false)

	second := make(chan struct{})
	s.Add("on_first", func(ctx context.Context, _ any, _ ...any) {
		s.Call(ctx, "on_second", s)
	})
	var once sync.Once
	s.Add("on_second", func(context.Context, any, ...any) {
		once.Do(func() { close(second) })
	})

	s.Call(t.Context(), "on_first", s)

	select {
	case <-second:
	case <-time.After(2 * time.Second):
		t.Fatal("the event raised from a handler never fired")
	}
}

// A typed handler is written against what the event carries, rather than
// against a slice of any.
func TestOnReadsTheValueTheEventCarries(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	var mu sync.Mutex
	var got []string
	On(&s.Registry, "on_thing", func(_ context.Context, v string) {
		mu.Lock()
		got = append(got, v)
		mu.Unlock()
	})

	s.Call(t.Context(), "on_thing", s, "first")
	s.Call(t.Context(), "on_thing", s, "second")
	s.Cleanup(t.Context())

	mu.Lock()
	defer mu.Unlock()
	if want := []string{"first", "second"}; !slices.Equal(got, want) {
		t.Errorf("handler saw %v, want %v", got, want)
	}
}

// A firing carrying something else is reported and skipped, rather than
// reaching a handler that cannot read it.
func TestOnSkipsAValueOfAnotherType(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	var mu sync.Mutex
	called := 0
	On(&s.Registry, "on_thing", func(context.Context, string) {
		mu.Lock()
		called++
		mu.Unlock()
	})

	s.Call(t.Context(), "on_thing", s, 42)
	s.Cleanup(t.Context())

	mu.Lock()
	defer mu.Unlock()
	if called != 0 {
		t.Errorf("the handler ran %d times for a value it cannot read, want 0", called)
	}
}

// An event that carries nothing says only that something happened.
func TestOnSignalRunsForAnEventCarryingNothing(t *testing.T) {
	s := &source{}
	s.Register("on_thing", false)

	fired := make(chan struct{})
	var once sync.Once
	OnSignal(&s.Registry, "on_thing", func(context.Context) {
		once.Do(func() { close(fired) })
	})

	s.Call(t.Context(), "on_thing", s)
	s.Cleanup(t.Context())

	select {
	case <-fired:
	default:
		t.Error("the handler never ran")
	}
}
