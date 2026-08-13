package registry_test

import (
	"context"
	"slices"
	"testing"

	"github.com/gojargo/jargo/registry"
)

// Ported from upstream's registry suite. Upstream dedups a watch by comparing
// the handler functions; Go cannot compare functions, so the watcher names its
// interest with a key and the idempotence case below watches twice under the
// same key.

func newRegistry() *registry.WorkerRegistry { return registry.New("runner_a") }

func collect(into *[]registry.WorkerReadyData) registry.WatchHandler {
	return func(_ context.Context, d registry.WorkerReadyData) { *into = append(*into, d) }
}

func TestRegisterLocalWorker(t *testing.T) {
	r := newRegistry()

	if !r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"}) {
		t.Fatal("Register reported the worker was already known")
	}
	if !slices.Contains(r.LocalWorkers(), "greeter") {
		t.Errorf("LocalWorkers = %v, want it to hold greeter", r.LocalWorkers())
	}
	if slices.Contains(r.RemoteWorkers(), "greeter") {
		t.Errorf("RemoteWorkers = %v, want greeter counted as local", r.RemoteWorkers())
	}
}

func TestRegisterRemoteWorker(t *testing.T) {
	r := newRegistry()

	if !r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "support", Runner: "runner_b"}) {
		t.Fatal("Register reported the worker was already known")
	}
	if !slices.Contains(r.RemoteWorkers(), "support") {
		t.Errorf("RemoteWorkers = %v, want it to hold support", r.RemoteWorkers())
	}
	if slices.Contains(r.LocalWorkers(), "support") {
		t.Errorf("LocalWorkers = %v, want support counted as remote", r.LocalWorkers())
	}
}

func TestDuplicateRegistrationReportsNotNew(t *testing.T) {
	r := newRegistry()
	data := registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"}

	if !r.Register(t.Context(), data) {
		t.Fatal("the first registration should be new")
	}
	if r.Register(t.Context(), data) {
		t.Error("the second registration should report the worker was already known")
	}
}

func TestGetLocalWorker(t *testing.T) {
	r := newRegistry()
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})

	got, ok := r.Get("greeter")
	if !ok || got.Runner != "runner_a" {
		t.Errorf("Get(greeter) = %+v, %v, want the local worker", got, ok)
	}
}

func TestGetRemoteWorker(t *testing.T) {
	r := newRegistry()
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "support", Runner: "runner_b"})

	got, ok := r.Get("support")
	if !ok || got.Runner != "runner_b" {
		t.Errorf("Get(support) = %+v, %v, want the remote worker", got, ok)
	}
}

func TestGetUnknownWorker(t *testing.T) {
	if _, ok := newRegistry().Get("nobody"); ok {
		t.Error("Get reported an unknown worker as registered")
	}
}

func TestContains(t *testing.T) {
	r := newRegistry()
	if r.Contains("greeter") {
		t.Error("Contains reported an unregistered worker")
	}
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})
	if !r.Contains("greeter") {
		t.Error("Contains did not report a registered worker")
	}
}

func TestWatchFiresOnRegistration(t *testing.T) {
	r := newRegistry()
	var got []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "w", collect(&got))

	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})

	if len(got) != 1 || got[0].WorkerName != "greeter" {
		t.Errorf("the watcher received %+v, want one greeter", got)
	}
}

// Watching the same worker twice from the same place must not double-fire: a
// parent can reach the same watch by two routes.
func TestWatchIsIdempotentForTheSameWatcher(t *testing.T) {
	r := newRegistry()
	var got []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "w", collect(&got))
	r.Watch(t.Context(), "greeter", "w", collect(&got))

	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})

	if len(got) != 1 {
		t.Errorf("the watcher fired %d times, want 1", len(got))
	}
}

func TestWatchDoesNotFireForOtherWorkers(t *testing.T) {
	r := newRegistry()
	var got []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "w", collect(&got))

	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "support", Runner: "runner_a"})

	if len(got) != 0 {
		t.Errorf("the watcher fired for another worker: %+v", got)
	}
}

func TestWatchDoesNotFireOnDuplicateRegistration(t *testing.T) {
	r := newRegistry()
	var got []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "w", collect(&got))

	data := registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"}
	r.Register(t.Context(), data)
	r.Register(t.Context(), data)

	if len(got) != 1 {
		t.Errorf("the watcher fired %d times, want 1: the second registration was not new", len(got))
	}
}

func TestMultipleWatchers(t *testing.T) {
	r := newRegistry()
	var a, b []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "a", collect(&a))
	r.Watch(t.Context(), "greeter", "b", collect(&b))

	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})

	if len(a) != 1 || len(b) != 1 {
		t.Errorf("watchers fired %d and %d times, want 1 each", len(a), len(b))
	}
}

func TestWatchFiresImmediatelyWhenAlreadyRegistered(t *testing.T) {
	r := newRegistry()
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "greeter", Runner: "runner_a"})

	var got []registry.WorkerReadyData
	r.Watch(t.Context(), "greeter", "w", collect(&got))

	if len(got) != 1 {
		t.Errorf("the watcher fired %d times, want 1 for a worker already registered", len(got))
	}
}

func TestRunnerName(t *testing.T) {
	if got := newRegistry().RunnerName(); got != "runner_a" {
		t.Errorf("RunnerName() = %q, want runner_a", got)
	}
}

func TestMultipleRemoteRunners(t *testing.T) {
	r := newRegistry()
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "support", Runner: "runner_b"})
	r.Register(t.Context(), registry.WorkerReadyData{WorkerName: "billing", Runner: "runner_c"})

	remote := r.RemoteWorkers()
	if len(remote) != 2 || !slices.Contains(remote, "support") || !slices.Contains(remote, "billing") {
		t.Errorf("RemoteWorkers = %v, want both remote workers", remote)
	}
	if got, _ := r.Get("billing"); got.Runner != "runner_c" {
		t.Errorf("Get(billing).Runner = %q, want runner_c", got.Runner)
	}
}
