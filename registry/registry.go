// Package registry tracks the workers a runner knows about, its own and those
// belonging to other runners, so a worker can find another by name and be told
// when one it is waiting for becomes ready.
//
// A runner owns one registry and shares it with its workers. Workers are kept
// under the runner they belong to, so a local one is told apart from a remote
// one, and each name is registered at most once.
package registry

import (
	"context"
	"log/slog"
	"sync"
)

// WorkerReadyData is what a worker reports about itself when it becomes ready.
type WorkerReadyData struct {
	// WorkerName is the worker's name.
	WorkerName string
	// Runner is the name of the runner managing it.
	Runner string
}

// WorkerErrorData describes a worker that failed.
type WorkerErrorData struct {
	// WorkerName is the name of the worker that failed.
	WorkerName string
	// Error describes the failure.
	Error string
}

// WorkerRegistryEntry is one worker in a snapshot of the registry.
type WorkerRegistryEntry struct {
	// Name is the worker's name.
	Name string
	// Parent is the name of the worker's parent, empty for a root worker.
	Parent string
	// Active reports whether the worker is currently active.
	Active bool
	// Bridged reports whether the worker is bridged.
	Bridged bool
	// StartedAt is when the worker became ready, as a Unix timestamp, and zero
	// when it has not.
	StartedAt float64
}

// WatchHandler is called with a worker's data when that worker registers.
type WatchHandler func(ctx context.Context, data WorkerReadyData)

// watch is one registered interest in a worker becoming ready.
type watch struct {
	key     string
	handler WatchHandler
}

// WorkerRegistry tracks the workers known to one runner.
//
// It is safe for concurrent use.
type WorkerRegistry struct {
	runnerName string

	mu sync.Mutex
	// local holds the workers belonging to this runner, by name.
	local map[string]WorkerReadyData
	// remote holds the workers of every other runner, by runner then by name.
	remote map[string]map[string]WorkerReadyData
	// watches holds the interests in each worker name.
	watches map[string][]watch
}

// New builds a registry owned by the named runner.
func New(runnerName string) *WorkerRegistry {
	return &WorkerRegistry{
		runnerName: runnerName,
		local:      make(map[string]WorkerReadyData),
		remote:     make(map[string]map[string]WorkerReadyData),
		watches:    make(map[string][]watch),
	}
}

// RunnerName is the name of the runner that owns this registry.
func (r *WorkerRegistry) RunnerName() string { return r.runnerName }

// LocalWorkers are the names of the workers registered under this runner.
func (r *WorkerRegistry) LocalWorkers() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.local))
	for name := range r.local {
		out = append(out, name)
	}
	return out
}

// RemoteWorkers are the names of the workers registered under other runners.
func (r *WorkerRegistry) RemoteWorkers() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, workers := range r.remote {
		for name := range workers {
			out = append(out, name)
		}
	}
	return out
}

// Get looks a worker up by name, reporting whether it is registered at all.
func (r *WorkerRegistry) Get(workerName string) (WorkerReadyData, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.getLocked(workerName)
}

// getLocked is Get with the lock already held.
func (r *WorkerRegistry) getLocked(workerName string) (WorkerReadyData, bool) {
	if d, ok := r.local[workerName]; ok {
		return d, true
	}
	for _, workers := range r.remote {
		if d, ok := workers[workerName]; ok {
			return d, true
		}
	}
	return WorkerReadyData{}, false
}

// Contains reports whether a worker of that name is registered.
func (r *WorkerRegistry) Contains(workerName string) bool {
	_, ok := r.Get(workerName)
	return ok
}

// Watch asks to be told when the named worker registers, and calls handler
// straight away when it already has.
//
// key identifies the watcher, so watching the same worker twice from the same
// place is a no-op rather than firing the handler twice. Upstream compares the
// handler functions themselves; Go cannot, so the caller names the interest
// instead. It matters because a parent can reach the same watch by two routes,
// adding a child worker and declaring a ready handler for it by name.
func (r *WorkerRegistry) Watch(ctx context.Context, workerName, key string, handler WatchHandler) {
	r.mu.Lock()
	for _, w := range r.watches[workerName] {
		if w.key == key {
			r.mu.Unlock()
			return
		}
	}
	r.watches[workerName] = append(r.watches[workerName], watch{key: key, handler: handler})
	existing, registered := r.getLocked(workerName)
	r.mu.Unlock()

	if registered {
		handler(ctx, existing)
	}
}

// Register records a worker and tells whoever was watching for it. It reports
// whether the worker was new; registering one already known changes nothing.
func (r *WorkerRegistry) Register(ctx context.Context, data WorkerReadyData) bool {
	r.mu.Lock()

	isLocal := data.Runner == r.runnerName
	target := r.local
	if !isLocal {
		if r.remote[data.Runner] == nil {
			r.remote[data.Runner] = make(map[string]WorkerReadyData)
		}
		target = r.remote[data.Runner]
	}

	if _, ok := target[data.WorkerName]; ok {
		r.mu.Unlock()
		return false
	}

	// The same name under two runners is not rejected, because neither runner is
	// authoritative over the other, but it is reported: whoever looks the name up
	// will reach one of them arbitrarily.
	if existing, ok := r.getLocked(data.WorkerName); ok && existing.Runner != data.Runner {
		slog.Warn("worker registered under two runners",
			"worker", data.WorkerName, "runners", []string{existing.Runner, data.Runner})
	}

	target[data.WorkerName] = data
	handlers := append([]watch(nil), r.watches[data.WorkerName]...)
	r.mu.Unlock()

	locality := data.Runner
	if isLocal {
		locality = "local"
	}
	slog.Debug("worker ready", "worker", data.WorkerName, "runner", locality)

	for _, w := range handlers {
		w.handler(ctx, data)
	}
	return true
}
