package workers

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/utils/events"
	"github.com/google/uuid"
)

// The events a runner raises.
const (
	// EventRunnerReady fires once the runner has set itself up and started the
	// workers added to it.
	EventRunnerReady = "on_ready"
	// EventRunnerError fires when a worker added to the runner fails to start.
	EventRunnerError = "on_error"
)

// StoppableWhenDone is a worker that can be asked to finish what it is doing
// and then stop, rather than being stopped where it stands. A worker driving a
// pipeline is one; a worker that only listens on the bus is not.
type StoppableWhenDone interface {
	Worker
	// StopWhenDone stops the worker once it has finished what it is doing.
	StopWhenDone()
}

// RunnerConfig configures a Runner.
type RunnerConfig struct {
	// Name identifies the runner, and must be unique among the runners sharing
	// a bus. Empty generates one.
	Name string
	// Bus is the bus to host and share with the workers; nil builds an
	// in-process one.
	Bus *bus.Bus
	// HandleInterrupt cancels the runner on an interrupt signal; nil defaults
	// to true.
	HandleInterrupt *bool
	// HandleTerminate cancels the runner on a termination signal.
	HandleTerminate bool
	// ForceGC collects garbage once every worker has finished.
	ForceGC bool
}

// RunOptions are the choices running offers.
type RunOptions struct {
	// AutoEnd ends the runner once every root worker has finished, which is
	// what makes a single-pipeline bot end when its pipeline does; nil defaults
	// to true. Set it false for a host that adds and removes workers across
	// many sessions, which should outlive any of them.
	AutoEnd *bool
}

// workerEntry is one worker registered on the runner, and the goroutine
// running it.
type workerEntry struct {
	worker Worker
	// done is closed when the worker's run returns, and is nil until it has
	// been started.
	done chan struct{}
}

// running reports whether the worker is still going.
func (e *workerEntry) running() bool {
	if e.done == nil {
		return false
	}
	select {
	case <-e.done:
		return false
	default:
		return true
	}
}

// finished reports whether the worker has run and stopped. One that has not been
// started has not finished either, so it is still owed the message that tells it
// to wind down.
func (e *workerEntry) finished() bool {
	return e.done != nil && !e.running()
}

// Runner runs workers to completion. It owns the bus they talk over, the
// registry they find each other through, and the goroutines they run on.
//
// Add the workers with AddWorkers and then call Run:
//
//	runner := workers.NewRunner(workers.RunnerConfig{})
//	runner.AddWorkers(ctx, worker)
//	err := runner.Run(ctx, workers.RunOptions{})
//
// Run ends once every root worker has finished, so a bot with one pipeline
// ends when that pipeline does. A bot whose helpers wait on the bus forever
// ends by calling End or Cancel instead.
type Runner struct {
	events.Registry

	name    string
	msgBus  *bus.Bus
	reg     *registry.WorkerRegistry
	forceGC bool

	handleInterrupt bool
	handleTerminate bool

	mu sync.Mutex
	// entries are the workers registered on the runner, by name, and order
	// keeps them in the order they were added.
	entries map[string]*workerEntry
	order   []string
	// knownRunners are the other runners this one has heard from.
	knownRunners map[string]struct{}
	started      bool
	autoEnd      bool
	// cancelReason is why the runner was canceled, carried on the messages the
	// exit path sends out.
	cancelReason string
	// shuttingDown is closed when the runner is asked to stop, whichever way.
	shuttingDown chan struct{}
	shutdownOnce sync.Once
}

// NewRunner builds a runner.
func NewRunner(cfg RunnerConfig) *Runner {
	r := &Runner{
		name:            cfg.Name,
		msgBus:          cfg.Bus,
		forceGC:         cfg.ForceGC,
		handleInterrupt: cfg.HandleInterrupt == nil || *cfg.HandleInterrupt,
		handleTerminate: cfg.HandleTerminate,
		entries:         make(map[string]*workerEntry),
		knownRunners:    make(map[string]struct{}),
		shuttingDown:    make(chan struct{}),
		autoEnd:         true,
	}
	if r.name == "" {
		r.name = "runner-" + uuid.NewString()[:8]
	}
	if r.msgBus == nil {
		r.msgBus = bus.NewAsyncQueueBus().Bus
	}
	r.reg = registry.New(r.name)

	r.Register(EventRunnerReady, false)
	r.Register(EventRunnerError, false)

	return r
}

// Name identifies the runner on the bus.
func (r *Runner) Name() string { return r.name }

// Bus is the bus this runner hosts and shares with its workers.
func (r *Runner) Bus() *bus.Bus { return r.msgBus }

// WorkerRegistry is the registry this runner owns and shares with its workers.
func (r *Runner) WorkerRegistry() *registry.WorkerRegistry { return r.reg }

// AddWorkers registers workers on the runner, attaching each to the bus and
// registry and starting it.
//
// A worker added before Run is started when the runner sets itself up; one
// added while it is running starts there and then. A name already registered
// is reported and skipped.
func (r *Runner) AddWorkers(ctx context.Context, added ...Worker) {
	for _, worker := range added {
		r.mu.Lock()
		if _, ok := r.entries[worker.Name()]; ok {
			r.mu.Unlock()
			slog.Error("worker of that name is already on the runner, skipping",
				"runner", r.name, "worker", worker.Name())
			continue
		}
		entry := &workerEntry{worker: worker}
		r.entries[worker.Name()] = entry
		r.order = append(r.order, worker.Name())
		startNow := r.started
		r.mu.Unlock()

		// Attached, and so subscribed, before anything else happens: a worker
		// added later has to be listening before the ones already there send
		// their first messages.
		worker.Attach(ctx, r.reg, r.msgBus)
		r.reg.Watch(ctx, worker.Name(), r.name, r.onLocalWorkerReady)
		slog.Debug("worker added to the runner", "runner", r.name, "worker", worker.Name())

		if startNow {
			r.startWorker(ctx, entry)
		}
	}
}

// Run starts the workers added to the runner and blocks until it stops.
func (r *Runner) Run(ctx context.Context, opts RunOptions) error {
	slog.Debug("runner started", "runner", r.name)

	r.mu.Lock()
	r.autoEnd = opts.AutoEnd == nil || *opts.AutoEnd
	r.mu.Unlock()

	stopSignals := r.setupSession(ctx)
	defer stopSignals()

	r.Call(ctx, EventRunnerReady, r)

	select {
	case <-r.shuttingDown:
	case <-ctx.Done():
	}

	// From here the runner is tearing down, on a context of its own: the
	// workers still have to be told to stop and waited for, and the context
	// that ended may be the reason we are here.
	stopCtx := context.WithoutCancel(ctx)

	r.stopRemainingWorkers(stopCtx)
	r.Registry.Cleanup(stopCtx) // the event registry, not the worker one
	r.msgBus.Stop()

	r.mu.Lock()
	r.started = false
	r.mu.Unlock()

	if r.forceGC {
		runtime.GC()
	}

	slog.Debug("runner finished", "runner", r.name)
	return nil
}

// StopWhenDone asks every root worker that can to finish what it is doing and
// then stop.
func (r *Runner) StopWhenDone() {
	slog.Debug("runner scheduled to stop once its workers are done", "runner", r.name)
	for _, entry := range r.rootEntries() {
		if stoppable, ok := entry.worker.(StoppableWhenDone); ok {
			stoppable.StopWhenDone()
		}
	}
}

// End asks every worker that has not finished to stop gracefully. Calling it
// again does nothing.
func (r *Runner) End(ctx context.Context, reason string) {
	if !r.beginShutdown() {
		return
	}
	slog.Debug("runner ending gracefully", "runner", r.name, "reason", reason)
	r.finishRunningWorkers(ctx, func(target string) bus.Message {
		m := &bus.EndWorkerMessage{Reason: reason}
		m.From = r.name
		m.To = target
		return m
	})
}

// Cancel stops every worker at once. It records why and signals the shutdown,
// which the run answers by canceling each worker still going and waiting for
// it. Calling it again does nothing.
//
// The messages go out from that one exit path rather than from here, so a worker
// is told once rather than twice on an ordinary shutdown, and the caller's reason
// travels with them.
func (r *Runner) Cancel(_ context.Context, reason string) {
	if !r.beginShutdown() {
		return
	}
	slog.Debug("runner canceling", "runner", r.name, "reason", reason)
	r.mu.Lock()
	r.cancelReason = reason
	r.mu.Unlock()
}

// finishRunningWorkers asks each root worker that has not finished to wind down,
// and reports the entries that were messaged.
//
// Both messages it is used with end a worker, gracefully or not. A worker whose
// run has returned is skipped; one that has not started yet has not finished
// either, so it is still told.
//
// Sending is all this does. The workers finish in their own time, which is what
// the returned entries are for.
func (r *Runner) finishRunningWorkers(
	ctx context.Context, build func(target string) bus.Message,
) []*workerEntry {
	var messaged []*workerEntry
	for _, entry := range r.allEntries() {
		if entry.finished() {
			continue
		}
		messaged = append(messaged, entry)
		if entry.worker.base().Parent() != "" {
			continue
		}
		r.msgBus.Send(ctx, build(entry.worker.Name()))
	}
	return messaged
}

// OnBusMessage handles the messages that are the runner's business rather than
// any one worker's.
func (r *Runner) OnBusMessage(ctx context.Context, m bus.Message) {
	if m.Source() == r.name {
		return
	}
	switch msg := m.(type) {
	case *bus.EndMessage:
		r.End(ctx, msg.Reason)
	case *bus.CancelMessage:
		r.Cancel(ctx, msg.Reason)
	case *bus.AddWorkerMessage:
		if worker, ok := msg.Worker.(Worker); ok {
			r.AddWorkers(ctx, worker)
		}
	case *bus.WorkerRegistryMessage:
		r.handleWorkerRegistry(ctx, msg)
	}
}

// setupSession is the once-per-run setup: the signal handlers, the bus, and the
// workers added so far. It returns the function that releases the signal
// handlers.
func (r *Runner) setupSession(ctx context.Context) func() {
	r.mu.Lock()
	if r.started {
		r.mu.Unlock()
		return func() {}
	}
	r.mu.Unlock()

	stopSignals := r.setupSignals(ctx)

	r.msgBus.Subscribe(r)
	r.msgBus.Start(ctx)

	r.mu.Lock()
	entries := r.entriesLocked()
	r.started = true
	r.mu.Unlock()

	for _, entry := range entries {
		r.startWorker(ctx, entry)
	}

	return stopSignals
}

// setupSignals cancels the runner on the signals it was asked to handle.
func (r *Runner) setupSignals(ctx context.Context) func() {
	var wanted []os.Signal
	if r.handleInterrupt {
		wanted = append(wanted, os.Interrupt)
	}
	if r.handleTerminate {
		wanted = append(wanted, syscall.SIGTERM)
	}
	if len(wanted) == 0 {
		return func() {}
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, wanted...)
	done := make(chan struct{})

	go func() {
		select {
		case <-signals:
			slog.Warn("interruption detected, canceling the runner", "runner", r.name)
			r.Cancel(ctx, "interrupt signal")
		case <-done:
		}
	}()

	return func() {
		signal.Stop(signals)
		close(done)
	}
}

// startWorker runs one worker on a goroutine of its own.
func (r *Runner) startWorker(ctx context.Context, entry *workerEntry) {
	slog.Debug("runner starting a worker", "runner", r.name, "worker", entry.worker.Name())

	done := make(chan struct{})
	r.mu.Lock()
	entry.done = done
	r.mu.Unlock()

	go func() {
		defer close(done)
		if err := entry.worker.Run(ctx); err != nil {
			slog.Error("worker stopped with an error",
				"runner", r.name, "worker", entry.worker.Name(), "error", err)
			r.Call(ctx, EventRunnerError, r, entry.worker.Name(), err)
		}
		r.maybeAutoEnd(entry)
	}()
}

// maybeAutoEnd ends the runner once the last root worker has finished, when it
// was asked to. The worker that just finished does not count itself: its
// goroutine is what is running this.
func (r *Runner) maybeAutoEnd(finished *workerEntry) {
	r.mu.Lock()
	if !r.autoEnd || finished.worker.base().Parent() != "" {
		r.mu.Unlock()
		return
	}
	for _, entry := range r.entriesLocked() {
		if entry == finished || entry.worker.base().Parent() != "" {
			continue
		}
		if entry.running() {
			r.mu.Unlock()
			return
		}
	}
	r.mu.Unlock()

	r.shutdownOnce.Do(func() { close(r.shuttingDown) })
}

// stopRemainingWorkers tells the root workers still going that the runner is
// leaving, and waits for them all.
func (r *Runner) stopRemainingWorkers(ctx context.Context) {
	r.mu.Lock()
	reason := r.cancelReason
	r.mu.Unlock()
	if reason == "" {
		reason = "runner exiting"
	}

	for _, entry := range r.finishRunningWorkers(ctx, func(target string) bus.Message {
		m := &bus.CancelWorkerMessage{Reason: reason}
		m.From = r.name
		m.To = target
		return m
	}) {
		if entry.done == nil {
			// Never started, so there is nothing to wait for.
			continue
		}
		select {
		case <-entry.done:
		case <-ctx.Done():
			return
		}
	}
}

// onLocalWorkerReady announces this runner's workers when one of its own root
// workers becomes ready, so the other runners on the bus learn of it.
func (r *Runner) onLocalWorkerReady(ctx context.Context, data registry.WorkerReadyData) {
	if data.Runner != r.name {
		return
	}
	r.mu.Lock()
	entry, ok := r.entries[data.WorkerName]
	r.mu.Unlock()
	if !ok || entry.worker.base().Parent() != "" {
		return
	}
	r.sendRegistry(ctx)
}

// sendRegistry broadcasts the workers this runner knows about.
func (r *Runner) sendRegistry(ctx context.Context) {
	entries := r.allEntries()
	if len(entries) == 0 {
		return
	}

	workers := make([]registry.WorkerRegistryEntry, 0, len(entries))
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		base := entry.worker.base()
		workers = append(workers, registry.WorkerRegistryEntry{
			Name:      entry.worker.Name(),
			Parent:    base.Parent(),
			Active:    base.Active(),
			Bridged:   entry.worker.Bridged(),
			StartedAt: base.StartedAt(),
		})
		names = append(names, entry.worker.Name())
	}

	slog.Debug("runner broadcasting its registry", "runner", r.name, "workers", names)
	m := &bus.WorkerRegistryMessage{Runner: r.name, Workers: workers}
	m.From = r.name
	r.msgBus.Send(ctx, m)
}

// handleWorkerRegistry takes another runner's snapshot, and answers with this
// runner's own the first time it hears from one.
func (r *Runner) handleWorkerRegistry(ctx context.Context, m *bus.WorkerRegistryMessage) {
	names := make([]string, 0, len(m.Workers))
	for _, entry := range m.Workers {
		names = append(names, entry.Name)
	}
	slog.Debug("runner received a registry", "runner", r.name, "from", m.Runner, "workers", names)

	for _, entry := range m.Workers {
		r.reg.Register(ctx, registry.WorkerReadyData{WorkerName: entry.Name, Runner: m.Runner})
	}

	r.mu.Lock()
	_, known := r.knownRunners[m.Runner]
	if !known {
		r.knownRunners[m.Runner] = struct{}{}
	}
	r.mu.Unlock()

	if !known {
		slog.Debug("runner heard from a new runner, answering with its own registry",
			"runner", r.name, "other", m.Runner)
		r.sendRegistry(ctx)
	}
}

// beginShutdown reports whether this call is the one that starts the shutdown,
// which is what makes ending and canceling do nothing the second time.
func (r *Runner) beginShutdown() bool {
	select {
	case <-r.shuttingDown:
		return false
	default:
	}
	first := false
	r.shutdownOnce.Do(func() {
		first = true
		close(r.shuttingDown)
	})
	return first
}

// allEntries are the registered workers, in the order they were added.
func (r *Runner) allEntries() []*workerEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.entriesLocked()
}

// entriesLocked is allEntries with the lock already held.
func (r *Runner) entriesLocked() []*workerEntry {
	out := make([]*workerEntry, 0, len(r.order))
	for _, name := range r.order {
		if entry, ok := r.entries[name]; ok {
			out = append(out, entry)
		}
	}
	return out
}

// rootEntries are the registered workers with no parent, which are the ones the
// runner addresses directly; a child is its parent's business.
func (r *Runner) rootEntries() []*workerEntry {
	var out []*workerEntry
	for _, entry := range r.allEntries() {
		if entry.worker.base().Parent() == "" {
			out = append(out, entry)
		}
	}
	return out
}

// String names the runner in a message.
func (r *Runner) String() string { return fmt.Sprintf("runner %q", r.name) }
