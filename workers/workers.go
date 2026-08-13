// Package workers is the base every worker is built on: the thing that
// connects to a bus, registers itself so others can find it, is activated and
// deactivated, and exchanges long-running jobs with the other workers.
//
// A worker is a participant in a session rather than a step in a pipeline. One
// may drive a pipeline of its own, another may only listen on the bus, and a
// parent may add children whose lifetime it manages. What they share is the
// protocol in this package: become ready, be told to start or stop, ask each
// other for work and report back on it.
//
// A worker embeds *Base and hands itself to New, which is what lets the base
// reach the worker's own versions of the hooks it calls:
//
//	type Greeter struct{ *workers.Base }
//
//	func NewGreeter() *Greeter {
//	    g := &Greeter{}
//	    g.Base = workers.New(workers.Config{Name: "greeter"}, g)
//	    return g
//	}
//
//	func (g *Greeter) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
//	    _ = g.SendJobResponse(ctx, m.JobID, map[string]any{"hello": true}, workers.JobResponseOptions{})
//	}
//
// A worker that overrides a hook the base calls should call the base's version
// too, the way Greeter would call g.Base.OnJobRequest, unless it means to
// replace the behavior entirely.
package workers

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/utils/events"
)

// The events a worker raises. Attach handlers with the Add method it inherits
// from the event registry.
const (
	// EventActivated fires when this worker is activated, with the arguments it
	// was activated with.
	EventActivated = "on_activated"
	// EventDeactivated fires when this worker is deactivated.
	EventDeactivated = "on_deactivated"
	// EventBusMessage fires for every bus message, after the built-in handling
	// of it.
	EventBusMessage = "on_bus_message"
	// EventJobRequest fires when this worker is asked to do a job.
	EventJobRequest = "on_job_request"
	// EventJobResponse fires when a worker reports how its job ended.
	EventJobResponse = "on_job_response"
	// EventJobUpdate fires when a worker reports progress on a job.
	EventJobUpdate = "on_job_update"
	// EventJobUpdateRequested fires when the requester asks for progress on a
	// job this worker is doing.
	EventJobUpdateRequested = "on_job_update_requested"
	// EventJobCompleted fires when every worker in a group has responded.
	EventJobCompleted = "on_job_completed"
	// EventJobError fires when a worker failed and the group was called off.
	EventJobError = "on_job_error"
	// EventJobStreamStart fires when a worker opens a stream of results.
	EventJobStreamStart = "on_job_stream_start"
	// EventJobStreamData fires for each item of a worker's result stream.
	EventJobStreamData = "on_job_stream_data"
	// EventJobStreamEnd fires when a worker closes its result stream.
	EventJobStreamEnd = "on_job_stream_end"
	// EventJobCancelled fires when a job this worker is doing is called off.
	EventJobCancelled = "on_job_cancelled" //nolint:misspell // the event name the protocol uses
	// EventWorkerReady fires when another worker becomes ready.
	EventWorkerReady = "on_worker_ready"
	// EventWorkerFailed fires when a child worker reports an error.
	EventWorkerFailed = "on_worker_failed"
)

// Worker is a worker as the base sees it: the hooks the base calls on the
// worker it belongs to, so a worker's own version of one is reached rather than
// the base's.
//
// Only a type embedding *Base satisfies it, which is the point: the hooks all
// have a default, and a worker overrides the ones it cares about.
type Worker interface {
	bus.Subscriber

	// Bridged reports whether the worker is bridged onto the bus, which is
	// announced when it becomes ready.
	Bridged() bool
	// Attach connects the worker to the registry and bus its runner provides.
	Attach(ctx context.Context, reg *registry.WorkerRegistry, b *bus.Bus)
	// Run runs the worker until it finishes.
	Run(ctx context.Context) error
	// Stop cleans up and marks the worker finished.
	Stop(ctx context.Context)
	// Wait blocks until the worker finishes, or ctx ends.
	Wait(ctx context.Context)
	// Cleanup releases the worker and stops it.
	Cleanup(ctx context.Context)

	// OnActivated is called when this worker is activated.
	OnActivated(ctx context.Context, args map[string]any)
	// OnDeactivated is called when this worker is deactivated.
	OnDeactivated(ctx context.Context)
	// OnWorkerReady is called when another worker becomes ready.
	OnWorkerReady(ctx context.Context, data registry.WorkerReadyData)
	// OnWorkerFailed is called when a child worker reports an error.
	OnWorkerFailed(ctx context.Context, data registry.WorkerErrorData)
	// OnJobRequest is called when this worker is asked to do a job.
	OnJobRequest(ctx context.Context, m *bus.JobRequestMessage)
	// OnJobResponse is called when a worker reports how its job ended.
	OnJobResponse(ctx context.Context, m bus.JobResponse)
	// OnJobUpdate is called when a worker reports progress on a job.
	OnJobUpdate(ctx context.Context, m bus.JobUpdate)
	// OnJobUpdateRequested is called when the requester asks for progress.
	OnJobUpdateRequested(ctx context.Context, m *bus.JobUpdateRequestMessage)
	// OnJobCompleted is called when every worker in a group has responded.
	OnJobCompleted(ctx context.Context, result jobcontext.GroupResponse)
	// OnJobError is called when a worker failed and the group was called off.
	OnJobError(ctx context.Context, m bus.JobResponse)
	// OnJobStreamStart is called when a worker opens a stream of results.
	OnJobStreamStart(ctx context.Context, m *bus.JobStreamStartMessage)
	// OnJobStreamData is called for each item of a worker's result stream.
	OnJobStreamData(ctx context.Context, m *bus.JobStreamDataMessage)
	// OnJobStreamEnd is called when a worker closes its result stream.
	OnJobStreamEnd(ctx context.Context, m *bus.JobStreamEndMessage)
	// OnJobCancelled is called when a job this worker is doing is called off.
	OnJobCancelled(ctx context.Context, m *bus.JobCancelMessage)

	// HandleWorkerEnd is called when this worker is asked to end gracefully.
	// The default passes the end to the children and stops; a worker with a
	// runtime of its own drives that runtime's shutdown instead, so that it
	// finishes at the right moment.
	HandleWorkerEnd(ctx context.Context, m *bus.EndWorkerMessage)
	// HandleWorkerCancel is called when this worker is asked to stop at once.
	// The default passes the cancel to the children and stops; see
	// HandleWorkerEnd.
	HandleWorkerCancel(ctx context.Context, m *bus.CancelWorkerMessage)

	// base is the embedded Base, which is how a parent reaches a child's
	// lifecycle. It is unexported so that only a worker built on Base can be
	// one.
	base() *Base
}

// ActivationArgs is what a worker is handed when it is activated. A worker with
// arguments of its own defines a type for them, embeds BaseActivationArgs and
// puts its own fields in the map alongside.
type ActivationArgs interface {
	// ToMap is the arguments as they travel on the bus, leaving out what is
	// unset.
	ToMap() map[string]any
}

// BaseActivationArgs is the part of the activation arguments every worker
// understands.
type BaseActivationArgs struct {
	// Metadata is structured data for the worker being activated, and may be
	// nil.
	Metadata map[string]any
}

// ToMap implements ActivationArgs.
func (a BaseActivationArgs) ToMap() map[string]any {
	m := make(map[string]any, 1)
	if a.Metadata != nil {
		m["metadata"] = a.Metadata
	}
	return m
}

// BaseActivationArgsFrom reads the part every worker understands out of the
// arguments a worker was activated with, ignoring anything else in them.
func BaseActivationArgsFrom(args map[string]any) BaseActivationArgs {
	var a BaseActivationArgs
	if md, ok := args["metadata"].(map[string]any); ok {
		a.Metadata = md
	}
	return a
}

// Config configures a worker.
type Config struct {
	// Name is what other workers address this one by, and must be unique among
	// them. Empty names it after its type, which suits a worker taking no part
	// in worker-to-worker messaging.
	Name string
	// Active reports whether the worker starts active; nil defaults to true.
	Active *bool
}

// ActivateOptions are the choices activating another worker offers.
type ActivateOptions struct {
	// Args are handed to the target worker's OnActivated, and may be nil.
	Args ActivationArgs
	// DeactivateSelf deactivates this worker before activating the target, so
	// the two are never briefly active together.
	DeactivateSelf bool
}

// nameCounter numbers the workers that were not given a name.
//
//nolint:gochecknoglobals // one counter for the process, as the processors have
var nameCounter atomic.Uint64

// Base is the worker every worker is built on. See the package documentation
// for how to build one.
//
// It is safe for concurrent use: the bus hands a worker its system messages and
// its data messages on separate goroutines, so a cancel arrives while the work
// it is calling off is still being handled.
type Base struct {
	events.Registry

	// self is the worker this base belongs to, so a hook reaches the worker's
	// own version rather than the base's.
	self Worker
	name string

	mu sync.Mutex
	// msgBus is the bus the worker is attached to, nil until it is.
	msgBus *bus.Bus
	// reg is the shared registry, nil until the worker is attached.
	reg *registry.WorkerRegistry

	// active, pendingActivation and activationArgs are the activation state. An
	// activation that arrives before the worker starts is held pending until it
	// does.
	active            bool
	pendingActivation bool
	activationArgs    map[string]any

	// parent is the name of the worker that added this one, empty for a root
	// worker, and children are the workers it added itself.
	parent   string
	children []Worker

	startedAt  float64
	finished   chan struct{}
	finishOnce sync.Once

	// activeJobs are the job requests this worker is working on, by job id, so
	// several can be in flight at once.
	activeJobs map[string]*bus.JobRequestMessage
	// jobHandlers are the running handlers, by job id, so one can be called off
	// on its own.
	jobHandlers map[string]*runningJob
	// groups are the job groups this worker launched, by job id.
	groups map[string]*jobcontext.Group
	// handlers are the named job handlers this worker declared.
	handlers map[string]jobHandler
	// sequential holds, per job name, the turn the next sequential request
	// waits for. See runJobHandler.
	sequential map[string]chan struct{}
	// readyHandlers are the per-worker ready handlers this worker declared.
	readyHandlers map[string]ReadyHandler
	// readySeq numbers the waits for a worker to become ready, so two waits for
	// the same worker are two watches rather than one.
	readySeq uint64
}

// New builds a worker base. self is the worker embedding it, which is how the
// base reaches that worker's own hooks; see the package documentation.
func New(cfg Config, self Worker) *Base {
	w := &Base{
		self:          self,
		name:          cfg.Name,
		active:        cfg.Active == nil || *cfg.Active,
		finished:      make(chan struct{}),
		activeJobs:    make(map[string]*bus.JobRequestMessage),
		jobHandlers:   make(map[string]*runningJob),
		groups:        make(map[string]*jobcontext.Group),
		handlers:      make(map[string]jobHandler),
		sequential:    make(map[string]chan struct{}),
		readyHandlers: make(map[string]ReadyHandler),
	}
	w.pendingActivation = w.active
	if w.name == "" {
		w.name = fmt.Sprintf("%s#%d", typeNameOf(self), nameCounter.Add(1))
	}
	if self == nil {
		w.self = w
	}

	w.Register(EventActivated, false)
	w.Register(EventDeactivated, false)
	w.Register(EventBusMessage, false)
	w.Register(EventJobRequest, false)
	w.Register(EventJobResponse, false)
	w.Register(EventJobUpdate, false)
	w.Register(EventJobUpdateRequested, false)
	w.Register(EventJobCompleted, false)
	w.Register(EventJobError, false)
	w.Register(EventJobStreamStart, false)
	w.Register(EventJobStreamData, false)
	w.Register(EventJobStreamEnd, false)
	w.Register(EventJobCancelled, false)
	w.Register(EventWorkerReady, false)
	w.Register(EventWorkerFailed, false)

	return w
}

// typeNameOf names a worker after its own type, for one built without a name.
func typeNameOf(self Worker) string {
	if self == nil {
		return "Worker"
	}
	t := reflect.TypeOf(self)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Name() == "" {
		return "Worker"
	}
	return t.Name()
}

// base implements Worker.
func (w *Base) base() *Base { return w }

// Name is what other workers address this one by, and identifies it on the bus.
func (w *Base) Name() string { return w.name }

// Bridged reports whether the worker is bridged onto the bus. A worker that
// wraps its pipeline in bus edges overrides it.
func (w *Base) Bridged() bool { return false }

// Bus is the bus the worker is attached to, and is nil until Attach.
func (w *Base) Bus() *bus.Bus {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.msgBus
}

// WorkerRegistry is the shared worker registry, and is nil until Attach.
func (w *Base) WorkerRegistry() *registry.WorkerRegistry {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.reg
}

// Active reports whether the worker is currently active.
func (w *Base) Active() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.active
}

// ActivationArgs are the arguments of the most recent activation, and are nil
// while the worker is inactive.
func (w *Base) ActivationArgs() map[string]any {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.activationArgs
}

// Parent is the name of the worker that added this one, empty for a root
// worker.
func (w *Base) Parent() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.parent
}

// Children are the workers this one added.
func (w *Base) Children() []Worker {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]Worker(nil), w.children...)
}

// StartedAt is when the worker became ready, as a Unix timestamp, and is zero
// until it has.
func (w *Base) StartedAt() float64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.startedAt
}

// Attach connects the worker to the registry and bus its runner provides, and
// subscribes it. It is called before the worker runs, so a worker added later
// is listening before any of them sends its first message.
func (w *Base) Attach(_ context.Context, reg *registry.WorkerRegistry, b *bus.Bus) {
	w.mu.Lock()
	w.reg = reg
	w.msgBus = b
	w.mu.Unlock()
	if b != nil {
		b.Subscribe(w.self)
	}
}

// Run runs the worker until it finishes.
//
// This is the plain bus-only worker: it starts, then waits to be stopped. A
// worker with a runtime of its own overrides it.
func (w *Base) Run(ctx context.Context) error {
	w.Start(ctx)
	select {
	case <-w.finished:
	case <-ctx.Done():
	}
	// Stopped on a context that is still live, so a worker cut short still gets
	// to answer the jobs it was holding rather than leaving its requesters
	// waiting on a worker that has gone.
	w.self.Stop(context.WithoutCancel(ctx))
	return nil
}

// Start marks the worker started, registers it as ready, activates it if it was
// built active, and watches the workers it declared ready handlers for.
func (w *Base) Start(ctx context.Context) {
	w.mu.Lock()
	w.startedAt = float64(time.Now().UnixNano()) / float64(time.Second)
	w.mu.Unlock()

	w.registerReady(ctx)
	w.maybeActivate(ctx)
	w.watchDeclaredWorkers(ctx)
}

// Stop calls off everything the worker had running and marks it finished.
//
// Every job group it launched is called off, and every job request it was still
// working on is answered as canceled, so nobody is left waiting on a worker
// that has stopped.
func (w *Base) Stop(ctx context.Context) {
	for _, jobID := range w.groupIDs() {
		w.CancelGroup(ctx, jobID, fmt.Sprintf("worker %q stopped", w.name))
	}
	for _, jobID := range w.activeJobIDs() {
		if err := w.SendJobResponse(ctx, jobID, nil, JobResponseOptions{
			Status: jobcontext.JobCancelled,
		}); err != nil {
			slog.Debug("could not report a job canceled while stopping",
				"worker", w.name, "job", jobID, "error", err)
		}
	}
	// The handlers still running are stopped here because the worker owns the
	// goroutines it started for them; nothing else would.
	for _, job := range w.takeRunningJobs() {
		job.stop()
	}
	w.finishOnce.Do(func() { close(w.finished) })
}

// Cleanup releases the worker and stops it. It waits for the handlers of the
// worker's own events before stopping, so none is left running past the worker
// it was watching.
func (w *Base) Cleanup(ctx context.Context) {
	w.Registry.Cleanup(ctx) // the event registry, not the worker one
	w.self.Stop(ctx)
}

// Wait blocks until the worker finishes, or ctx ends.
func (w *Base) Wait(ctx context.Context) {
	select {
	case <-w.finished:
	case <-ctx.Done():
	}
}

// Finished is closed when the worker finishes.
func (w *Base) Finished() <-chan struct{} { return w.finished }

// End asks for the session to end gracefully.
func (w *Base) End(ctx context.Context, reason string) {
	m := &bus.EndMessage{Reason: reason}
	m.From = w.name
	w.SendBusMessage(ctx, m)
}

// Cancel asks for every worker to stop at once.
func (w *Base) Cancel(ctx context.Context, reason string) {
	m := &bus.CancelMessage{Reason: reason}
	m.From = w.name
	w.SendBusMessage(ctx, m)
}

// SendBusMessage puts a message on the bus. It is a no-op for a worker that is
// not attached to one.
func (w *Base) SendBusMessage(ctx context.Context, m bus.Message) {
	b := w.Bus()
	if b == nil {
		return
	}
	b.Send(ctx, m)
}

// SendBusErrorMessage reports that this worker failed.
//
// A child reports to its parent, in this process only. A root worker reports to
// everyone, over the network.
func (w *Base) SendBusErrorMessage(ctx context.Context, workerErr string) {
	if w.Parent() != "" {
		m := &bus.WorkerLocalErrorMessage{Error: workerErr}
		m.From = w.name
		w.SendBusMessage(ctx, m)
		return
	}
	m := &bus.WorkerErrorMessage{Error: workerErr}
	m.From = w.name
	w.SendBusMessage(ctx, m)
}

// AddWorkers puts workers under this one as its children, and watches each, so
// OnWorkerReady fires as each becomes ready.
//
// A parent manages its children's lifetime: ending or canceling it ends or
// cancels them. A worker that already has a parent is left where it is.
func (w *Base) AddWorkers(ctx context.Context, children ...Worker) {
	w.addWorkers(ctx, true, children)
}

// AddWorkersUnwatched is AddWorkers without watching the children, for a parent
// that does not need to be told when they become ready. It can still call
// WatchWorkers later.
func (w *Base) AddWorkersUnwatched(ctx context.Context, children ...Worker) {
	w.addWorkers(ctx, false, children)
}

func (w *Base) addWorkers(ctx context.Context, watch bool, children []Worker) {
	var added []string
	for _, child := range children {
		cb := child.base()
		cb.mu.Lock()
		if cb.parent != "" {
			parent := cb.parent
			cb.mu.Unlock()
			slog.Warn("worker already has a parent, leaving it there",
				"worker", child.Name(), "parent", parent)
			continue
		}
		cb.parent = w.name
		cb.mu.Unlock()

		w.mu.Lock()
		w.children = append(w.children, child)
		w.mu.Unlock()

		added = append(added, child.Name())
		m := &bus.AddWorkerMessage{Worker: child}
		m.From = w.name
		w.SendBusMessage(ctx, m)
	}
	if watch && len(added) > 0 {
		w.WatchWorkers(ctx, added...)
	}
}

// ActivateWorker activates another worker by name, which calls that worker's
// OnActivated with the arguments given.
func (w *Base) ActivateWorker(ctx context.Context, workerName string, opts ActivateOptions) {
	if opts.DeactivateSelf {
		w.mu.Lock()
		wasActive := w.active
		if wasActive {
			// Deactivated here rather than on the bus message coming back, so
			// this worker and the target are never both active at once.
			w.active = false
		}
		w.mu.Unlock()
		if wasActive {
			w.DeactivateWorker(ctx, w.name)
		}
	}

	var args map[string]any
	if opts.Args != nil {
		args = opts.Args.ToMap()
	}
	m := &bus.ActivateWorkerMessage{Args: args}
	m.From = w.name
	m.To = workerName
	w.SendBusMessage(ctx, m)
}

// DeactivateWorker deactivates another worker by name, which calls that
// worker's OnDeactivated.
func (w *Base) DeactivateWorker(ctx context.Context, workerName string) {
	m := &bus.DeactivateWorkerMessage{}
	m.From = w.name
	m.To = workerName
	w.SendBusMessage(ctx, m)
}

// WatchWorkers asks to be told when the named workers register. A worker that
// has already registered is reported straight away.
func (w *Base) WatchWorkers(ctx context.Context, workerNames ...string) {
	reg := w.WorkerRegistry()
	if reg == nil {
		return
	}
	for _, workerName := range workerNames {
		slog.Debug("watching for a worker", "worker", w.name, "watched", workerName)
		// Keyed by this worker, so watching the same one twice, by adding it as
		// a child and by declaring a ready handler for it, is a single watch.
		reg.Watch(ctx, workerName, w.name, w.onWatchedWorkerReady)
	}
}

// OnActivated is called when this worker is activated. The default does
// nothing.
func (w *Base) OnActivated(context.Context, map[string]any) {}

// OnDeactivated is called when this worker is deactivated. The default does
// nothing.
func (w *Base) OnDeactivated(context.Context) {}

// OnWorkerReady is called when another worker becomes ready.
//
// It fires for a local root worker on its own, for a child only on the parent
// that added it, and for a remote worker only when it is watched. The default
// does nothing.
func (w *Base) OnWorkerReady(context.Context, registry.WorkerReadyData) {}

// OnWorkerFailed is called when a child worker reports an error. The default
// does nothing.
func (w *Base) OnWorkerFailed(context.Context, registry.WorkerErrorData) {}

// OnBusMessage handles one bus message: the built-in lifecycle and job
// handling, and then the on_bus_message event.
//
// A worker with message types of its own overrides it, calls this one, and
// handles the rest itself.
func (w *Base) OnBusMessage(ctx context.Context, m bus.Message) {
	// Frames between workers are the business of the bridges, not of the base.
	if _, ok := m.(*bus.FrameMessage); ok {
		return
	}

	// A message addressed to another worker is not this worker's business.
	if m.Target() != "" && m.Target() != w.name {
		return
	}

	if !w.handleLifecycleMessage(ctx, m) {
		w.handleJobMessage(ctx, m)
	}

	w.Call(ctx, EventBusMessage, w.self, m)
}

// handleLifecycleMessage handles the messages about a worker's own life, and
// reports whether the message was one of them.
func (w *Base) handleLifecycleMessage(ctx context.Context, m bus.Message) bool {
	switch msg := m.(type) {
	case *bus.WorkerErrorMessage:
		w.handleWorkerError(ctx, msg.Source(), msg.Error)
	case *bus.WorkerLocalErrorMessage:
		w.handleWorkerError(ctx, msg.Source(), msg.Error)
	case *bus.ActivateWorkerMessage:
		w.handleWorkerActivate(ctx, msg)
	case *bus.DeactivateWorkerMessage:
		w.handleWorkerDeactivate(ctx)
	case *bus.EndWorkerMessage:
		w.self.HandleWorkerEnd(ctx, msg)
	case *bus.CancelWorkerMessage:
		w.self.HandleWorkerCancel(ctx, msg)
	default:
		return false
	}
	return true
}

// handleJobMessage handles the messages about a job, and reports whether the
// message was one of them.
func (w *Base) handleJobMessage(ctx context.Context, m bus.Message) bool {
	switch msg := m.(type) {
	case *bus.JobRequestMessage:
		w.handleJobRequest(ctx, msg)
	case bus.JobResponse:
		w.handleJobResponse(ctx, msg)
	case bus.JobUpdate:
		w.handleJobUpdate(ctx, msg)
	case *bus.JobUpdateRequestMessage:
		w.handleJobUpdateRequest(ctx, msg)
	case *bus.JobCancelMessage:
		w.handleJobCancel(ctx, msg)
	case *bus.JobStreamStartMessage:
		w.handleJobStreamStart(ctx, msg)
	case *bus.JobStreamDataMessage:
		w.handleJobStreamData(ctx, msg)
	case *bus.JobStreamEndMessage:
		w.handleJobStreamEnd(ctx, msg)
	default:
		return false
	}
	return true
}

// registerReady announces the worker on the bus and registers it, which tells
// whoever was watching for it.
func (w *Base) registerReady(ctx context.Context) {
	reg := w.WorkerRegistry()
	if reg == nil {
		return
	}

	w.mu.Lock()
	parent, active, startedAt := w.parent, w.active, w.startedAt
	w.mu.Unlock()

	// Announced before registering, because registering runs the watchers and
	// those may send messages of their own; sending this one first keeps the
	// order an observer sees the right way round.
	m := &bus.WorkerReadyMessage{
		Runner:    reg.RunnerName(),
		Parent:    parent,
		Active:    active,
		Bridged:   w.self.Bridged(),
		StartedAt: startedAt,
	}
	m.From = w.name
	w.SendBusMessage(ctx, m)

	reg.Register(ctx, registry.WorkerReadyData{
		WorkerName: w.name,
		Runner:     reg.RunnerName(),
	})
}

// maybeActivate activates the worker when an activation is pending and it has
// started.
func (w *Base) maybeActivate(ctx context.Context) {
	w.mu.Lock()
	if w.startedAt == 0 || !w.pendingActivation {
		w.mu.Unlock()
		return
	}
	w.active = true
	w.pendingActivation = false
	args := w.activationArgs
	w.mu.Unlock()

	slog.Debug("worker activated", "worker", w.name)
	w.self.OnActivated(ctx, args)
	w.Call(ctx, EventActivated, w.self, args)
}

// watchDeclaredWorkers watches every worker this one declared a ready handler
// for.
func (w *Base) watchDeclaredWorkers(ctx context.Context) {
	w.mu.Lock()
	names := make([]string, 0, len(w.readyHandlers))
	for name := range w.readyHandlers {
		names = append(names, name)
	}
	w.mu.Unlock()
	w.WatchWorkers(ctx, names...)
}

// onWatchedWorkerReady runs the handler declared for the worker that became
// ready, if there is one, and then the general hook.
func (w *Base) onWatchedWorkerReady(ctx context.Context, data registry.WorkerReadyData) {
	slog.Debug("watched worker ready", "worker", w.name, "ready", data.WorkerName)

	w.mu.Lock()
	handler := w.readyHandlers[data.WorkerName]
	w.mu.Unlock()
	if handler != nil {
		handler(ctx, data)
	}

	w.self.OnWorkerReady(ctx, data)
	w.Call(ctx, EventWorkerReady, w.self, data)
}

// handleWorkerError reports an error from one of this worker's children.
func (w *Base) handleWorkerError(ctx context.Context, source, workerErr string) {
	if !w.hasChild(source) {
		return
	}
	data := registry.WorkerErrorData{WorkerName: source, Error: workerErr}
	w.self.OnWorkerFailed(ctx, data)
	w.Call(ctx, EventWorkerFailed, w.self, data)
}

func (w *Base) hasChild(name string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, child := range w.children {
		if child.Name() == name {
			return true
		}
	}
	return false
}

// handleWorkerActivate holds the activation arguments and activates, or waits
// to, when the worker has not started yet.
func (w *Base) handleWorkerActivate(ctx context.Context, m *bus.ActivateWorkerMessage) {
	w.mu.Lock()
	w.activationArgs = m.Args
	w.pendingActivation = true
	w.mu.Unlock()
	w.maybeActivate(ctx)
}

// handleWorkerDeactivate deactivates this worker.
func (w *Base) handleWorkerDeactivate(ctx context.Context) {
	slog.Debug("worker deactivated", "worker", w.name)
	w.mu.Lock()
	w.active = false
	w.activationArgs = nil
	w.mu.Unlock()
	w.self.OnDeactivated(ctx)
	w.Call(ctx, EventDeactivated, w.self)
}

// HandleWorkerEnd passes the end on to the children, waits for them, and then
// stops. A worker with a runtime of its own overrides it, calls
// PropagateEndToChildren and drives its own shutdown, so that it finishes at
// the right moment.
func (w *Base) HandleWorkerEnd(ctx context.Context, m *bus.EndWorkerMessage) {
	slog.Debug("worker received end", "worker", w.name)
	w.PropagateEndToChildren(ctx, m.Reason)
	w.self.Stop(ctx)
}

// HandleWorkerCancel passes the cancel on to the children and stops. See
// HandleWorkerEnd.
func (w *Base) HandleWorkerCancel(ctx context.Context, m *bus.CancelWorkerMessage) {
	slog.Debug("worker received cancel", "worker", w.name)
	w.PropagateCancelToChildren(ctx, m.Reason)
	w.self.Stop(ctx)
}

// PropagateEndToChildren asks each child to end and waits for them all.
func (w *Base) PropagateEndToChildren(ctx context.Context, reason string) {
	children := w.Children()
	for _, child := range children {
		m := &bus.EndWorkerMessage{Reason: reason}
		m.From = w.name
		m.To = child.Name()
		w.SendBusMessage(ctx, m)
	}
	for _, child := range children {
		child.Wait(ctx)
	}
}

// PropagateCancelToChildren asks each child to stop at once.
func (w *Base) PropagateCancelToChildren(ctx context.Context, reason string) {
	for _, child := range w.Children() {
		m := &bus.CancelWorkerMessage{Reason: reason}
		m.From = w.name
		m.To = child.Name()
		w.SendBusMessage(ctx, m)
	}
}
