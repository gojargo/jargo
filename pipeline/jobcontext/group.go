package jobcontext

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
)

// The event kinds a worker reports while a job runs.
const (
	// EventUpdate is progress on work still running.
	EventUpdate = "update"
	// EventStreamStart opens a stream of results.
	EventStreamStart = "stream_start"
	// EventStreamData is one item of a result stream.
	EventStreamData = "stream_data"
	// EventStreamEnd closes a result stream.
	EventStreamEnd = "stream_end"
)

// Event is something a worker reported while a single-worker job was running.
type Event struct {
	// Type is the kind of event, one of the Event constants.
	Type string
	// Data is the payload, and may be nil.
	Data map[string]any
}

// GroupEvent is something one worker of a group reported while the group was
// running.
type GroupEvent struct {
	// Type is the kind of event, one of the Event constants.
	Type string
	// WorkerName is the worker that reported it.
	WorkerName string
	// Data is the payload, and may be nil.
	Data map[string]any
}

// GroupResponse is what a finished job group produced.
type GroupResponse struct {
	// JobID is the identifier every worker in the group shared.
	JobID string
	// Responses is what each worker returned, by worker name.
	Responses map[string]map[string]any
}

// ErrGroup marks a job group cut short because a worker failed or the group ran
// out of time. Unwrap it for the reason.
var ErrGroup = errors.New("job group stopped before it finished")

// Group tracks a set of workers asked to do a job together, and is what the
// worker running the group writes results into as they arrive.
//
// Its events reach a caller over a channel that is closed when the group
// finishes, so ranging over it ends exactly when the group does. Upstream marks
// the end by queueing a sentinel; a closed channel is the same thing said in
// Go, and it cannot be missed.
//
// It is safe for concurrent use.
type Group struct {
	// JobID is the identifier every worker in the group shares.
	JobID string
	// WorkerNames are the workers the job went to, each named once however many
	// times the caller listed it.
	WorkerNames []string
	// CancelOnError reports whether one worker failing calls off the rest.
	CancelOnError bool

	mu sync.Mutex
	// responses is what each worker returned so far, by worker name. A group cut
	// short keeps whatever arrived before it was.
	responses map[string]map[string]any
	done      chan struct{}
	closed    bool
	err       error
	events    chan GroupEvent
	// stopTimeout calls off the group's timeout, and is nil when it is running
	// without one.
	stopTimeout func()
}

// NewGroup builds a group for the named workers. A name listed twice is one
// worker, which is why the group holds a set: it answers once, and calling the
// group off tells it once.
func NewGroup(jobID string, workerNames []string, cancelOnError bool) *Group {
	names := make([]string, 0, len(workerNames))
	seen := make(map[string]struct{}, len(workerNames))
	for _, name := range workerNames {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	return &Group{
		JobID:         jobID,
		WorkerNames:   names,
		CancelOnError: cancelOnError,
		responses:     make(map[string]map[string]any),
		done:          make(chan struct{}),
		events:        make(chan GroupEvent, eventBuffer),
	}
}

// eventBuffer is how many events may be waiting for a caller that is not
// reading them. A caller need not read at all, so reporting must never block
// the worker doing the job; past the buffer the oldest events are dropped.
const eventBuffer = 64

// SetTimeout records how to call off the group's timeout. The worker running
// the group starts the timeout and hands the cancellation here, so completing
// in time can stop it.
func (g *Group) SetTimeout(stop func()) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.stopTimeout = stop
}

// HasTimeout reports whether the group is running under a timeout.
func (g *Group) HasTimeout() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.stopTimeout != nil
}

// CancelTimeout calls off the group's timeout, if it has one. It is a no-op
// otherwise, and calling it twice is harmless.
func (g *Group) CancelTimeout() {
	g.mu.Lock()
	stop := g.stopTimeout
	g.stopTimeout = nil
	g.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// IsDone reports whether the group has finished, either way.
func (g *Group) IsDone() bool {
	select {
	case <-g.done:
		return true
	default:
		return false
	}
}

// Done is closed when the group finishes.
func (g *Group) Done() <-chan struct{} { return g.done }

// Events carries what the workers report while the group runs. It is closed
// when the group finishes, so ranging over it ends there.
func (g *Group) Events() <-chan GroupEvent { return g.events }

// Wait blocks until the group finishes, or ctx ends. It reports ErrGroup when
// the group was cut short, and the context's error when ctx ended first.
func (g *Group) Wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-g.done:
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.err
}

// Responses is what each worker returned, by worker name. A group cut short
// reports whatever arrived before it was.
func (g *Group) Responses() map[string]map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[string]map[string]any, len(g.responses))
	maps.Copy(out, g.responses)
	return out
}

// Response is what one worker returned, and whether it returned at all.
func (g *Group) Response(workerName string) (map[string]any, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	r, ok := g.responses[workerName]
	return r, ok
}

// SetResponse records what one worker returned.
func (g *Group) SetResponse(workerName string, response map[string]any) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.responses[workerName] = response
}

// Report hands an event to whoever is watching the group. It never blocks: a
// caller is free not to watch, and the worker doing the job must not be held up
// by one that is not reading.
func (g *Group) Report(e GroupEvent) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	select {
	case g.events <- e:
	default:
	}
}

// Complete finishes the group successfully.
func (g *Group) Complete() { g.finish(nil) }

// Fail finishes the group as cut short, for the given reason.
func (g *Group) Fail(reason string) {
	err := ErrGroup
	if reason != "" {
		err = fmt.Errorf("%w: %s", ErrGroup, reason)
	}
	g.finish(err)
}

// finish records how the group ended, once. Finishing closes the event channel,
// which is what ends a caller ranging over it.
func (g *Group) finish(err error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.closed = true
	g.err = err
	close(g.events)
	close(g.done)
}
