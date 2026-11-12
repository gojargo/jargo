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
// Every event a worker reports is queued until the caller takes it, however far
// behind the caller falls, so none is lost. The caller takes them by ranging
// over Events, a channel closed once the group has finished and every event
// has been taken, or one at a time with NextEvent. Upstream marks the end by
// queueing a sentinel; a closed channel is the same thing said in Go, and it
// cannot be missed.
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
	// queued holds the events reported and not yet taken, and arrived is
	// signaled whenever one is added or the group finishes.
	queued  []GroupEvent
	arrived chan struct{}
	// events is the channel Events hands out, fed from the queue by a goroutine
	// started the first time it is asked for.
	events     chan GroupEvent
	eventsOnce sync.Once
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
		arrived:       make(chan struct{}, 1),
	}
}

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

// Events carries what the workers report while the group runs, every event in
// the order it was reported. It is closed once the group has finished and every
// event has been delivered, so ranging over it ends there. Range over it to the
// end: the goroutine feeding it waits on a caller that stops reading early.
func (g *Group) Events() <-chan GroupEvent {
	g.eventsOnce.Do(func() {
		g.events = make(chan GroupEvent)
		go func() {
			defer close(g.events)
			for {
				e, ok := g.NextEvent(context.Background())
				if !ok {
					return
				}
				g.events <- e
			}
		}()
	})
	return g.events
}

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

// Report hands an event to whoever is watching the group. It never blocks: the
// event is queued until the caller takes it, so the worker doing the job is not
// held up by a caller that is slow to read, and none is lost to one.
func (g *Group) Report(e GroupEvent) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.queued = append(g.queued, e)
	g.signalArrived()
}

// NextEvent returns the next event a worker reported, waiting for one. It
// reports false once the group has finished and every event has been taken, or
// when ctx ends first.
func (g *Group) NextEvent(ctx context.Context) (GroupEvent, bool) {
	for {
		g.mu.Lock()
		if len(g.queued) > 0 {
			e := g.queued[0]
			g.queued[0] = GroupEvent{}
			g.queued = g.queued[1:]
			g.mu.Unlock()
			return e, true
		}
		finished := g.closed
		g.mu.Unlock()
		if finished {
			return GroupEvent{}, false
		}
		select {
		case <-g.arrived:
		case <-ctx.Done():
			return GroupEvent{}, false
		}
	}
}

// signalArrived wakes a caller waiting in NextEvent. The caller holds g.mu.
func (g *Group) signalArrived() {
	select {
	case g.arrived <- struct{}{}:
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

// finish records how the group ended, once. The events still queued are
// delivered first; the event channel closes once they have been.
func (g *Group) finish(err error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.closed = true
	g.err = err
	close(g.done)
	g.signalArrived()
}
