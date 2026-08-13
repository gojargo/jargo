package jobcontext

import (
	"context"
	"fmt"
	"time"
)

// Running a job means asking one or more workers for it, watching what they
// report while it runs, and collecting what they returned. RunJob and RunGroup
// do all three around a block of the caller's own code.
//
// Upstream expresses this as an async context manager entered with a job
// request and exited on the responses. In Go that is a function taking the
// block: what runs inside the block is what ran inside the `with`, and the
// group is waited for, or called off, on the way out.

// Request describes a job to ask a worker for.
type Request struct {
	// Name routes the request to the worker's handler for that kind of work,
	// and may be empty for its default one.
	Name string
	// Payload describes the work, and may be nil.
	Payload map[string]any
	// Timeout bounds the wait for the workers to be ready and the job itself.
	// Zero runs without a timeout.
	Timeout time.Duration
	// CancelOnError calls the whole group off when one worker reports it
	// failed; nil defaults to true.
	CancelOnError *bool
}

// CancelsOnError reports whether one worker failing calls off the rest.
func (r Request) CancelsOnError() bool {
	return r.CancelOnError == nil || *r.CancelOnError
}

// Requester is the part of a worker that runs jobs for a caller. It is named
// here rather than imported because the workers are built on this package; a
// worker satisfies it.
type Requester interface {
	// CreateGroupAndRequestJob waits for the named workers to be ready, opens a
	// group for them and sends each the request.
	CreateGroupAndRequestJob(ctx context.Context, workerNames []string, req Request) (*Group, error)
	// CancelGroup calls a running group off, for the given reason.
	CancelGroup(ctx context.Context, jobID, reason string)
	// HasGroup reports whether a group of that id is still running.
	HasGroup(jobID string) bool
}

// blockFailedReason is what a group is called off for when the block around it
// did not finish.
const blockFailedReason = "context exited with error"

// RunGroup asks a set of workers for a job together, runs block while they work
// on it, and waits for them all on the way out.
//
// The group is returned whether or not the job succeeded, so a caller can read
// what did arrive from a group that was cut short. block may range over
// Group.Events to see what the workers report while they work; the channel
// closes when the group finishes. A block that returns an error calls the group
// off and that error is returned; otherwise the error is the group's own,
// ErrGroup when it was cut short.
func RunGroup(
	ctx context.Context, r Requester, workerNames []string, req Request, block func(g *Group) error,
) (*Group, error) {
	group, err := r.CreateGroupAndRequestJob(ctx, workerNames, req)
	if err != nil {
		return nil, err
	}

	if block != nil {
		if err := block(group); err != nil {
			cancelGroup(ctx, r, group)
			return group, err
		}
	}

	return group, group.Wait(ctx)
}

// RunJob asks one worker for a job, runs block while it works on it, and waits
// for it on the way out.
//
// It is RunGroup for a single worker, and reports ErrJob rather than ErrGroup
// when the job was cut short. One worker failing always calls the job off, so
// Request.CancelOnError is not consulted.
func RunJob(ctx context.Context, r Requester, workerName string, req Request, block func(j *Job) error) (*Job, error) {
	cancelOnError := true
	req.CancelOnError = &cancelOnError

	group, err := r.CreateGroupAndRequestJob(ctx, []string{workerName}, req)
	if err != nil {
		return nil, err
	}
	job := newJob(workerName, group)

	if block != nil {
		if err := block(job); err != nil {
			cancelGroup(ctx, r, group)
			return job, err
		}
	}

	if err := group.Wait(ctx); err != nil {
		return job, fmt.Errorf("%w: %s", ErrJob, err.Error())
	}
	return job, nil
}

// cancelGroup calls a group off after the block around it failed. The
// cancellation is detached from ctx, so it still reaches the workers when what
// ended the block was ctx itself: a job left running is worse than one called
// off late.
func cancelGroup(ctx context.Context, r Requester, group *Group) {
	if !r.HasGroup(group.JobID) {
		return
	}
	r.CancelGroup(context.WithoutCancel(ctx), group.JobID, blockFailedReason)
}

// Job is one worker's job: what it reports while it works, and what it
// returned.
type Job struct {
	workerName string
	group      *Group
	events     chan Event
}

// newJob wraps a single-worker group, translating what that worker reports into
// events with no worker name on them, since there is only the one.
func newJob(workerName string, group *Group) *Job {
	j := &Job{
		workerName: workerName,
		group:      group,
		events:     make(chan Event, eventBuffer),
	}
	go j.forwardEvents()
	return j
}

// forwardEvents copies the group's events onto the job's channel until the
// group finishes. Like the group's own reporting it never blocks, so a caller
// that is not watching cannot hold up the worker doing the job.
func (j *Job) forwardEvents() {
	defer close(j.events)
	for e := range j.group.Events() {
		select {
		case j.events <- Event{Type: e.Type, Data: e.Data}:
		default:
		}
	}
}

// JobID identifies the job.
func (j *Job) JobID() string { return j.group.JobID }

// Response is what the worker returned, and is empty until it has.
func (j *Job) Response() map[string]any {
	r, ok := j.group.Response(j.workerName)
	if !ok {
		return map[string]any{}
	}
	return r
}

// Events carries what the worker reports while the job runs. It is closed when
// the job finishes, so ranging over it ends there.
func (j *Job) Events() <-chan Event { return j.events }
