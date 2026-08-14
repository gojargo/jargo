package workers

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"sync"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/registry"
	"github.com/google/uuid"
)

// A job is long-running work one worker asks another for. The requester opens a
// group, one entry per worker asked, and the group finishes when they have all
// answered. The worker doing the work runs it on a goroutine of its own, so the
// messages behind the request are not held up by it, and may report progress or
// stream results back while it runs.

// ErrNoActiveJob is reported when a worker answers or reports on a job it is
// not working on, usually one it has already finished or that was called off.
var ErrNoActiveJob = errors.New("no active job of that id")

// ErrNoRegistry is reported when a worker asks for a job before it has been
// attached to a registry, so it cannot tell whether the workers exist.
var ErrNoRegistry = errors.New("worker is not attached to a registry")

// JobHandler does the work one kind of job asks for. Report progress with
// SendJobUpdate and finish with SendJobResponse; a handler that returns without
// answering leaves the requester waiting until it gives up.
type JobHandler func(ctx context.Context, m *bus.JobRequestMessage)

// ReadyHandler is called when the worker it was declared for becomes ready.
type ReadyHandler func(ctx context.Context, data registry.WorkerReadyData)

// JobOptions are the choices a job handler is declared with.
type JobOptions struct {
	// Sequential runs the requests for this job one at a time, in the order
	// they arrived, rather than concurrently. Waiting counts against the
	// requester's timeout, so a slow predecessor can time a queued request out
	// before it starts.
	Sequential bool
}

// JobResponseOptions are the choices answering a job offers.
type JobResponseOptions struct {
	// Status is how the job ended; empty reports it completed.
	Status jobcontext.JobStatus
	// Urgent delivers the answer ahead of the data messages already queued.
	Urgent bool
}

// jobHandler is one declared handler.
type jobHandler struct {
	fn         JobHandler
	sequential bool
}

// runningJob is a handler currently running: how to call it off, and how to
// tell when it has stopped.
type runningJob struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// stopWait bounds how long calling off a handler waits for it to return. A
// handler that ignores its context must not hold up the worker canceling it.
const stopWait = time.Second

// stop calls the handler off and waits briefly for it to return.
func (r *runningJob) stop() {
	r.cancel()
	select {
	case <-r.done:
	case <-time.After(stopWait):
	}
}

// HandleJob declares fn as this worker's handler for job requests named name.
// A request naming no job, or one no handler was declared for, goes to
// OnJobRequest instead.
//
// Declare handlers when building the worker. Declaring two for one name panics:
// the second would never run, and a worker whose handlers are not the ones its
// author declared cannot do the work it was built for. It is a mistake in the
// program rather than something that goes wrong at run time, so it is refused
// where it is made.
func (w *Base) HandleJob(name string, opts JobOptions, fn JobHandler) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, ok := w.handlers[name]; ok {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic(fmt.Sprintf("workers: worker %q declares two job handlers for %q", w.name, name))
	}
	w.handlers[name] = jobHandler{fn: fn, sequential: opts.Sequential}
}

// HandleWorkerReady declares fn as this worker's handler for the named worker
// becoming ready. The worker is watched when this one starts, and fn runs
// before the general OnWorkerReady hook.
//
// Declaring two handlers for one worker panics, for the reason HandleJob does.
func (w *Base) HandleWorkerReady(name string, fn ReadyHandler) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, ok := w.readyHandlers[name]; ok {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic(fmt.Sprintf("workers: worker %q declares two worker-ready handlers for %q",
			w.name, name))
	}
	w.readyHandlers[name] = fn
}

// ActiveJobs are the job requests this worker is working on, by job id.
func (w *Base) ActiveJobs() map[string]*bus.JobRequestMessage {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make(map[string]*bus.JobRequestMessage, len(w.activeJobs))
	maps.Copy(out, w.activeJobs)
	return out
}

// JobGroups are the job groups this worker launched and is still waiting on, by
// job id.
func (w *Base) JobGroups() map[string]*jobcontext.Group {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make(map[string]*jobcontext.Group, len(w.groups))
	maps.Copy(out, w.groups)
	return out
}

// HasGroup reports whether a group of that id is still running.
func (w *Base) HasGroup(jobID string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	_, ok := w.groups[jobID]
	return ok
}

// RequestJob asks one worker for a job and does not wait for it.
//
// It waits for the worker to be ready, sends the request and returns the job's
// id. Watch for the answer with OnJobResponse or OnJobCompleted, or use Job to
// wait for it.
func (w *Base) RequestJob(ctx context.Context, workerName string, req jobcontext.Request) (string, error) {
	group, err := w.CreateGroupAndRequestJob(ctx, []string{workerName}, cancellingOnError(req))
	if err != nil {
		return "", err
	}
	return group.JobID, nil
}

// RequestJobGroup asks several workers for a job together and does not wait for
// it. It returns the id they all share.
func (w *Base) RequestJobGroup(ctx context.Context, workerNames []string, req jobcontext.Request) (string, error) {
	group, err := w.CreateGroupAndRequestJob(ctx, workerNames, req)
	if err != nil {
		return "", err
	}
	return group.JobID, nil
}

// Job asks one worker for a job, runs block while it works, and waits for it on
// the way out. See jobcontext.RunJob.
func (w *Base) Job(
	ctx context.Context, workerName string, req jobcontext.Request, block func(j *jobcontext.Job) error,
) (*jobcontext.Job, error) {
	return jobcontext.RunJob(ctx, w, workerName, req, block)
}

// JobGroup asks several workers for a job together, runs block while they work,
// and waits for them all on the way out. See jobcontext.RunGroup.
func (w *Base) JobGroup(
	ctx context.Context, workerNames []string, req jobcontext.Request, block func(g *jobcontext.Group) error,
) (*jobcontext.Group, error) {
	return jobcontext.RunGroup(ctx, w, workerNames, req, block)
}

// CancelGroup calls a running job group off, telling each of its workers.
func (w *Base) CancelGroup(ctx context.Context, jobID, reason string) {
	w.mu.Lock()
	group, ok := w.groups[jobID]
	delete(w.groups, jobID)
	w.mu.Unlock()
	if !ok {
		return
	}

	group.CancelTimeout()
	for _, workerName := range group.WorkerNames {
		m := &bus.JobCancelMessage{JobID: jobID, Reason: reason}
		m.From = w.name
		m.To = workerName
		w.SendBusMessage(ctx, m)
	}
	group.Fail(reason)
}

// RequestJobUpdate asks a worker how far along it is with a job.
func (w *Base) RequestJobUpdate(ctx context.Context, jobID, workerName string) {
	m := &bus.JobUpdateRequestMessage{JobID: jobID}
	m.From = w.name
	m.To = workerName
	w.SendBusMessage(ctx, m)
}

// CreateGroupAndRequestJob waits for the named workers to be ready, opens a
// group for them and sends each the request. It does not wait for the group;
// use Group.Wait, or JobGroup, for that.
//
// It reports ErrGroup when the workers are not all ready within the request's
// timeout.
func (w *Base) CreateGroupAndRequestJob(
	ctx context.Context, workerNames []string, req jobcontext.Request,
) (*jobcontext.Group, error) {
	if err := w.waitWorkersReady(ctx, workerNames, req.Timeout); err != nil {
		return nil, err
	}

	group := w.createGroup(ctx, workerNames, req)

	for _, workerName := range workerNames {
		m := &bus.JobRequestMessage{
			JobID:   group.JobID,
			JobName: req.Name,
			Payload: req.Payload,
		}
		m.From = w.name
		m.To = workerName
		w.SendBusMessage(ctx, m)
	}

	return group, nil
}

// SendJobResponse answers the worker that asked for a job, and ends this
// worker's part in it.
func (w *Base) SendJobResponse(
	ctx context.Context, jobID string, response map[string]any, opts JobResponseOptions,
) error {
	request, ok := w.activeJob(jobID)
	if !ok {
		return fmt.Errorf("worker %q: %w: %q", w.name, ErrNoActiveJob, jobID)
	}

	status := opts.Status
	if status == "" {
		status = jobcontext.JobCompleted
	}
	result := bus.JobResult{JobID: jobID, Status: status, Response: response}

	var m bus.Message
	if opts.Urgent {
		urgent := &bus.JobResponseUrgentMessage{JobResult: result}
		urgent.From = w.name
		urgent.To = request.Source()
		m = urgent
	} else {
		ordinary := &bus.JobResponseMessage{JobResult: result}
		ordinary.From = w.name
		ordinary.To = request.Source()
		m = ordinary
	}
	w.SendBusMessage(ctx, m)

	w.forgetActiveJob(jobID)
	return nil
}

// SendJobUpdate reports progress on a job to the worker that asked for it.
// Urgent delivers it ahead of the data messages already queued.
func (w *Base) SendJobUpdate(ctx context.Context, jobID string, update map[string]any, urgent bool) error {
	request, ok := w.activeJob(jobID)
	if !ok {
		return fmt.Errorf("worker %q: %w: %q", w.name, ErrNoActiveJob, jobID)
	}

	progress := bus.JobProgress{JobID: jobID, Update: update}

	var m bus.Message
	if urgent {
		msg := &bus.JobUpdateUrgentMessage{JobProgress: progress}
		msg.From = w.name
		msg.To = request.Source()
		m = msg
	} else {
		msg := &bus.JobUpdateMessage{JobProgress: progress}
		msg.From = w.name
		msg.To = request.Source()
		m = msg
	}
	w.SendBusMessage(ctx, m)
	return nil
}

// SendJobStreamStart opens a stream of results back to the worker that asked
// for the job.
func (w *Base) SendJobStreamStart(ctx context.Context, jobID string, data map[string]any) error {
	request, ok := w.activeJob(jobID)
	if !ok {
		return fmt.Errorf("worker %q: %w: %q", w.name, ErrNoActiveJob, jobID)
	}
	m := &bus.JobStreamStartMessage{JobID: jobID, Data: data}
	m.From = w.name
	m.To = request.Source()
	w.SendBusMessage(ctx, m)
	return nil
}

// SendJobStreamData sends one item of a job's result stream.
func (w *Base) SendJobStreamData(ctx context.Context, jobID string, data map[string]any) error {
	request, ok := w.activeJob(jobID)
	if !ok {
		return fmt.Errorf("worker %q: %w: %q", w.name, ErrNoActiveJob, jobID)
	}
	m := &bus.JobStreamDataMessage{JobID: jobID, Data: data}
	m.From = w.name
	m.To = request.Source()
	w.SendBusMessage(ctx, m)
	return nil
}

// SendJobStreamEnd closes a job's result stream, and ends this worker's part in
// the job: the stream's end is its answer, so a cancel arriving afterwards
// finds nothing to call off.
func (w *Base) SendJobStreamEnd(ctx context.Context, jobID string, data map[string]any) error {
	request, ok := w.activeJob(jobID)
	if !ok {
		return fmt.Errorf("worker %q: %w: %q", w.name, ErrNoActiveJob, jobID)
	}
	m := &bus.JobStreamEndMessage{JobID: jobID, Data: data}
	m.From = w.name
	m.To = request.Source()
	w.SendBusMessage(ctx, m)

	w.forgetActiveJob(jobID)
	return nil
}

// OnJobRequest is called when this worker is asked for a job no declared
// handler matched. Override it to do the work. The default does nothing, which
// leaves the requester waiting.
func (w *Base) OnJobRequest(context.Context, *bus.JobRequestMessage) {}

// OnJobResponse is called when a worker reports how its job ended. Override it
// to take each answer as it arrives, rather than waiting for the group.
func (w *Base) OnJobResponse(context.Context, bus.JobResponse) {}

// OnJobUpdate is called when a worker reports progress on a job.
func (w *Base) OnJobUpdate(context.Context, bus.JobUpdate) {}

// OnJobUpdateRequested is called when the requester asks how far along this
// worker is. Override it to answer with SendJobUpdate.
func (w *Base) OnJobUpdateRequested(context.Context, *bus.JobUpdateRequestMessage) {}

// OnJobCompleted is called when every worker in a group has answered.
func (w *Base) OnJobCompleted(context.Context, jobcontext.GroupResponse) {}

// OnJobError is called when a worker reported it failed and the group was
// called off, so OnJobCompleted will not fire. What the other workers answered
// before it is in the group's responses.
func (w *Base) OnJobError(context.Context, bus.JobResponse) {}

// OnJobStreamStart is called when a worker opens a stream of results.
func (w *Base) OnJobStreamStart(context.Context, *bus.JobStreamStartMessage) {}

// OnJobStreamData is called for each item of a worker's result stream.
func (w *Base) OnJobStreamData(context.Context, *bus.JobStreamDataMessage) {}

// OnJobStreamEnd is called when a worker closes its result stream.
func (w *Base) OnJobStreamEnd(context.Context, *bus.JobStreamEndMessage) {}

// OnJobCancelled is called when a job this worker is doing is called off.
// Override it to release what the work was holding; the canceled answer is
// sent for you afterwards.
func (w *Base) OnJobCancelled(context.Context, *bus.JobCancelMessage) {}

// handleJobRequest takes on a job and starts its handler.
//
// The handler runs on a goroutine of its own so the messages queued behind the
// request are not held up by the work. A handler declared sequential waits for
// the previous request of the same name to finish first.
func (w *Base) handleJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	// Detached from the message's own context, because the work outlives
	// delivery: it ends when it is done, when the job is called off, or when
	// the worker stops.
	jobCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	running := &runningJob{cancel: cancel, done: make(chan struct{})}

	w.mu.Lock()
	w.activeJobs[m.JobID] = m
	handler, declared := w.handlers[m.JobName]
	if m.JobName == "" {
		declared = false
	}

	// The turn a sequential request waits for is taken here, where requests are
	// still in the order they arrived, rather than in the goroutine, where the
	// order they start in is not decided.
	var wait, release chan struct{}
	if declared && handler.sequential {
		wait = w.sequential[m.JobName]
		release = make(chan struct{})
		w.sequential[m.JobName] = release
	}

	w.jobHandlers[m.JobID] = running
	w.mu.Unlock()

	fn := handler.fn
	if !declared {
		fn = func(ctx context.Context, m *bus.JobRequestMessage) { w.self.OnJobRequest(ctx, m) }
	}

	go w.runJobHandler(jobCtx, running, m, fn, wait, release)

	w.Call(ctx, EventJobRequest, w.self, m)
}

// runJobHandler waits for this request's turn, if it has one, and runs the
// handler.
func (w *Base) runJobHandler(
	ctx context.Context, running *runningJob, m *bus.JobRequestMessage, fn JobHandler, wait, release chan struct{},
) {
	defer close(running.done)
	defer w.forgetJobHandler(m.JobID)
	// Released whatever happened, so a request called off while queued does not
	// strand the ones behind it.
	if release != nil {
		defer close(release)
	}

	if wait != nil {
		select {
		case <-wait:
		case <-ctx.Done():
			return
		}
	}

	fn(ctx, m)
}

// handleJobResponse takes a worker's answer, and calls the group off when the
// answer says the worker failed.
func (w *Base) handleJobResponse(ctx context.Context, m bus.JobResponse) {
	w.self.OnJobResponse(ctx, m)
	w.Call(ctx, EventJobResponse, w.self, m)

	result := m.Result()
	if result.Status == jobcontext.JobError || result.Status == jobcontext.JobFailed {
		w.mu.Lock()
		group, running := w.groups[result.JobID]
		w.mu.Unlock()
		if running && group.CancelOnError {
			group.SetResponse(m.Source(), orEmpty(result.Response))
			w.self.OnJobError(ctx, m)
			w.Call(ctx, EventJobError, w.self, m)
			w.CancelGroup(ctx, result.JobID, fmt.Sprintf("worker %q errored", m.Source()))
			return
		}
	}

	w.trackGroupResponse(ctx, result.JobID, m.Source(), result.Response)
}

// handleJobUpdate takes a worker's progress report.
func (w *Base) handleJobUpdate(ctx context.Context, m bus.JobUpdate) {
	w.self.OnJobUpdate(ctx, m)
	w.Call(ctx, EventJobUpdate, w.self, m)
	progress := m.Progress()
	w.reportGroupEvent(progress.JobID, jobcontext.GroupEvent{
		Type:       jobcontext.EventUpdate,
		WorkerName: m.Source(),
		Data:       progress.Update,
	})
}

// handleJobUpdateRequest answers a request for progress on a job this worker is
// doing, and ignores one for a job it is not.
func (w *Base) handleJobUpdateRequest(ctx context.Context, m *bus.JobUpdateRequestMessage) {
	if _, ok := w.activeJob(m.JobID); !ok {
		return
	}
	w.self.OnJobUpdateRequested(ctx, m)
	w.Call(ctx, EventJobUpdateRequested, w.self, m)
}

// handleJobCancel calls off a job this worker is doing: it stops the handler,
// lets the worker release what the work was holding, and answers the requester
// as canceled, which reaches it the same way a completed or failed job would.
func (w *Base) handleJobCancel(ctx context.Context, m *bus.JobCancelMessage) {
	if _, ok := w.activeJob(m.JobID); !ok {
		return
	}

	w.mu.Lock()
	running := w.jobHandlers[m.JobID]
	w.mu.Unlock()
	if running != nil {
		running.stop()
	}

	w.self.OnJobCancelled(ctx, m)
	w.Call(ctx, EventJobCancelled, w.self, m)

	if err := w.SendJobResponse(ctx, m.JobID, nil, JobResponseOptions{
		Status: jobcontext.JobCancelled,
	}); err != nil {
		slog.Debug("could not report a job canceled",
			"worker", w.name, "job", m.JobID, "error", err)
	}
}

// handleJobStreamStart takes the opening of a worker's result stream.
func (w *Base) handleJobStreamStart(ctx context.Context, m *bus.JobStreamStartMessage) {
	w.self.OnJobStreamStart(ctx, m)
	w.Call(ctx, EventJobStreamStart, w.self, m)
	w.reportGroupEvent(m.JobID, jobcontext.GroupEvent{
		Type:       jobcontext.EventStreamStart,
		WorkerName: m.Source(),
		Data:       m.Data,
	})
}

// handleJobStreamData takes one item of a worker's result stream.
func (w *Base) handleJobStreamData(ctx context.Context, m *bus.JobStreamDataMessage) {
	w.self.OnJobStreamData(ctx, m)
	w.Call(ctx, EventJobStreamData, w.self, m)
	w.reportGroupEvent(m.JobID, jobcontext.GroupEvent{
		Type:       jobcontext.EventStreamData,
		WorkerName: m.Source(),
		Data:       m.Data,
	})
}

// handleJobStreamEnd takes the close of a worker's result stream, which is also
// that worker's answer.
func (w *Base) handleJobStreamEnd(ctx context.Context, m *bus.JobStreamEndMessage) {
	w.self.OnJobStreamEnd(ctx, m)
	w.Call(ctx, EventJobStreamEnd, w.self, m)
	w.reportGroupEvent(m.JobID, jobcontext.GroupEvent{
		Type:       jobcontext.EventStreamEnd,
		WorkerName: m.Source(),
		Data:       m.Data,
	})
	w.trackGroupResponse(ctx, m.JobID, m.Source(), m.Data)
}

// createGroup opens a group for the workers a job is going to, and starts its
// timeout when it has one.
func (w *Base) createGroup(ctx context.Context, workerNames []string, req jobcontext.Request) *jobcontext.Group {
	group := jobcontext.NewGroup(uuid.NewString(), workerNames, req.CancelsOnError())

	w.mu.Lock()
	w.groups[group.JobID] = group
	w.mu.Unlock()

	if req.Timeout > 0 {
		w.startGroupTimeout(ctx, group, req.Timeout)
	}
	return group
}

// startGroupTimeout calls the group off if it is still running once the timeout
// is up. It is detached from ctx for the same reason a job handler is: it
// outlives the call that asked for the job.
func (w *Base) startGroupTimeout(ctx context.Context, group *jobcontext.Group, timeout time.Duration) {
	timeoutCtx := context.WithoutCancel(ctx)
	stop := make(chan struct{})
	group.SetTimeout(func() { close(stop) })

	go func() {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		select {
		case <-stop:
			return
		case <-timer.C:
		}
		w.CancelGroup(timeoutCtx, group.JobID, "timeout")
	}()
}

// waitWorkersReady waits for every named worker to be registered, for at most
// timeout when one is given.
func (w *Base) waitWorkersReady(ctx context.Context, workerNames []string, timeout time.Duration) error {
	reg := w.WorkerRegistry()
	if reg == nil {
		return fmt.Errorf("worker %q: %w", w.name, ErrNoRegistry)
	}

	ready := make([]chan struct{}, 0, len(workerNames))
	for _, workerName := range workerNames {
		ch := make(chan struct{})
		var once sync.Once

		w.mu.Lock()
		w.readySeq++
		// Keyed uniquely, so waiting twice for the same worker is two waits.
		key := fmt.Sprintf("%s:ready-wait:%d", w.name, w.readySeq)
		w.mu.Unlock()

		reg.Watch(ctx, workerName, key, func(context.Context, registry.WorkerReadyData) {
			once.Do(func() { close(ch) })
		})
		ready = append(ready, ch)
	}

	var deadline <-chan time.Time
	if timeout > 0 {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		deadline = timer.C
	}

	for _, ch := range ready {
		select {
		case <-ch:
		case <-deadline:
			return fmt.Errorf("%w: workers not ready within timeout", jobcontext.ErrGroup)
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// reportGroupEvent hands an event to whoever is watching the group it belongs
// to, and drops it when no group of that id is running.
func (w *Base) reportGroupEvent(jobID string, e jobcontext.GroupEvent) {
	w.mu.Lock()
	group, ok := w.groups[jobID]
	w.mu.Unlock()
	if ok {
		group.Report(e)
	}
}

// trackGroupResponse records what a worker answered and finishes the group once
// they all have.
func (w *Base) trackGroupResponse(ctx context.Context, jobID, source string, response map[string]any) {
	w.mu.Lock()
	group, ok := w.groups[jobID]
	w.mu.Unlock()
	if !ok {
		return
	}
	group.SetResponse(source, orEmpty(response))

	// Taken under the lock, so two answers arriving at once cannot both finish
	// the group.
	w.mu.Lock()
	_, stillRunning := w.groups[jobID]
	complete := stillRunning && respondedToAll(group)
	if complete {
		delete(w.groups, jobID)
	}
	w.mu.Unlock()
	if !complete {
		return
	}

	group.CancelTimeout()
	result := jobcontext.GroupResponse{JobID: jobID, Responses: group.Responses()}
	w.self.OnJobCompleted(ctx, result)
	w.Call(ctx, EventJobCompleted, w.self, result)
	group.Complete()
}

// respondedToAll reports whether every worker in the group has answered.
func respondedToAll(group *jobcontext.Group) bool {
	responses := group.Responses()
	for _, workerName := range group.WorkerNames {
		if _, ok := responses[workerName]; !ok {
			return false
		}
	}
	return true
}

// cancellingOnError is req with one worker failing calling the job off, which
// is what asking a single worker always does.
func cancellingOnError(req jobcontext.Request) jobcontext.Request {
	cancelOnError := true
	req.CancelOnError = &cancelOnError
	return req
}

// orEmpty is a response as it is recorded: a worker that answered with nothing
// still answered.
func orEmpty(response map[string]any) map[string]any {
	if response == nil {
		return map[string]any{}
	}
	return response
}

func (w *Base) activeJob(jobID string) (*bus.JobRequestMessage, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	m, ok := w.activeJobs[jobID]
	return m, ok
}

func (w *Base) forgetActiveJob(jobID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.activeJobs, jobID)
}

func (w *Base) forgetJobHandler(jobID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.jobHandlers, jobID)
}

// groupIDs are the ids of the groups this worker is waiting on.
func (w *Base) groupIDs() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]string, 0, len(w.groups))
	for id := range w.groups {
		out = append(out, id)
	}
	return out
}

// activeJobIDs are the ids of the jobs this worker is working on.
func (w *Base) activeJobIDs() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]string, 0, len(w.activeJobs))
	for id := range w.activeJobs {
		out = append(out, id)
	}
	return out
}

// takeRunningJobs hands over the handlers still running and forgets them.
func (w *Base) takeRunningJobs() []*runningJob {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]*runningJob, 0, len(w.jobHandlers))
	for id, running := range w.jobHandlers {
		out = append(out, running)
		delete(w.jobHandlers, id)
	}
	return out
}
