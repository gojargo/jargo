package workers_test

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/workers"
)

// Running a job group: what a caller sees while the workers work, what it
// collects when they finish, and how a group ends when one of them fails, when
// it runs out of time, or when the block around it does not finish.

func TestJobGroupCollectsResponses(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("w1", map[string]any{"a": 1}, jobcontext.JobCompleted))
	env.setup(newJobWorker("w2", map[string]any{"b": 2}, jobcontext.JobCompleted))

	group, err := parent.JobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{
		Payload: map[string]any{"work": true},
	}, nil)
	if err != nil {
		t.Fatalf("job group: %v", err)
	}

	want := map[string]map[string]any{"w1": {"a": 1}, "w2": {"b": 2}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupSendsRequest(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("w1", map[string]any{"ok": true}, jobcontext.JobCompleted))

	if _, err := parent.JobGroup(env.ctx, []string{"w1"}, jobcontext.Request{
		Payload: map[string]any{"data": 1},
	}, nil); err != nil {
		t.Fatalf("job group: %v", err)
	}

	requests := await[*bus.JobRequestMessage](t, env.recorder, 1)
	if len(requests) != 1 {
		t.Fatalf("got %d job requests, want 1", len(requests))
	}
	if want := (map[string]any{"data": 1}); !reflect.DeepEqual(requests[0].Payload, want) {
		t.Errorf("payload = %v, want %v", requests[0].Payload, want)
	}
}

func TestJobGroupReportsCancel(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	// The stub never answers, so the group can be called off by hand.
	env.setup(newStubWorker("worker"))

	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, func(g *jobcontext.Group) error {
		go parent.CancelGroup(env.ctx, g.JobID, "manual cancel")
		return nil
	})
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}
	if !strings.Contains(err.Error(), "manual cancel") {
		t.Errorf("error = %q, want it to carry the reason", err)
	}
}

func TestJobGroupReportsTimeout(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	// The stub never answers, so the timeout fires.
	env.setup(newStubWorker("worker"))

	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{
		Timeout: 50 * time.Millisecond,
	}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}
	if !strings.Contains(err.Error(), "timeout") {
		t.Errorf("error = %q, want it to say it timed out", err)
	}
}

func TestJobGroupReportsReadyTimeout(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)

	// "ghost" never registers, so the wait for it to be ready runs out.
	_, err := parent.JobGroup(env.ctx, []string{"ghost"}, jobcontext.Request{
		Timeout: 50 * time.Millisecond,
	}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}
	if !strings.Contains(err.Error(), "not ready") {
		t.Errorf("error = %q, want it to say the workers were not ready", err)
	}
}

func TestJobGroupCancelsWhenBlockFails(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStubWorker("worker"))

	wantErr := errBlockFailed
	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, func(*jobcontext.Group) error {
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want the block's own", err)
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].Reason != "context exited with error" {
		t.Errorf("reason = %q", cancels[0].Reason)
	}
}

func TestJobGroupReportsWorkerError(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"error": "failed"}, jobcontext.JobError))

	group, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}

	// What the failing worker reported is kept among the partial responses.
	want := map[string]map[string]any{"worker": {"error": "failed"}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupFiresJobError(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"error": "boom"}, jobcontext.JobError))

	var mu sync.Mutex
	var errs []bus.JobResponse
	parent.Add(workers.EventJobError, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(bus.JobResponse)
		if !ok {
			return
		}
		mu.Lock()
		errs = append(errs, typed)
		mu.Unlock()
	})

	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}

	eventually(t, "the job error is reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(errs) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	result := errs[0].Result()
	if errs[0].Source() != "worker" {
		t.Errorf("source = %q, want worker", errs[0].Source())
	}
	if want := (map[string]any{"error": "boom"}); !reflect.DeepEqual(result.Response, want) {
		t.Errorf("response = %v, want %v", result.Response, want)
	}
	if result.Status != jobcontext.JobError {
		t.Errorf("status = %q, want error", result.Status)
	}
}

func TestJobGroupKeepsPartialResponsesOnError(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	// One worker succeeds and one fails. Which answers first is up to the bus,
	// but both are asked.
	env.setup(newJobWorker("w1", map[string]any{"ok": true}, jobcontext.JobCompleted))
	env.setup(newJobWorker("w2", map[string]any{"error": "fail"}, jobcontext.JobError))

	group, err := parent.JobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}

	responses := group.Responses()
	got, ok := responses["w2"]
	if !ok {
		t.Fatalf("responses = %v, want the failing worker's among them", responses)
	}
	if want := (map[string]any{"error": "fail"}); !reflect.DeepEqual(got, want) {
		t.Errorf("w2 response = %v, want %v", got, want)
	}
}

func TestJobGroupExposesJobID(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{}, jobcontext.JobCompleted))

	var captured string
	group, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, func(g *jobcontext.Group) error {
		captured = g.JobID
		return nil
	})
	if err != nil {
		t.Fatalf("job group: %v", err)
	}
	if captured == "" {
		t.Fatal("job id was empty inside the block")
	}
	if captured != group.JobID {
		t.Errorf("job id = %q inside the block, %q after it", captured, group.JobID)
	}
}

func TestJobGroupFiresJobCompleted(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("w1", map[string]any{"ok": true}, jobcontext.JobCompleted))

	var mu sync.Mutex
	var completed []jobcontext.GroupResponse
	parent.Add(workers.EventJobCompleted, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(jobcontext.GroupResponse)
		if !ok {
			return
		}
		mu.Lock()
		completed = append(completed, typed)
		mu.Unlock()
	})

	if _, err := parent.JobGroup(env.ctx, []string{"w1"}, jobcontext.Request{}, nil); err != nil {
		t.Fatalf("job group: %v", err)
	}

	eventually(t, "the group completion is reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(completed) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	want := map[string]map[string]any{"w1": {"ok": true}}
	if !reflect.DeepEqual(completed[0].Responses, want) {
		t.Errorf("responses = %v, want %v", completed[0].Responses, want)
	}
}

func TestJobGroupIteratesUpdates(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUpdatingWorker("worker",
		[]map[string]any{{"progress": 25}, {"progress": 75}},
		map[string]any{"result": "done"}))

	var events []jobcontext.GroupEvent
	group, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{
		Payload: map[string]any{"work": true},
	}, func(g *jobcontext.Group) error {
		for e := range g.Events() {
			events = append(events, e)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("job group: %v", err)
	}

	if len(events) != 2 {
		t.Fatalf("got %d events, want 2: %v", len(events), events)
	}
	if events[0].Type != jobcontext.EventUpdate {
		t.Errorf("first event type = %q, want update", events[0].Type)
	}
	if events[0].WorkerName != "worker" {
		t.Errorf("first event worker = %q, want worker", events[0].WorkerName)
	}
	if want := (map[string]any{"progress": 25}); !reflect.DeepEqual(events[0].Data, want) {
		t.Errorf("first event data = %v, want %v", events[0].Data, want)
	}
	if want := (map[string]any{"progress": 75}); !reflect.DeepEqual(events[1].Data, want) {
		t.Errorf("second event data = %v, want %v", events[1].Data, want)
	}
	want := map[string]map[string]any{"worker": {"result": "done"}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupIteratesStreamEvents(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStreamingWorker("worker",
		[]map[string]any{{"text": "hello "}, {"text": "world"}},
		map[string]any{"final": true}))

	var events []jobcontext.GroupEvent
	if _, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, func(g *jobcontext.Group) error {
		for e := range g.Events() {
			events = append(events, e)
		}
		return nil
	}); err != nil {
		t.Fatalf("job group: %v", err)
	}

	wantTypes := []string{
		jobcontext.EventStreamStart,
		jobcontext.EventStreamData,
		jobcontext.EventStreamData,
		jobcontext.EventStreamEnd,
	}
	var gotTypes []string
	for _, e := range events {
		gotTypes = append(gotTypes, e.Type)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	wantData := []map[string]any{
		{"content_type": "text"},
		{"text": "hello "},
		{"text": "world"},
		{"final": true},
	}
	for i, want := range wantData {
		if !reflect.DeepEqual(events[i].Data, want) {
			t.Errorf("event %d data = %v, want %v", i, events[i].Data, want)
		}
	}
}

func TestJobGroupIteratesMixedEvents(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUpdatingWorker("w1", []map[string]any{{"status": "working"}}, map[string]any{"a": 1}))
	env.setup(newJobWorker("w2", map[string]any{"b": 2}, jobcontext.JobCompleted))

	var events []jobcontext.GroupEvent
	group, err := parent.JobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{}, func(g *jobcontext.Group) error {
		for e := range g.Events() {
			events = append(events, e)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("job group: %v", err)
	}

	if len(events) != 1 {
		t.Fatalf("got %d events, want 1: %v", len(events), events)
	}
	if events[0].Type != jobcontext.EventUpdate {
		t.Errorf("event type = %q, want update", events[0].Type)
	}
	if events[0].WorkerName != "w1" {
		t.Errorf("event worker = %q, want w1", events[0].WorkerName)
	}
	want := map[string]map[string]any{"w1": {"a": 1}, "w2": {"b": 2}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupWithoutIterating(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUpdatingWorker("worker", []map[string]any{{"progress": 50}}, map[string]any{"ok": true}))

	group, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, nil)
	if err != nil {
		t.Fatalf("job group: %v", err)
	}
	want := map[string]map[string]any{"worker": {"ok": true}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupCollectsUrgentResponse(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUrgentJobWorker("worker", map[string]any{"urgent": true}, jobcontext.JobCompleted))

	group, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, nil)
	if err != nil {
		t.Fatalf("job group: %v", err)
	}
	want := map[string]map[string]any{"worker": {"urgent": true}}
	if got := group.Responses(); !reflect.DeepEqual(got, want) {
		t.Errorf("responses = %v, want %v", got, want)
	}
}

func TestJobGroupUrgentErrorFiresJobError(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUrgentJobWorker("worker", map[string]any{"error": "critical"}, jobcontext.JobError))

	var mu sync.Mutex
	var errs []bus.JobResponse
	parent.Add(workers.EventJobError, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(bus.JobResponse)
		if !ok {
			return
		}
		mu.Lock()
		errs = append(errs, typed)
		mu.Unlock()
	})

	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, nil)
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Fatalf("error = %v, want an ErrGroup", err)
	}

	eventually(t, "the urgent job error is reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(errs) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	if want := (map[string]any{"error": "critical"}); !reflect.DeepEqual(errs[0].Result().Response, want) {
		t.Errorf("response = %v, want %v", errs[0].Result().Response, want)
	}
}

func TestUrgentResponseHasSystemPriority(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	msgBus := bus.NewAsyncQueueBus()
	reg := newRegistry()

	parent := newRecordingWorker("parent")
	parent.Attach(ctx, reg, msgBus.Bus)
	reg.Register(ctx, readyData("parent"))

	// Queued before dispatch starts: the data messages first, then the urgent
	// answer behind them.
	for range 3 {
		data := &bus.EndMessage{}
		data.From = "other"
		msgBus.Send(ctx, data)
	}
	urgent := &bus.JobResponseUrgentMessage{
		JobResult: bus.JobResult{JobID: "t1", Status: jobcontext.JobCompleted},
	}
	urgent.From = "worker"
	urgent.To = "parent"
	msgBus.Send(ctx, urgent)

	msgBus.Start(ctx)
	defer msgBus.Stop()

	eventually(t, "the messages are delivered", func() bool {
		return len(parent.messages()) >= 4
	})

	if received := parent.messages(); received[0] == nil {
		t.Fatal("no message was delivered")
	} else if _, ok := received[0].(*bus.JobResponseUrgentMessage); !ok {
		t.Errorf("first message = %T, want the urgent answer ahead of the queued data", received[0])
	}
}

// Running a single-worker job: the same machinery seen through one worker, so
// what it reports carries no worker name and a failure is reported as a job
// error rather than a group one.

func TestJobCollectsResponse(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"result": 42}, jobcontext.JobCompleted))

	job, err := parent.Job(env.ctx, "worker", jobcontext.Request{
		Payload: map[string]any{"x": 1},
	}, nil)
	if err != nil {
		t.Fatalf("job: %v", err)
	}
	if want := (map[string]any{"result": 42}); !reflect.DeepEqual(job.Response(), want) {
		t.Errorf("response = %v, want %v", job.Response(), want)
	}
}

func TestJobSendsRequest(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"ok": true}, jobcontext.JobCompleted))

	if _, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, nil); err != nil {
		t.Fatalf("job: %v", err)
	}

	requests := await[*bus.JobRequestMessage](t, env.recorder, 1)
	if len(requests) != 1 {
		t.Fatalf("got %d job requests, want 1", len(requests))
	}
	if requests[0].Target() != "worker" {
		t.Errorf("target = %q, want worker", requests[0].Target())
	}
	if requests[0].Source() != "parent" {
		t.Errorf("source = %q, want parent", requests[0].Source())
	}
}

func TestJobIteratesEvents(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newUpdatingWorker("worker", []map[string]any{{"progress": 50}}, map[string]any{"done": true}))

	var events []jobcontext.Event
	job, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, func(j *jobcontext.Job) error {
		for e := range j.Events() {
			events = append(events, e)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("job: %v", err)
	}

	if len(events) != 1 {
		t.Fatalf("got %d events, want 1: %v", len(events), events)
	}
	if events[0].Type != jobcontext.EventUpdate {
		t.Errorf("event type = %q, want update", events[0].Type)
	}
	if want := (map[string]any{"done": true}); !reflect.DeepEqual(job.Response(), want) {
		t.Errorf("response = %v, want %v", job.Response(), want)
	}
}

func TestJobStreamsData(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStreamingWorker("worker",
		[]map[string]any{{"text": "hello"}, {"text": "world"}},
		map[string]any{"ok": true}))

	var events []jobcontext.Event
	job, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, func(j *jobcontext.Job) error {
		for e := range j.Events() {
			events = append(events, e)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("job: %v", err)
	}

	wantTypes := []string{
		jobcontext.EventStreamStart,
		jobcontext.EventStreamData,
		jobcontext.EventStreamData,
		jobcontext.EventStreamEnd,
	}
	var gotTypes []string
	for _, e := range events {
		gotTypes = append(gotTypes, e.Type)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	if want := (map[string]any{"ok": true}); !reflect.DeepEqual(job.Response(), want) {
		t.Errorf("response = %v, want %v", job.Response(), want)
	}
}

func TestJobCancelsWhenBlockFails(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStubWorker("worker"))

	wantErr := errBlockFailed
	if _, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, func(*jobcontext.Job) error {
		return wantErr
	}); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want the block's own", err)
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].Reason != "context exited with error" {
		t.Errorf("reason = %q", cancels[0].Reason)
	}
}

func TestJobReportsWorkerError(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"error": "boom"}, jobcontext.JobError))

	_, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, nil)
	if !errors.Is(err, jobcontext.ErrJob) {
		t.Fatalf("error = %v, want an ErrJob", err)
	}
	// A single-worker job reports its own error, not the group's.
	if errors.Is(err, jobcontext.ErrGroup) {
		t.Errorf("error = %v, want it not to be an ErrGroup", err)
	}
}

func TestJobExposesJobID(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newJobWorker("worker", map[string]any{"ok": true}, jobcontext.JobCompleted))

	if _, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, func(j *jobcontext.Job) error {
		if j.JobID() == "" {
			t.Error("job id was empty inside the block")
		}
		return nil
	}); err != nil {
		t.Fatalf("job: %v", err)
	}
}

func TestJobGroupCancelsOnContextCancellation(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStubWorker("worker"))

	// What a tool being interrupted mid-call looks like from here: the block
	// does not finish, and it is the context that ended it.
	_, err := parent.JobGroup(env.ctx, []string{"worker"}, jobcontext.Request{}, func(*jobcontext.Group) error {
		return context.Canceled
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want the cancellation", err)
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].Reason != "context exited with error" {
		t.Errorf("reason = %q", cancels[0].Reason)
	}
	if got := len(parent.JobGroups()); got != 0 {
		t.Errorf("%d groups left running, want none", got)
	}
}

func TestJobCancelsOnContextCancellation(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStubWorker("worker"))

	_, err := parent.Job(env.ctx, "worker", jobcontext.Request{}, func(*jobcontext.Job) error {
		return context.Canceled
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want the cancellation", err)
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].Reason != "context exited with error" {
		t.Errorf("reason = %q", cancels[0].Reason)
	}
	if got := len(parent.JobGroups()); got != 0 {
		t.Errorf("%d groups left running, want none", got)
	}
}

func TestFireAndForgetJobsCancelledByHand(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.setup(newStubWorker("w1"))
	env.setup(newStubWorker("w2"))

	// What a caller does when its own work is interrupted: it keeps the ids of
	// the jobs it started and calls off only those.
	var jobIDs []string
	for i, workerName := range []string{"w1", "w2"} {
		jobID, err := parent.RequestJob(env.ctx, workerName, jobcontext.Request{
			Payload: map[string]any{"job": i + 1},
		})
		if err != nil {
			t.Fatalf("request job: %v", err)
		}
		jobIDs = append(jobIDs, jobID)
	}
	if got := len(parent.JobGroups()); got != 2 {
		t.Fatalf("%d groups running, want 2", got)
	}

	for _, jobID := range jobIDs {
		parent.CancelGroup(env.ctx, jobID, "tool canceled")
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 2)
	if len(cancels) != 2 {
		t.Fatalf("got %d cancels, want 2", len(cancels))
	}
	targets := map[string]bool{}
	for _, m := range cancels {
		targets[m.Target()] = true
		if m.Reason != "tool canceled" {
			t.Errorf("reason = %q, want the caller's", m.Reason)
		}
	}
	if !maps.Equal(targets, map[string]bool{"w1": true, "w2": true}) {
		t.Errorf("called off %v, want w1 and w2", targets)
	}
	if got := len(parent.JobGroups()); got != 0 {
		t.Errorf("%d groups left running, want none", got)
	}
}

func TestCancelInterruptsRunningHandler(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	worker := newSlowWorker("worker")
	env.setup(worker)

	jobID, err := parent.RequestJob(env.ctx, "worker", jobcontext.Request{
		Payload: map[string]any{"job": 1},
	})
	if err != nil {
		t.Fatalf("request job: %v", err)
	}

	select {
	case <-worker.started:
	case <-time.After(2 * time.Second):
		t.Fatal("the handler never started")
	}

	parent.CancelGroup(env.ctx, jobID, "no longer needed")

	eventually(t, "the running handler is called off", worker.canceled)

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].Reason != "no longer needed" {
		t.Errorf("reason = %q", cancels[0].Reason)
	}

	// The worker answers with the canceled status, the same way it would
	// answer a job that completed or failed.
	responses := await[*bus.JobResponseMessage](t, env.recorder, 1)
	if len(responses) != 1 {
		t.Fatalf("got %d answers, want 1", len(responses))
	}
	if responses[0].Status != jobcontext.JobCancelled {
		t.Errorf("status = %q, want the job called off", responses[0].Status)
	}
}
