package workers_test

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/workers"
)

// A worker's own lifecycle: becoming active, ending, canceling, and the
// children whose lifetime it manages.
//
// Upstream drives some of these through a pipeline worker and its runner,
// because that is what its harness has to hand. Neither exists here yet, so the
// ones whose subject is a pipeline (queueing frames, the bus edges, speaking)
// are left for that port; the ones below are the base's own behavior and are
// driven straight through it.

func TestStartedAtUnsetBeforeStart(t *testing.T) {
	t.Parallel()
	worker := newStubWorker("test")
	if got := worker.StartedAt(); got != 0 {
		t.Errorf("started at = %v before starting, want zero", got)
	}
}

func TestActivationArgsNilBeforeActivation(t *testing.T) {
	t.Parallel()
	inactive := false
	worker := &stubWorker{}
	worker.Base = workers.New(workers.Config{Name: "test", Active: &inactive}, worker)

	if got := worker.ActivationArgs(); got != nil {
		t.Errorf("activation args = %v before any activation, want nil", got)
	}
}

func TestStartFiresActivatedWhenBuiltActive(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("test")
	env.setup(worker)

	activated := make(chan struct{})
	var once sync.Once
	worker.Add(workers.EventActivated, func(context.Context, any, ...any) {
		once.Do(func() { close(activated) })
	})

	worker.Start(env.ctx)

	select {
	case <-activated:
	case <-time.After(2 * time.Second):
		t.Fatal("the activation was never reported")
	}
	if !worker.Active() {
		t.Error("worker is not active after starting")
	}
}

func TestActivationArgsSetAndCleared(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	inactive := false
	worker := &stubWorker{}
	worker.Base = workers.New(workers.Config{Name: "test", Active: &inactive}, worker)
	env.setup(worker)
	worker.Start(env.ctx)

	args := map[string]any{"messages": []string{"hello"}}
	activate := &bus.ActivateWorkerMessage{Args: args}
	activate.From = "other"
	activate.To = "test"
	worker.OnBusMessage(env.ctx, activate)

	if !worker.Active() {
		t.Error("worker is not active after being activated")
	}
	if got := worker.ActivationArgs(); !reflect.DeepEqual(got, args) {
		t.Errorf("activation args = %v, want %v", got, args)
	}

	deactivate := &bus.DeactivateWorkerMessage{}
	deactivate.From = "other"
	deactivate.To = "test"
	worker.OnBusMessage(env.ctx, deactivate)

	if worker.Active() {
		t.Error("worker is still active after being deactivated")
	}
	if got := worker.ActivationArgs(); got != nil {
		t.Errorf("activation args = %v after deactivating, want nil", got)
	}
}

func TestDeactivateSelfTakesEffectBeforeActivateIsSent(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("test")
	env.setup(worker)
	env.recorder.watching = worker.Active

	if !worker.Active() {
		t.Fatal("worker did not start active")
	}
	worker.ActivateWorker(env.ctx, "other", workers.ActivateOptions{DeactivateSelf: true})

	// Flipped there and then, without waiting for the bus to carry the message
	// and bring it back.
	if worker.Active() {
		t.Error("worker is still active after handing off")
	}

	await[*bus.ActivateWorkerMessage](t, env.recorder, 1)
	var order []string
	for _, m := range env.recorder.messages() {
		switch m.(type) {
		case *bus.DeactivateWorkerMessage:
			order = append(order, "deactivate")
		case *bus.ActivateWorkerMessage:
			order = append(order, "activate")
		}
	}
	if !reflect.DeepEqual(order, []string{"deactivate", "activate"}) {
		t.Errorf("message order = %v, want deactivate then activate", order)
	}
	// Both went out with this worker already inactive.
	for i, active := range env.recorder.activeFlags() {
		if active {
			t.Errorf("message %d went out with the worker still active", i)
		}
	}
}

func TestActivateWorkerWithDeactivateSelfSendsBoth(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker_a")
	env.setup(worker)

	worker.ActivateWorker(env.ctx, "worker_b", workers.ActivateOptions{DeactivateSelf: true})

	deactivates := await[*bus.DeactivateWorkerMessage](t, env.recorder, 1)
	if len(deactivates) != 1 {
		t.Fatalf("got %d deactivations, want 1", len(deactivates))
	}
	if deactivates[0].Target() != "worker_a" {
		t.Errorf("deactivation target = %q, want worker_a", deactivates[0].Target())
	}
	activates := await[*bus.ActivateWorkerMessage](t, env.recorder, 1)
	if len(activates) != 1 {
		t.Fatalf("got %d activations, want 1", len(activates))
	}
	if activates[0].Target() != "worker_b" {
		t.Errorf("activation target = %q, want worker_b", activates[0].Target())
	}
}

func TestEndWithoutParentSendsEndMessage(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker_a")
	env.setup(worker)
	worker.End(env.ctx, "done")

	ends := await[*bus.EndMessage](t, env.recorder, 1)
	if len(ends) != 1 {
		t.Fatalf("got %d end messages, want 1", len(ends))
	}
	if ends[0].Source() != "worker_a" {
		t.Errorf("source = %q, want worker_a", ends[0].Source())
	}
	if ends[0].Reason != "done" {
		t.Errorf("reason = %q, want done", ends[0].Reason)
	}
}

func TestEndWithParentSendsEndMessage(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	child := newStubWorker("child")
	env.setup(child)
	parent.AddWorkers(env.ctx, child)

	child.End(env.ctx, "goodbye")

	// A child still asks for the session to end; the runner is what acts on it.
	ends := await[*bus.EndMessage](t, env.recorder, 1)
	if len(ends) != 1 {
		t.Fatalf("got %d end messages, want 1", len(ends))
	}
	if ends[0].Source() != "child" {
		t.Errorf("source = %q, want child", ends[0].Source())
	}
	if ends[0].Reason != "goodbye" {
		t.Errorf("reason = %q, want goodbye", ends[0].Reason)
	}
}

func TestCancelSendsCancelMessage(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker_a")
	env.setup(worker)
	worker.Cancel(env.ctx, "")

	cancels := await[*bus.CancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancel messages, want 1", len(cancels))
	}
	if cancels[0].Source() != "worker_a" {
		t.Errorf("source = %q, want worker_a", cancels[0].Source())
	}
}

func TestAddWorkersSendsAddWorkerMessage(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker_a")
	env.setup(worker)
	added := newStubWorker("worker_b")
	worker.AddWorkers(env.ctx, added)

	msgs := await[*bus.AddWorkerMessage](t, env.recorder, 1)
	if len(msgs) != 1 {
		t.Fatalf("got %d add-worker messages, want 1", len(msgs))
	}
	if msgs[0].Worker != workers.Worker(added) {
		t.Errorf("the message carries %v, want the worker that was added", msgs[0].Worker)
	}
}

func TestAddWorkersTracksChildren(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	childA := newStubWorker("child_a")
	childB := newStubWorker("child_b")

	parent.AddWorkers(env.ctx, childA)
	parent.AddWorkers(env.ctx, childB)

	children := parent.Children()
	if len(children) != 2 {
		t.Fatalf("got %d children, want 2", len(children))
	}
	if children[0] != workers.Worker(childA) || children[1] != workers.Worker(childB) {
		t.Errorf("children = %v, want the two that were added, in order", children)
	}
	if got := childA.Parent(); got != "parent" {
		t.Errorf("child parent = %q, want parent", got)
	}
}

func TestEndPropagatesToChildren(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	childA := newStubWorker("child_a")
	childB := newStubWorker("child_b")
	parent.AddWorkers(env.ctx, childA, childB)

	// Finished up front, so the parent's wait for them returns at once.
	childA.Stop(env.ctx)
	childB.Stop(env.ctx)

	end := &bus.EndWorkerMessage{Reason: "shutdown"}
	end.From = "runner"
	end.To = "parent"
	parent.OnBusMessage(env.ctx, end)

	msgs := await[*bus.EndWorkerMessage](t, env.recorder, 2)
	targets := map[string]bool{}
	for _, m := range msgs {
		targets[m.Target()] = true
	}
	for _, want := range []string{"child_a", "child_b"} {
		if !targets[want] {
			t.Errorf("no end reached %s, targets were %v", want, targets)
		}
	}
}

func TestEndWaitsForChildren(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	child := newStubWorker("child")
	parent.AddWorkers(env.ctx, child)

	var mu sync.Mutex
	var order []string

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond)
		mu.Lock()
		order = append(order, "child finished")
		mu.Unlock()
		child.Stop(env.ctx)
	}()
	go func() {
		defer wg.Done()
		time.Sleep(50 * time.Millisecond)
		end := &bus.EndWorkerMessage{Reason: "shutdown"}
		end.From = "runner"
		end.To = "parent"
		parent.OnBusMessage(env.ctx, end)
		mu.Lock()
		order = append(order, "parent end returned")
		mu.Unlock()
	}()
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if want := []string{"child finished", "parent end returned"}; !reflect.DeepEqual(order, want) {
		t.Errorf("order = %v, want %v", order, want)
	}
}

func TestCancelPropagatesToChildren(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	childA := newStubWorker("child_a")
	childB := newStubWorker("child_b")
	parent.AddWorkers(env.ctx, childA, childB)

	cancel := &bus.CancelWorkerMessage{Reason: "abort"}
	cancel.From = "runner"
	cancel.To = "parent"
	parent.OnBusMessage(env.ctx, cancel)

	msgs := await[*bus.CancelWorkerMessage](t, env.recorder, 2)
	targets := map[string]bool{}
	for _, m := range msgs {
		targets[m.Target()] = true
	}
	for _, want := range []string{"child_a", "child_b"} {
		if !targets[want] {
			t.Errorf("no cancel reached %s, targets were %v", want, targets)
		}
	}
}

// A job seen from both ends: asking for one, being asked, answering, reporting
// progress, streaming results and calling it off.

func TestRequestJobSendsRequest(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("worker")

	jobID, err := parent.RequestJob(env.ctx, "worker", jobcontext.Request{
		Payload: map[string]any{"key": "val"},
	})
	if err != nil {
		t.Fatalf("request job: %v", err)
	}

	requests := await[*bus.JobRequestMessage](t, env.recorder, 1)
	if len(requests) != 1 {
		t.Fatalf("got %d job requests, want 1", len(requests))
	}
	if requests[0].JobID != jobID {
		t.Errorf("job id = %q, want %q", requests[0].JobID, jobID)
	}
	if requests[0].Target() != "worker" {
		t.Errorf("target = %q, want worker", requests[0].Target())
	}
	if want := (map[string]any{"key": "val"}); !reflect.DeepEqual(requests[0].Payload, want) {
		t.Errorf("payload = %v, want %v", requests[0].Payload, want)
	}
}

func TestRequestJobGroupSendsToEachWorker(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("w1", "w2")

	jobID, err := parent.RequestJobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{
		Payload: map[string]any{"work": true},
	})
	if err != nil {
		t.Fatalf("request job group: %v", err)
	}

	requests := await[*bus.JobRequestMessage](t, env.recorder, 2)
	if len(requests) != 2 {
		t.Fatalf("got %d job requests, want 2", len(requests))
	}
	targets := map[string]bool{}
	for _, m := range requests {
		targets[m.Target()] = true
		if m.JobID != jobID {
			t.Errorf("job id = %q, want the group's %q", m.JobID, jobID)
		}
	}
	if !maps.Equal(targets, map[string]bool{"w1": true, "w2": true}) {
		t.Errorf("targets = %v, want w1 and w2", targets)
	}
}

func TestJobRequestFiresOnJobRequest(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)

	var mu sync.Mutex
	var received []*bus.JobRequestMessage
	worker.Add(workers.EventJobRequest, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(*bus.JobRequestMessage)
		if !ok {
			return
		}
		mu.Lock()
		received = append(received, typed)
		mu.Unlock()
	})

	request := &bus.JobRequestMessage{JobID: "t1", Payload: map[string]any{"x": 1}}
	request.From = "parent"
	request.To = "worker"
	worker.OnBusMessage(env.ctx, request)

	eventually(t, "the job request is reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(received) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	if received[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", received[0].JobID)
	}
	if received[0].Source() != "parent" {
		t.Errorf("source = %q, want parent", received[0].Source())
	}
	if want := (map[string]any{"x": 1}); !reflect.DeepEqual(received[0].Payload, want) {
		t.Errorf("payload = %v, want %v", received[0].Payload, want)
	}
	if _, ok := worker.ActiveJobs()["t1"]; !ok {
		t.Error("the job is not among the active ones")
	}
}

func TestSendJobResponse(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	err := worker.SendJobResponse(env.ctx, "t1", map[string]any{"result": 42}, workers.JobResponseOptions{})
	if err != nil {
		t.Fatalf("send job response: %v", err)
	}

	responses := await[*bus.JobResponseMessage](t, env.recorder, 1)
	if len(responses) != 1 {
		t.Fatalf("got %d answers, want 1", len(responses))
	}
	if responses[0].Target() != "parent" {
		t.Errorf("target = %q, want parent", responses[0].Target())
	}
	if responses[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", responses[0].JobID)
	}
	if want := (map[string]any{"result": 42}); !reflect.DeepEqual(responses[0].Response, want) {
		t.Errorf("response = %v, want %v", responses[0].Response, want)
	}
	if responses[0].Status != jobcontext.JobCompleted {
		t.Errorf("status = %q, want completed", responses[0].Status)
	}
}

func TestSendJobUpdate(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	if err := worker.SendJobUpdate(env.ctx, "t1", map[string]any{"progress": 50}, false); err != nil {
		t.Fatalf("send job update: %v", err)
	}

	updates := await[*bus.JobUpdateMessage](t, env.recorder, 1)
	if len(updates) != 1 {
		t.Fatalf("got %d updates, want 1", len(updates))
	}
	if updates[0].Target() != "parent" {
		t.Errorf("target = %q, want parent", updates[0].Target())
	}
	if updates[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", updates[0].JobID)
	}
	if want := (map[string]any{"progress": 50}); !reflect.DeepEqual(updates[0].Update, want) {
		t.Errorf("update = %v, want %v", updates[0].Update, want)
	}
}

func TestJobCompletedFiresWhenAllRespond(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("w1", "w2")

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

	jobID, err := parent.RequestJobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{})
	if err != nil {
		t.Fatalf("request job group: %v", err)
	}

	// The first answer does not finish the group.
	parent.OnBusMessage(env.ctx, answer("w1", "parent", jobID, jobcontext.JobCompleted, map[string]any{"a": 1}))
	settle()
	mu.Lock()
	if len(completed) != 0 {
		t.Errorf("the group finished after one answer of two")
	}
	mu.Unlock()

	// The second does.
	parent.OnBusMessage(env.ctx, answer("w2", "parent", jobID, jobcontext.JobCompleted, map[string]any{"b": 2}))

	eventually(t, "the group finishes", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(completed) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	if completed[0].JobID != jobID {
		t.Errorf("job id = %q, want %q", completed[0].JobID, jobID)
	}
	want := map[string]map[string]any{"w1": {"a": 1}, "w2": {"b": 2}}
	if !reflect.DeepEqual(completed[0].Responses, want) {
		t.Errorf("responses = %v, want %v", completed[0].Responses, want)
	}
}

func TestCancelGroupSendsCancelToEachWorker(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("w1", "w2")

	jobID, err := parent.RequestJobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{})
	if err != nil {
		t.Fatalf("request job group: %v", err)
	}
	await[*bus.JobRequestMessage](t, env.recorder, 2)
	env.recorder.clear()

	parent.CancelGroup(env.ctx, jobID, "no longer needed")

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 2)
	if len(cancels) != 2 {
		t.Fatalf("got %d cancels, want 2", len(cancels))
	}
	targets := map[string]bool{}
	for _, m := range cancels {
		targets[m.Target()] = true
		if m.JobID != jobID {
			t.Errorf("job id = %q, want %q", m.JobID, jobID)
		}
		if m.Reason != "no longer needed" {
			t.Errorf("reason = %q", m.Reason)
		}
	}
	if !maps.Equal(targets, map[string]bool{"w1": true, "w2": true}) {
		t.Errorf("targets = %v, want w1 and w2", targets)
	}
}

func TestSendJobResponseWithoutActiveJob(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)

	err := worker.SendJobResponse(env.ctx, "unknown", map[string]any{"result": 1}, workers.JobResponseOptions{})
	if !errors.Is(err, workers.ErrNoActiveJob) {
		t.Errorf("error = %v, want ErrNoActiveJob", err)
	}
}

func TestSendJobUpdateWithoutActiveJob(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)

	err := worker.SendJobUpdate(env.ctx, "unknown", map[string]any{"progress": 50}, false)
	if !errors.Is(err, workers.ErrNoActiveJob) {
		t.Errorf("error = %v, want ErrNoActiveJob", err)
	}
}

func TestCancelAnswersAsCancelled(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	cancel := &bus.JobCancelMessage{JobID: "t1"}
	cancel.From = "parent"
	cancel.To = "worker"
	worker.OnBusMessage(env.ctx, cancel)

	if _, ok := worker.ActiveJobs()["t1"]; ok {
		t.Error("the canceled job is still among the active ones")
	}
	responses := await[*bus.JobResponseMessage](t, env.recorder, 1)
	if responses[0].Status != jobcontext.JobCancelled {
		t.Errorf("status = %q, want the job called off", responses[0].Status)
	}
}

func TestCancelFiresOnJobCancelled(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)

	var mu sync.Mutex
	var received []*bus.JobCancelMessage
	worker.Add(workers.EventJobCancelled, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(*bus.JobCancelMessage)
		if !ok {
			return
		}
		mu.Lock()
		received = append(received, typed)
		mu.Unlock()
	})

	giveJob(env, worker, "t1")

	cancel := &bus.JobCancelMessage{JobID: "t1", Reason: "no longer needed"}
	cancel.From = "parent"
	cancel.To = "worker"
	worker.OnBusMessage(env.ctx, cancel)

	eventually(t, "the cancellation is reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(received) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	if received[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", received[0].JobID)
	}
	if received[0].Reason != "no longer needed" {
		t.Errorf("reason = %q", received[0].Reason)
	}
}

func TestSendJobStreamStart(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	if err := worker.SendJobStreamStart(env.ctx, "t1", map[string]any{"content_type": "text"}); err != nil {
		t.Fatalf("send stream start: %v", err)
	}

	msgs := await[*bus.JobStreamStartMessage](t, env.recorder, 1)
	if msgs[0].Target() != "parent" {
		t.Errorf("target = %q, want parent", msgs[0].Target())
	}
	if msgs[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", msgs[0].JobID)
	}
	if want := (map[string]any{"content_type": "text"}); !reflect.DeepEqual(msgs[0].Data, want) {
		t.Errorf("data = %v, want %v", msgs[0].Data, want)
	}
}

func TestSendJobStreamData(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	if err := worker.SendJobStreamData(env.ctx, "t1", map[string]any{"text": "hello "}); err != nil {
		t.Fatalf("send stream data: %v", err)
	}

	msgs := await[*bus.JobStreamDataMessage](t, env.recorder, 1)
	if msgs[0].Target() != "parent" {
		t.Errorf("target = %q, want parent", msgs[0].Target())
	}
	if msgs[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", msgs[0].JobID)
	}
	if want := (map[string]any{"text": "hello "}); !reflect.DeepEqual(msgs[0].Data, want) {
		t.Errorf("data = %v, want %v", msgs[0].Data, want)
	}
}

func TestSendJobStreamEnd(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	if err := worker.SendJobStreamEnd(env.ctx, "t1", map[string]any{"final": true}); err != nil {
		t.Fatalf("send stream end: %v", err)
	}

	msgs := await[*bus.JobStreamEndMessage](t, env.recorder, 1)
	if msgs[0].Target() != "parent" {
		t.Errorf("target = %q, want parent", msgs[0].Target())
	}
	if msgs[0].JobID != "t1" {
		t.Errorf("job id = %q, want t1", msgs[0].JobID)
	}
	if want := (map[string]any{"final": true}); !reflect.DeepEqual(msgs[0].Data, want) {
		t.Errorf("data = %v, want %v", msgs[0].Data, want)
	}
}

func TestStreamEndEndsTheJob(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)
	giveJob(env, worker, "t1")

	var mu sync.Mutex
	var canceled []*bus.JobCancelMessage
	worker.Add(workers.EventJobCancelled, func(_ context.Context, _ any, args ...any) {
		typed, ok := args[0].(*bus.JobCancelMessage)
		if !ok {
			return
		}
		mu.Lock()
		canceled = append(canceled, typed)
		mu.Unlock()
	})

	if err := worker.SendJobStreamEnd(env.ctx, "t1", map[string]any{"final": true}); err != nil {
		t.Fatalf("send stream end: %v", err)
	}
	if _, ok := worker.ActiveJobs()["t1"]; ok {
		t.Error("the job is still active after its stream ended")
	}

	// A cancel arriving after the job has finished finds nothing to call off.
	cancel := &bus.JobCancelMessage{JobID: "t1"}
	cancel.From = "parent"
	cancel.To = "worker"
	worker.OnBusMessage(env.ctx, cancel)
	settle()

	mu.Lock()
	defer mu.Unlock()
	if len(canceled) != 0 {
		t.Errorf("the late cancel was reported, want it ignored")
	}
	for _, m := range of[*bus.JobResponseMessage](env.recorder) {
		if m.Status == jobcontext.JobCancelled {
			t.Errorf("a canceled answer went out for a job that had finished")
		}
	}
}

func TestStreamEndFinishesTheGroup(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("w1", "w2")

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

	jobID, err := parent.RequestJobGroup(env.ctx, []string{"w1", "w2"}, jobcontext.Request{})
	if err != nil {
		t.Fatalf("request job group: %v", err)
	}

	parent.OnBusMessage(env.ctx, streamEnd("w1", "parent", jobID, map[string]any{"result": "a"}))
	settle()
	mu.Lock()
	if len(completed) != 0 {
		t.Errorf("the group finished after one stream of two")
	}
	mu.Unlock()

	parent.OnBusMessage(env.ctx, streamEnd("w2", "parent", jobID, map[string]any{"result": "b"}))

	eventually(t, "the group finishes", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(completed) == 1
	})

	mu.Lock()
	defer mu.Unlock()
	if completed[0].JobID != jobID {
		t.Errorf("job id = %q, want %q", completed[0].JobID, jobID)
	}
	want := map[string]map[string]any{"w1": {"result": "a"}, "w2": {"result": "b"}}
	if !reflect.DeepEqual(completed[0].Responses, want) {
		t.Errorf("responses = %v, want %v", completed[0].Responses, want)
	}
}

func TestSendJobStreamWithoutActiveJob(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	worker := newStubWorker("worker")
	env.setup(worker)

	if err := worker.SendJobStreamStart(env.ctx, "unknown", nil); !errors.Is(err, workers.ErrNoActiveJob) {
		t.Errorf("stream start error = %v, want ErrNoActiveJob", err)
	}
	if err := worker.SendJobStreamData(env.ctx, "unknown", nil); !errors.Is(err, workers.ErrNoActiveJob) {
		t.Errorf("stream data error = %v, want ErrNoActiveJob", err)
	}
	if err := worker.SendJobStreamEnd(env.ctx, "unknown", nil); !errors.Is(err, workers.ErrNoActiveJob) {
		t.Errorf("stream end error = %v, want ErrNoActiveJob", err)
	}
}

func TestJobTimeoutCancelsTheJob(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("worker")

	jobID, err := parent.RequestJob(env.ctx, "worker", jobcontext.Request{
		Timeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("request job: %v", err)
	}

	cancels := await[*bus.JobCancelMessage](t, env.recorder, 1)
	if len(cancels) != 1 {
		t.Fatalf("got %d cancels, want 1", len(cancels))
	}
	if cancels[0].JobID != jobID {
		t.Errorf("job id = %q, want %q", cancels[0].JobID, jobID)
	}
	if cancels[0].Reason != "timeout" {
		t.Errorf("reason = %q, want timeout", cancels[0].Reason)
	}

	parent.Cleanup(env.ctx)
}

func TestJobTimeoutCalledOffOnCompletion(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("worker")

	jobID, err := parent.RequestJob(env.ctx, "worker", jobcontext.Request{
		Timeout: 500 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("request job: %v", err)
	}

	// Answered before the timeout is up.
	parent.OnBusMessage(env.ctx, answer("worker", "parent", jobID, jobcontext.JobCompleted, map[string]any{"ok": true}))

	// Waited out past when the timeout would have fired.
	time.Sleep(600 * time.Millisecond)

	if cancels := of[*bus.JobCancelMessage](env.recorder); len(cancels) != 0 {
		t.Errorf("got %d cancels, want none: the timeout should have been called off", len(cancels))
	}

	parent.Cleanup(env.ctx)
}

func TestJobHasNoTimeoutByDefault(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	parent := newStubWorker("parent")
	env.setup(parent)
	env.register("worker")

	jobID, err := parent.RequestJob(env.ctx, "worker", jobcontext.Request{})
	if err != nil {
		t.Fatalf("request job: %v", err)
	}

	group, ok := parent.JobGroups()[jobID]
	if !ok {
		t.Fatalf("no group is running for %q", jobID)
	}
	if group.HasTimeout() {
		t.Error("the group is running under a timeout, want none")
	}
}

// Declared job handlers: how a request is routed to one, and what sequential
// means for the requests that arrive while one is running.

// handlerWorker declares handlers whose bodies the test supplies, which is how
// a worker declares the kinds of work it does.
type handlerWorker struct {
	*workers.Base
}

func newHandlerWorker(name string, declare func(w *handlerWorker)) *handlerWorker {
	w := &handlerWorker{}
	w.Base = workers.New(workers.Config{Name: name}, w)
	declare(w)
	return w
}

func TestSequentialJobsRunOneAtATimeInOrder(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	var mu sync.Mutex
	running, maxRunning := 0, 0
	var completionOrder []string
	gates := map[string]chan struct{}{
		"t1": make(chan struct{}),
		"t2": make(chan struct{}),
		"t3": make(chan struct{}),
	}

	worker := newHandlerWorker("worker", func(w *handlerWorker) {
		w.HandleJob("work", workers.JobOptions{Sequential: true}, func(_ context.Context, m *bus.JobRequestMessage) {
			mu.Lock()
			running++
			maxRunning = max(maxRunning, running)
			mu.Unlock()

			<-gates[m.JobID]

			mu.Lock()
			completionOrder = append(completionOrder, m.JobID)
			running--
			mu.Unlock()
		})
	})
	env.setup(worker)

	for _, jobID := range []string{"t1", "t2", "t3"} {
		giveJobNamed(env, worker, jobID, "work")
	}

	// Let the first handler get inside.
	eventually(t, "the first handler starts", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return running == 1
	})

	// Released in order; each finishes before the next starts.
	for _, jobID := range []string{"t1", "t2", "t3"} {
		close(gates[jobID])
		eventually(t, "handler "+jobID+" finishes", func() bool {
			mu.Lock()
			defer mu.Unlock()
			return slices.Contains(completionOrder, jobID)
		})
	}

	mu.Lock()
	defer mu.Unlock()
	if want := []string{"t1", "t2", "t3"}; !reflect.DeepEqual(completionOrder, want) {
		t.Errorf("completion order = %v, want %v", completionOrder, want)
	}
	if maxRunning != 1 {
		t.Errorf("%d handlers ran at once, want 1", maxRunning)
	}
}

func TestNonSequentialJobsRunConcurrently(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	var mu sync.Mutex
	running, maxRunning := 0, 0
	release := make(chan struct{})

	worker := newHandlerWorker("worker", func(w *handlerWorker) {
		w.HandleJob("work", workers.JobOptions{}, func(context.Context, *bus.JobRequestMessage) {
			mu.Lock()
			running++
			maxRunning = max(maxRunning, running)
			mu.Unlock()

			<-release

			mu.Lock()
			running--
			mu.Unlock()
		})
	})
	env.setup(worker)

	for _, jobID := range []string{"t1", "t2"} {
		giveJobNamed(env, worker, jobID, "work")
	}

	eventually(t, "both handlers run at once", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return running == 2
	})

	mu.Lock()
	if maxRunning != 2 {
		t.Errorf("%d handlers ran at once, want 2", maxRunning)
	}
	mu.Unlock()

	close(release)
	eventually(t, "both handlers finish", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return running == 0
	})
}

func TestSequentialIsPerJobName(t *testing.T) {
	t.Parallel()
	env := newTestEnv(t)

	var mu sync.Mutex
	running, maxRunning := 0, 0
	release := make(chan struct{})

	body := func(context.Context, *bus.JobRequestMessage) {
		mu.Lock()
		running++
		maxRunning = max(maxRunning, running)
		mu.Unlock()

		<-release

		mu.Lock()
		running--
		mu.Unlock()
	}

	worker := newHandlerWorker("worker", func(w *handlerWorker) {
		w.HandleJob("a", workers.JobOptions{Sequential: true}, body)
		w.HandleJob("b", workers.JobOptions{Sequential: true}, body)
	})
	env.setup(worker)

	giveJobNamed(env, worker, "ta", "a")
	giveJobNamed(env, worker, "tb", "b")

	eventually(t, "both handlers run at once", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return running == 2
	})

	mu.Lock()
	if maxRunning != 2 {
		t.Errorf("%d handlers ran at once, want 2: one job name must not hold up another", maxRunning)
	}
	mu.Unlock()

	close(release)
	eventually(t, "both handlers finish", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return running == 0
	})
}

// giveJob hands a worker a job request from "parent", as the bus would.
func giveJob(env *testEnv, w workers.Worker, jobID string) {
	giveJobNamed(env, w, jobID, "")
}

func giveJobNamed(env *testEnv, w workers.Worker, jobID, jobName string) {
	m := &bus.JobRequestMessage{JobID: jobID, JobName: jobName}
	m.From = "parent"
	m.To = w.Name()
	w.OnBusMessage(env.ctx, m)
}

// answer builds the message a worker sends to report how its job ended.
func answer(
	source, target, jobID string, status jobcontext.JobStatus, response map[string]any,
) *bus.JobResponseMessage {
	m := &bus.JobResponseMessage{
		JobResult: bus.JobResult{JobID: jobID, Status: status, Response: response},
	}
	m.From = source
	m.To = target
	return m
}

// streamEnd builds the message that closes a worker's result stream.
func streamEnd(source, target, jobID string, data map[string]any) *bus.JobStreamEndMessage {
	m := &bus.JobStreamEndMessage{JobID: jobID, Data: data}
	m.From = source
	m.To = target
	return m
}
