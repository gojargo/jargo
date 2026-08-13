package jobcontext_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/gojargo/jargo/pipeline/jobcontext"
)

// Upstream's job-group suite drives everything through the worker that runs a
// group, so there is nothing there to port onto the group itself. These are
// ours, and cover the contract the worker will depend on.

func newGroup() *jobcontext.Group {
	return jobcontext.NewGroup("job-1", []string{"w1", "w2"}, true)
}

func TestGroupCollectsResponses(t *testing.T) {
	g := newGroup()
	g.SetResponse("w1", map[string]any{"ok": true})

	got, ok := g.Response("w1")
	if !ok || got["ok"] != true {
		t.Errorf("Response(w1) = %v, %v, want the recorded response", got, ok)
	}
	if _, ok := g.Response("w2"); ok {
		t.Error("Response(w2) reported a response that never arrived")
	}
	if len(g.Responses()) != 1 {
		t.Errorf("Responses() holds %d, want 1", len(g.Responses()))
	}
}

// Responses is a copy: a caller holding it cannot rewrite the group's own.
func TestGroupResponsesIsACopy(t *testing.T) {
	g := newGroup()
	g.SetResponse("w1", map[string]any{"ok": true})

	got := g.Responses()
	delete(got, "w1")

	if _, ok := g.Response("w1"); !ok {
		t.Error("changing the returned map changed the group's own responses")
	}
}

func TestGroupCompletes(t *testing.T) {
	g := newGroup()
	if g.IsDone() {
		t.Fatal("a fresh group reported itself done")
	}

	g.Complete()

	if !g.IsDone() {
		t.Error("the group did not report itself done after completing")
	}
	if err := g.Wait(t.Context()); err != nil {
		t.Errorf("Wait after Complete = %v, want nil", err)
	}
}

func TestGroupFails(t *testing.T) {
	g := newGroup()
	g.Fail("a worker gave up")

	if !g.IsDone() {
		t.Error("the group did not report itself done after failing")
	}
	err := g.Wait(t.Context())
	if !errors.Is(err, jobcontext.ErrGroup) {
		t.Errorf("Wait after Fail = %v, want it to be an ErrGroup", err)
	}
	if err.Error() == jobcontext.ErrGroup.Error() {
		t.Error("the failure did not carry its reason")
	}
}

// A group cut short keeps whatever arrived before it was.
func TestGroupKeepsPartialResponsesOnFailure(t *testing.T) {
	g := newGroup()
	g.SetResponse("w1", map[string]any{"ok": true})
	g.Fail("w2 gave up")

	if _, ok := g.Response("w1"); !ok {
		t.Error("the response that did arrive was lost when the group failed")
	}
}

// Finishing twice keeps the first outcome, so a late failure cannot rewrite a
// group that already completed.
func TestGroupFinishesOnlyOnce(t *testing.T) {
	g := newGroup()
	g.Complete()
	g.Fail("too late")

	if err := g.Wait(t.Context()); err != nil {
		t.Errorf("Wait = %v, want the first outcome to stand", err)
	}
}

// The event channel closes when the group finishes, which is what ends a caller
// ranging over it.
func TestGroupEventsCloseOnCompletion(t *testing.T) {
	g := newGroup()
	g.Report(jobcontext.GroupEvent{Type: jobcontext.EventUpdate, WorkerName: "w1"})
	g.Report(jobcontext.GroupEvent{Type: jobcontext.EventStreamData, WorkerName: "w2"})
	g.Complete()

	var got []jobcontext.GroupEvent
	for e := range g.Events() {
		got = append(got, e)
	}

	if len(got) != 2 {
		t.Fatalf("received %d events, want 2", len(got))
	}
	if got[0].Type != jobcontext.EventUpdate || got[1].WorkerName != "w2" {
		t.Errorf("events arrived as %+v, want them in the order reported", got)
	}
}

func TestGroupEventsCloseOnFailure(t *testing.T) {
	g := newGroup()
	g.Fail("gave up")

	for range g.Events() {
		t.Error("an event arrived for a group that reported none")
	}
}

// Reporting after the group has finished is a no-op rather than a panic on a
// closed channel: a worker may report as the group is being called off.
func TestGroupReportAfterFinishIsIgnored(t *testing.T) {
	g := newGroup()
	g.Complete()
	g.Report(jobcontext.GroupEvent{Type: jobcontext.EventUpdate, WorkerName: "w1"})

	for range g.Events() {
		t.Error("an event reported after the group finished was delivered")
	}
}

// A caller that never reads the events must not hold up the worker doing the
// job, so reporting never blocks.
func TestGroupReportNeverBlocks(t *testing.T) {
	g := newGroup()

	done := make(chan struct{})
	go func() {
		for range 1000 {
			g.Report(jobcontext.GroupEvent{Type: jobcontext.EventUpdate, WorkerName: "w1"})
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("reporting blocked on a caller that was not reading")
	}
	g.Complete()
}

// Wait returns when the caller's context ends, so a caller is never stuck on a
// group that never finishes.
func TestGroupWaitHonorsTheContext(t *testing.T) {
	g := newGroup()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if err := g.Wait(ctx); err == nil {
		t.Error("Wait returned nil for a context that had ended, want its error")
	}
	if g.IsDone() {
		t.Error("the group reported itself done because the caller gave up")
	}
}
