package llmwithbackend

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers/llmworker"
)

// --- The backend's work, as the pipeline hears of it ---

// workRecorder records the background-work frames a service broadcasts, once
// per broadcast rather than once per direction.
type workRecorder struct {
	mu   sync.Mutex
	ids  map[uint64]bool
	work []string
}

// recordWork watches what svc pushes.
func recordWork(svc *llm.Base) *workRecorder {
	r := &workRecorder{ids: make(map[uint64]bool)}
	event.On(svc.Events(), processor.EventBeforePushFrame, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		// The second half of a broadcast names the first as its sibling.
		if sibling, ok := f.Base().BroadcastSiblingID(); ok && r.ids[sibling] {
			return
		}
		r.ids[f.ID()] = true
		switch fr := f.(type) {
		case *frames.BackgroundWorkStartedFrame:
			r.work = append(r.work, "started:"+fr.WorkID)
		case *frames.BackgroundWorkFinishedFrame:
			r.work = append(r.work, "finished:"+fr.WorkID)
		}
	})
	return r
}

func (r *workRecorder) frames() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.work)
}

// delegateFrom builds a delegate call over convo made by frontend.
func delegateFrom(frontend *llm.Base, convo *frames.LLMContext) llm.FunctionCallParams {
	p, _ := params(convo, "")
	p.LLM = frontend
	return p
}

func TestADelegationStartsTheBackendsWorkOncePerLull(t *testing.T) {
	c := bound(t, nil, false, &fakeSession{})
	frontend := newTextFrontend()
	work := recordWork(frontend.Base)
	convo := conversation(user("do it"))

	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, convo)); err != nil {
		t.Fatal(err)
	}
	convo.AddMessage(user("and this"))
	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, convo)); err != nil {
		t.Fatal(err)
	}
	if got, want := work.frames(), []string{"started:backend"}; !slices.Equal(got, want) {
		t.Fatalf("work frames = %v, want %v", got, want)
	}

	c.Deliver(t.Context(), frontend.Base, llmworker.BackendIdle{})
	c.Deliver(t.Context(), frontend.Base, llmworker.BackendIdle{})
	if got, want := work.frames(), []string{"started:backend", "finished:backend"}; !slices.Equal(got, want) {
		t.Fatalf("work frames = %v, want %v", got, want)
	}

	convo.AddMessage(user("once more"))
	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, convo)); err != nil {
		t.Fatal(err)
	}
	if got := work.frames(); got[len(got)-1] != "started:backend" {
		t.Errorf("work frames = %v, want the work started again", got)
	}
}

func TestABackendErrorFinishesItsWork(t *testing.T) {
	c := bound(t, nil, false, &fakeSession{})
	frontend := newTextFrontend()
	work := recordWork(frontend.Base)

	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, conversation(user("do it")))); err != nil {
		t.Fatal(err)
	}
	// The error is also spoken, which queues into the frontend: a recording one
	// stands in for it there.
	delivered := &recordingFrontend{}
	c.Deliver(t.Context(), delivered, llmworker.BackendError{Error: "provider down"})

	if got, want := work.frames(), []string{"started:backend"}; !slices.Equal(got, want) {
		t.Errorf("work frames from the delegation = %v, want %v", got, want)
	}
	if got, want := delivered.workFrames(), []string{"finished:backend"}; !slices.Equal(got, want) {
		t.Errorf("work frames from the error = %v, want %v", got, want)
	}
}

// endingRequester stands in for the worker the frontend runs in: the attach job
// stays open until the test ends it, and each message is answered at once.
type endingRequester struct {
	mu     sync.Mutex
	attach *jobcontext.Group
}

func (r *endingRequester) CreateGroupAndRequestJob(
	_ context.Context, names []string, req jobcontext.Request,
) (*jobcontext.Group, error) {
	g := jobcontext.NewGroup(req.Name, names, false)
	if req.Name == llmworker.AttachJobName {
		g.Report(jobcontext.GroupEvent{
			Type: jobcontext.EventUpdate, WorkerName: names[0],
			Data: map[string]any{"type": llmworker.AttachedUpdateType},
		})
		r.mu.Lock()
		r.attach = g
		r.mu.Unlock()
		return g, nil
	}
	g.SetResponse(names[0], map[string]any{"backend": "working"})
	g.Complete()
	return g, nil
}

func (r *endingRequester) CancelGroup(context.Context, string, string) {}

func (r *endingRequester) HasGroup(string) bool { return false }

// end ends the attach job, the way a backend going away does.
func (r *endingRequester) end() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.attach.Complete()
}

func TestASessionEndingMidWorkFinishesIt(t *testing.T) {
	c := bound(t, NewBackendConnector(ConnectorConfig{Timeout: 3 * time.Second}), false, nil)
	frontend := newTextFrontend()
	work := recordWork(frontend.Base)
	requester := &endingRequester{}

	done := make(chan struct{})
	go func() {
		defer close(done)
		c.RunSession(t.Context(), requester, frontend.Base)
	}()
	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, conversation(user("do it")))); err != nil {
		t.Fatal(err)
	}
	if got, want := work.frames(), []string{"started:backend"}; !slices.Equal(got, want) {
		t.Fatalf("work frames = %v, want %v", got, want)
	}

	requester.end()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the session did not end")
	}

	if got, want := work.frames(), []string{"started:backend", "finished:backend"}; !slices.Equal(got, want) {
		t.Errorf("work frames = %v, want %v", got, want)
	}
}

// A session closed from this side, with the frontend going away, announces
// nothing: there is no pipeline left to hear it.
func TestASessionCanceledWithTheFrontendAnnouncesNothing(t *testing.T) {
	c := bound(t, NewBackendConnector(ConnectorConfig{Timeout: 3 * time.Second}), false, nil)
	frontend := newTextFrontend()
	work := recordWork(frontend.Base)
	requester := &endingRequester{}

	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		defer close(done)
		c.RunSession(ctx, requester, frontend.Base)
	}()
	if err := c.Delegate(t.Context(), delegateFrom(frontend.Base, conversation(user("do it")))); err != nil {
		t.Fatal(err)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the session did not end")
	}

	if got, want := work.frames(), []string{"started:backend"}; !slices.Equal(got, want) {
		t.Errorf("work frames = %v, want %v", got, want)
	}
}

// The whole path: the delegation starts the backend's work and the backend
// going idle finishes it, both broadcast through the frontend's pipeline.
func TestTheBackendsWorkReachesThePipeline(t *testing.T) {
	backend := weatherBackend(t, scriptedRun{text: ">> It's 62 and raining."})
	service, err := New(Config{Frontend: newDelegatingFrontend(""), Backend: backend})
	if err != nil {
		t.Fatal(err)
	}

	down := runService(t, service, conversation(user("weather in seattle?")))

	var work []string
	for _, f := range down {
		switch fr := f.(type) {
		case *frames.BackgroundWorkStartedFrame:
			work = append(work, "started:"+fr.WorkID)
		case *frames.BackgroundWorkFinishedFrame:
			work = append(work, "finished:"+fr.WorkID)
		}
	}
	if !slices.Contains(work, "started:backend") || !slices.Contains(work, "finished:backend") {
		t.Errorf("work frames = %v, want the backend's work started and finished", work)
	}
}
