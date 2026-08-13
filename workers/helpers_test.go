package workers_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/workers"
)

// The suites here are ported from upstream's worker and job-group suites.
// Upstream reaches inside a running bus by replacing its send method to record
// what went out; Go cannot, so a recorder subscribes to the bus instead and
// sees the same messages as they are delivered. Where upstream then sleeps for
// a fixed spell before asserting, these wait for the message they expect.

const runnerName = "test-runner"

// testEnv is a bus, a registry and the recorder watching them, which is what
// every test here starts from.
type testEnv struct {
	ctx      context.Context
	bus      *bus.AsyncQueueBus
	registry *registry.WorkerRegistry
	recorder *recorder
}

// newTestEnv builds a running bus with a recorder on it, and stops the bus when
// the test ends.
func newTestEnv(t *testing.T) *testEnv {
	t.Helper()
	ctx := t.Context()
	env := &testEnv{
		ctx:      ctx,
		bus:      bus.NewAsyncQueueBus(),
		registry: registry.New(runnerName),
		recorder: &recorder{name: "recorder"},
	}
	env.bus.Subscribe(env.recorder)
	env.bus.Start(ctx)
	t.Cleanup(env.bus.Stop)
	return env
}

// setup attaches a worker to the bus and registry and registers it as ready,
// which is what a runner would do for it.
func (e *testEnv) setup(w workers.Worker) {
	w.Attach(e.ctx, e.registry, e.bus.Bus)
	e.registry.Register(e.ctx, registry.WorkerReadyData{WorkerName: w.Name(), Runner: runnerName})
}

// register registers worker names as ready without a worker behind them, so a
// wait for them completes at once.
func (e *testEnv) register(names ...string) {
	for _, name := range names {
		e.registry.Register(e.ctx, registry.WorkerReadyData{WorkerName: name, Runner: runnerName})
	}
}

// recorder is a bus subscriber that keeps everything it is delivered.
type recorder struct {
	name string
	mu   sync.Mutex
	got  []bus.Message
	// watching, when set, is recorded alongside each message, for the tests
	// that care what was true of a worker as its messages went out.
	watching func() bool
	active   []bool
}

func (r *recorder) Name() string { return r.name }

func (r *recorder) OnBusMessage(_ context.Context, m bus.Message) {
	r.mu.Lock()
	r.got = append(r.got, m)
	if r.watching != nil {
		r.active = append(r.active, r.watching())
	}
	r.mu.Unlock()
}

// messages is everything recorded so far.
func (r *recorder) messages() []bus.Message {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]bus.Message(nil), r.got...)
}

// clear forgets everything recorded so far.
func (r *recorder) clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.got = nil
	r.active = nil
}

// activeFlags is what watching reported as each message was recorded.
func (r *recorder) activeFlags() []bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]bool(nil), r.active...)
}

// of picks the messages of one type out of what a recorder holds.
func of[T bus.Message](r *recorder) []T {
	var out []T
	for _, m := range r.messages() {
		if typed, ok := m.(T); ok {
			out = append(out, typed)
		}
	}
	return out
}

// await waits for the recorder to hold n messages of a type, reporting them. It
// is how a test waits for delivery instead of sleeping through it.
func await[T bus.Message](t *testing.T, r *recorder, n int) []T {
	t.Helper()
	var msgs []T
	for range 400 {
		msgs = of[T](r)
		if len(msgs) >= n {
			return msgs
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("waited for %d messages, got %d", n, len(msgs))
	return nil
}

// settle gives delivery a moment to happen, for the assertions that nothing
// arrives.
func settle() { time.Sleep(50 * time.Millisecond) }

// eventually waits for cond to hold, and fails with msg when it does not.
func eventually(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	for range 400 {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("condition not met within timeout: %s", msg)
}

// stubWorker takes part in the protocol and does nothing else, so a test can
// drive a job to a worker that never answers.
type stubWorker struct{ *workers.Base }

func newStubWorker(name string) *stubWorker {
	w := &stubWorker{}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

// jobWorker answers every job request the same way.
type jobWorker struct {
	*workers.Base
	response map[string]any
	status   jobcontext.JobStatus
	urgent   bool
}

func newJobWorker(name string, response map[string]any, status jobcontext.JobStatus) *jobWorker {
	w := &jobWorker{response: response, status: status}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

// newUrgentJobWorker answers ahead of the data messages already queued.
func newUrgentJobWorker(name string, response map[string]any, status jobcontext.JobStatus) *jobWorker {
	w := newJobWorker(name, response, status)
	w.urgent = true
	return w
}

func (w *jobWorker) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	w.Base.OnJobRequest(ctx, m)
	_ = w.SendJobResponse(ctx, m.JobID, w.response, workers.JobResponseOptions{
		Status: w.status,
		Urgent: w.urgent,
	})
}

// updatingWorker reports progress before it answers.
type updatingWorker struct {
	*workers.Base
	updates  []map[string]any
	response map[string]any
}

func newUpdatingWorker(name string, updates []map[string]any, response map[string]any) *updatingWorker {
	w := &updatingWorker{updates: updates, response: response}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

func (w *updatingWorker) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	w.Base.OnJobRequest(ctx, m)
	for _, update := range w.updates {
		_ = w.SendJobUpdate(ctx, m.JobID, update, false)
	}
	_ = w.SendJobResponse(ctx, m.JobID, w.response, workers.JobResponseOptions{})
}

// streamingWorker streams its results and closes the stream to answer.
type streamingWorker struct {
	*workers.Base
	chunks   []map[string]any
	response map[string]any
}

func newStreamingWorker(name string, chunks []map[string]any, response map[string]any) *streamingWorker {
	w := &streamingWorker{chunks: chunks, response: response}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

func (w *streamingWorker) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	w.Base.OnJobRequest(ctx, m)
	_ = w.SendJobStreamStart(ctx, m.JobID, map[string]any{"content_type": "text"})
	for _, chunk := range w.chunks {
		_ = w.SendJobStreamData(ctx, m.JobID, chunk)
	}
	_ = w.SendJobStreamEnd(ctx, m.JobID, w.response)
}

// slowWorker holds a job open until it is called off.
type slowWorker struct {
	*workers.Base
	started     chan struct{}
	startOnce   sync.Once
	mu          sync.Mutex
	wasCanceled bool
}

func newSlowWorker(name string) *slowWorker {
	w := &slowWorker{started: make(chan struct{})}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

func (w *slowWorker) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	w.Base.OnJobRequest(ctx, m)
	w.startOnce.Do(func() { close(w.started) })
	<-ctx.Done()
	w.mu.Lock()
	w.wasCanceled = true
	w.mu.Unlock()
}

func (w *slowWorker) canceled() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.wasCanceled
}

// newRegistry and readyData build the registry pieces for a test that wires a
// worker up by hand rather than through a testEnv.
func newRegistry() *registry.WorkerRegistry { return registry.New(runnerName) }

func readyData(name string) registry.WorkerReadyData {
	return registry.WorkerReadyData{WorkerName: name, Runner: runnerName}
}

// recordingWorker keeps the bus messages it is delivered, in the order they
// arrive.
//
// Upstream observes the same order through the on_bus_message event. jargo runs
// an asynchronous event's handlers on a goroutine apiece, so the order they
// record in is not the order the messages arrived in; overriding the handler
// itself observes delivery directly, which is what the ordering is about.
type recordingWorker struct {
	*workers.Base
	mu  sync.Mutex
	got []bus.Message
}

func newRecordingWorker(name string) *recordingWorker {
	w := &recordingWorker{}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}

func (w *recordingWorker) OnBusMessage(ctx context.Context, m bus.Message) {
	w.mu.Lock()
	w.got = append(w.got, m)
	w.mu.Unlock()
	w.Base.OnBusMessage(ctx, m)
}

func (w *recordingWorker) messages() []bus.Message {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]bus.Message(nil), w.got...)
}

// errBlockFailed is what a test block returns when it means to fail.
var errBlockFailed = errors.New("something went wrong")
