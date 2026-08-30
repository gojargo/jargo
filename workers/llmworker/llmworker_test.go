package llmworker_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

// Ported from upstream's LLM worker suite. What it guards is the holding: a
// tool handler runs while the model is still waiting for its result, so
// anything the handler queues has to land behind that result rather than ahead
// of it. Get it wrong and the turn is delivered out of order.
//
// Upstream drives the handlers by reaching for the worker's private wrapper.
// Here the handlers are run through the ones the worker registered on the model,
// which is the same path a real call takes, and each handler queues with the
// context it was given, as a real one would.

//nolint:gochecknoglobals // sentinel error for the tests below
var errBoom = errors.New("boom")

// recordingLLM stands in for a model service, holding the tools registered on
// it so a test can run one the way a call would.
type recordingLLM struct {
	*processor.Base

	mu       sync.Mutex
	handlers map[string]llm.FunctionCallHandler
}

func newRecordingLLM() *recordingLLM {
	l := &recordingLLM{handlers: map[string]llm.FunctionCallHandler{}}
	l.Base = processor.New("RecordingLLM", l)
	return l
}

// ProcessFrame forwards everything. A real service consumes what it answers and
// pushes the rest; this one only has to let frames past so a test can see them
// at the far end.
func (l *recordingLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return l.PushFrame(ctx, f, dir)
}

func (l *recordingLLM) RegisterFunction(
	name string, h llm.FunctionCallHandler, _ ...llm.RegisterOption,
) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.handlers[name] = h
}

func (l *recordingLLM) handler(t *testing.T, name string) llm.FunctionCallHandler {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	h, ok := l.handlers[name]
	if !ok {
		t.Fatalf("no tool registered as %q", name)
	}
	return h
}

func (l *recordingLLM) registered(name string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	_, ok := l.handlers[name]
	return ok
}

// sink records the conversation appends that reach the end of the pipeline,
// which is what shows whether a held frame was released and when.
type sink struct {
	*processor.Base
	mu sync.Mutex
	// got is the text of each conversation append that arrived.
	got []string
	// tools records that the model was told which tools it may call.
	tools bool
	// ran is whether the last append asked the model to answer.
	ran *bool
}

func newSink() *sink {
	s := &sink{}
	s.Base = processor.New("Sink", s)
	return s
}

func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMMessagesAppendFrame:
		if len(fr.Messages) > 0 {
			ran := fr.RunLLM
			s.mu.Lock()
			s.got = append(s.got, fr.Messages[0].Text)
			s.ran = &ran
			s.mu.Unlock()
		}
	case *frames.LLMSetToolsFrame:
		s.mu.Lock()
		s.tools = true
		s.mu.Unlock()
	}
	return s.PushFrame(ctx, f, dir)
}

// sawTools reports whether the model was told which tools it may call.
func (s *sink) sawTools() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tools
}

// lastAppendRan is whether the last append asked the model to answer.
func (s *sink) lastAppendRan() *bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ran
}

func (s *sink) appended() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.got...)
}

// harness is a running worker with a recording model behind it.
type harness struct {
	ctx    context.Context
	worker *llmworker.Worker
	llm    *recordingLLM
	sink   *sink
	bus    *busRecorder
}

// busRecorder collects what the worker put on the bus, which is where ending
// and handing over show up.
type busRecorder struct {
	mu  sync.Mutex
	got []bus.Message
}

func (r *busRecorder) Name() string { return "recorder" }

func (r *busRecorder) OnBusMessage(_ context.Context, m bus.Message) {
	r.mu.Lock()
	r.got = append(r.got, m)
	r.mu.Unlock()
}

// seen reports whether a message of the same type as sample reached the bus.
func (r *busRecorder) seen(sample bus.Message) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	want := reflect.TypeOf(sample)
	for _, m := range r.got {
		if reflect.TypeOf(m) == want {
			return true
		}
	}
	return false
}

// toolFunc is what a tool handler does before it waits to be let go. It is
// given the call's own context, which is what marks anything it queues as the
// call's.
type toolFunc func(ctx context.Context, h *harness)

// gatedTool builds a tool that runs fn and then blocks until gate is closed,
// which is how a test holds a call open while it looks at the worker.
func gatedTool(h **harness, name string, gate <-chan struct{}, fn toolFunc) frames.Tool {
	return frames.Tool{
		Name:       name,
		Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: llm.FunctionCallHandler(func(ctx context.Context, _ llm.FunctionCallParams) error {
			if fn != nil {
				fn(ctx, *h)
			}
			<-gate
			return nil
		}),
	}
}

// newHarness stands up a running worker. The tools are built against the
// harness pointer, so a handler can reach the worker that runs it.
func newHarness(t *testing.T, holder **harness, defer_ *bool, build func(**harness) []frames.Tool) *harness {
	t.Helper()
	ctx := t.Context()
	model := newRecordingLLM()
	sk := newSink()
	off, on := false, true

	var tools []frames.Tool
	if build != nil {
		tools = build(holder)
	}

	w := llmworker.New(llmworker.Config{
		Name:            "llm-worker",
		LLM:             model,
		Pipeline:        pipeline.New(model, sk),
		Tools:           tools,
		DeferToolFrames: defer_,
		Active:          &on,
		WorkerConfig: pipeline.WorkerConfig{
			EnableRTVI:         &off,
			EnableTurnTracking: &off,
			// Nothing here speaks, so the pipeline must not be cut off for
			// being quiet.
			IdleTimeout: -1,
		},
	})

	msgBus := bus.NewAsyncQueueBus()
	rec := &busRecorder{}
	msgBus.Subscribe(rec)
	msgBus.Start(ctx)
	t.Cleanup(msgBus.Stop)
	w.Attach(ctx, registry.New("test-runner"), msgBus.Bus)

	runDone := make(chan error, 1)
	go func() { runDone <- w.Run(ctx) }()
	t.Cleanup(func() {
		// There is no runner here to turn an end request into a stop, so the
		// worker is stopped directly.
		w.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Error("the worker did not finish")
		}
	})

	h := &harness{ctx: ctx, worker: w, llm: model, sink: sk, bus: rec}
	if holder != nil {
		*holder = h
	}

	// The worker starts inactive and its pipeline only carries anything once it
	// is going. A probe through the pipeline is what shows it is.
	w.OnActivated(ctx, nil)
	h.waitForPipeline(t)
	return h
}

// waitForPipeline waits until a queued frame reaches the far end, then forgets
// it, so a test starts from an empty sink and a running pipeline.
func (h *harness) waitForPipeline(t *testing.T) {
	t.Helper()
	h.worker.QueueFrame(h.ctx, appendFrame("__ready__"))
	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) > 0 }) {
		t.Fatal("the pipeline never carried anything")
	}
	h.sink.mu.Lock()
	h.sink.got = nil
	h.sink.mu.Unlock()
}

// run calls one of the worker's registered tools, the way the model would.
func (h *harness) run(t *testing.T, name string) error {
	t.Helper()
	return h.llm.handler(t, name)(h.ctx, llm.FunctionCallParams{
		FunctionName: name,
		ToolCallID:   "call-1",
		Arguments:    json.RawMessage(`{}`),
		Result: func(context.Context, string, *frames.FunctionCallResultProperties) error {
			return nil
		},
	})
}

// waitFor polls until cond holds.
func waitFor(timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond()
}

// appendFrame is a conversation append carrying text.
func appendFrame(text string) *frames.LLMMessagesAppendFrame {
	return frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: text}})
}

// TestToolCallActiveTracksTheCall is upstream's
// test_tool_call_active_initially_false and test_tool_call_active_during_execution.
func TestToolCallActiveTracksTheCall(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "gated", gate, nil)}
	})

	if h.worker.ToolCallActive() {
		t.Error("a tool was reported running before any call")
	}

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "gated") }()

	if !waitFor(3*time.Second, h.worker.ToolCallActive) {
		t.Fatal("the call was not reported as running")
	}
	close(gate)
	<-done
	if !waitFor(3*time.Second, func() bool { return !h.worker.ToolCallActive() }) {
		t.Error("the call was still reported as running after it finished")
	}
}

// TestFramesGoStraightThroughWhenNoToolIsRunning is upstream's
// test_queue_frame_delivers_immediately_when_idle.
func TestFramesGoStraightThroughWhenNoToolIsRunning(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, nil)

	h.worker.QueueFrame(h.ctx, appendFrame("hello"))

	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v, want the frame delivered at once", h.sink.appended())
	}
}

// TestAToolsOwnFrameIsHeld is upstream's
// test_queue_frame_defers_a_frame_the_tool_queued.
func TestAToolsOwnFrameIsHeld(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "queueing", gate, func(ctx context.Context, h *harness) {
			h.worker.QueueFrame(ctx, appendFrame("held"))
		})}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "queueing") }()
	waitFor(3*time.Second, h.worker.ToolCallActive)

	time.Sleep(300 * time.Millisecond)
	if got := h.sink.appended(); len(got) != 0 {
		t.Errorf("the pipeline saw %v while the call ran, want nothing", got)
	}

	close(gate)
	<-done
	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v after the call, want the held frame released",
			h.sink.appended())
	}
}

// TestTrafficFromElsewhereIsNotHeld is upstream's
// test_queue_frame_does_not_defer_traffic_from_elsewhere.
func TestTrafficFromElsewhereIsNotHeld(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "blocking", gate, nil)}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "blocking") }()
	waitFor(3*time.Second, h.worker.ToolCallActive)

	// Queued with a context that was never inside the call.
	h.worker.QueueFrame(h.ctx, appendFrame("from elsewhere"))

	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v, want the outside frame delivered at once",
			h.sink.appended())
	}
	close(gate)
	<-done
}

// TestConcurrentToolsReleaseOnlyWhenTheLastFinishes is upstream's
// test_concurrent_tools_flush_only_when_all_done.
func TestConcurrentToolsReleaseOnlyWhenTheLastFinishes(t *testing.T) {
	gateA, gateB := make(chan struct{}), make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{
			gatedTool(hp, "tool_a", gateA, func(ctx context.Context, h *harness) {
				h.worker.QueueFrame(ctx, appendFrame("held"))
			}),
			gatedTool(hp, "tool_b", gateB, nil),
		}
	})

	doneA, doneB := make(chan struct{}), make(chan struct{})
	go func() { defer close(doneA); _ = h.run(t, "tool_a") }()
	go func() { defer close(doneB); _ = h.run(t, "tool_b") }()
	waitFor(3*time.Second, func() bool { return h.worker.ToolCallActive() })
	time.Sleep(200 * time.Millisecond) // let both calls get under way

	close(gateA)
	<-doneA
	time.Sleep(300 * time.Millisecond)
	if got := h.sink.appended(); len(got) != 0 {
		t.Errorf("the pipeline saw %v with a call still running, want nothing", got)
	}

	close(gateB)
	<-doneB
	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v once every call finished, want the held frame",
			h.sink.appended())
	}
}

// TestHeldFramesAreReleasedInOrder is upstream's
// test_multiple_deferred_frames_flush_in_order.
func TestHeldFramesAreReleasedInOrder(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "blocking", gate, func(ctx context.Context, h *harness) {
			h.worker.QueueFrame(ctx, appendFrame("first"))
			h.worker.QueueFrame(ctx, appendFrame("second"))
		})}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "blocking") }()
	waitFor(3*time.Second, h.worker.ToolCallActive)

	close(gate)
	<-done

	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 2 }) {
		t.Fatalf("the pipeline saw %v, want both held frames", h.sink.appended())
	}
	if got := h.sink.appended(); got[0] != "first" || got[1] != "second" {
		t.Errorf("released as %v, want them in the order they were queued", got)
	}
}

// TestAFailingToolStillReleasesWhatItHeld is upstream's
// test_tool_error_still_decrements_and_flushes.
func TestAFailingToolStillReleasesWhatItHeld(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{{
			Name:       "failing",
			Parameters: json.RawMessage(`{"type":"object"}`),
			Handler: llm.FunctionCallHandler(func(ctx context.Context, _ llm.FunctionCallParams) error {
				(*hp).worker.QueueFrame(ctx, appendFrame("recover"))
				return errBoom
			}),
		}}
	})

	if err := h.run(t, "failing"); !errors.Is(err, errBoom) {
		t.Fatalf("the failing tool returned %v, want its own error", err)
	}
	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v, want what it held released anyway", h.sink.appended())
	}
	if h.worker.ToolCallActive() {
		t.Error("a call was still reported running after the handler failed")
	}
}

// TestAToolWithoutAHandlerIsAdvertisedOnly checks a tool nothing here answers is
// advertised without being registered, rather than registered against a handler
// that is not there.
func TestAToolWithoutAHandlerIsAdvertisedOnly(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, func(**harness) []frames.Tool {
		return []frames.Tool{{Name: "elsewhere", Parameters: json.RawMessage(`{}`)}}
	})

	if h.llm.registered("elsewhere") {
		t.Error("a tool with no handler was registered on the model")
	}
	if tools := h.worker.Tools(); len(tools) != 1 || tools[0].Name != "elsewhere" {
		t.Errorf("tools = %+v, want the tool still advertised", tools)
	}
}

// TestHoldingCanBeTurnedOff checks the worker can be told not to hold, for a
// caller that wants a tool's frames delivered as they are queued.
func TestHoldingCanBeTurnedOff(t *testing.T) {
	gate := make(chan struct{})
	off := false
	var h *harness
	h = newHarness(t, &h, &off, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "blocking", gate, func(ctx context.Context, h *harness) {
			h.worker.QueueFrame(ctx, appendFrame("straight through"))
		})}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "blocking") }()

	if !waitFor(3*time.Second, func() bool { return len(h.sink.appended()) == 1 }) {
		t.Errorf("the pipeline saw %v, want the frame delivered without holding",
			h.sink.appended())
	}
	close(gate)
	<-done
}

// TestEndWaitsForTheCallThatAskedForIt checks the rule a tool that closes the
// session depends on: the worker is still running that call when it asks, and
// ending underneath it would leave the rest of the call to nobody.
func TestEndWaitsForTheCallThatAskedForIt(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "closing", gate, func(ctx context.Context, h *harness) {
			h.worker.End(ctx, "the tool asked")
		})}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "closing") }()
	waitFor(3*time.Second, h.worker.ToolCallActive)

	time.Sleep(300 * time.Millisecond)
	if h.bus.seen(&bus.EndMessage{}) {
		t.Error("the session ended while the call that asked for it was still running")
	}

	close(gate)
	<-done
	if !waitFor(3*time.Second, func() bool { return h.bus.seen(&bus.EndMessage{}) }) {
		t.Error("the session never ended once the call finished")
	}
}

// TestActivateWorkerWaitsForTheCallThatAskedForIt checks handing over waits for
// the same moment, for the same reason.
func TestActivateWorkerWaitsForTheCallThatAskedForIt(t *testing.T) {
	gate := make(chan struct{})
	var h *harness
	h = newHarness(t, &h, nil, func(hp **harness) []frames.Tool {
		return []frames.Tool{gatedTool(hp, "handing", gate, func(ctx context.Context, h *harness) {
			h.worker.ActivateWorker(ctx, "next", workers.ActivateOptions{})
		})}
	})

	done := make(chan struct{})
	go func() { defer close(done); _ = h.run(t, "handing") }()
	waitFor(3*time.Second, h.worker.ToolCallActive)

	time.Sleep(300 * time.Millisecond)
	if h.bus.seen(&bus.ActivateWorkerMessage{}) {
		t.Error("the handover happened while the call that asked for it was still running")
	}

	close(gate)
	<-done
	if !waitFor(3*time.Second, func() bool { return h.bus.seen(&bus.ActivateWorkerMessage{}) }) {
		t.Error("the handover never happened once the call finished")
	}
}

// TestEndOutsideACallHappensAtOnce checks there is nothing to wait for when no
// tool is running.
func TestEndOutsideACallHappensAtOnce(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, nil)

	h.worker.End(h.ctx, "nothing running")

	if !waitFor(3*time.Second, func() bool { return h.bus.seen(&bus.EndMessage{}) }) {
		t.Error("ending outside a call was held up")
	}
}

// TestActivationAdvertisesTheToolsAndTheMessages checks what a worker does when
// it is activated: it tells the model what it may call, and puts the messages
// the activation carried into the conversation.
func TestActivationAdvertisesTheToolsAndTheMessages(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, func(**harness) []frames.Tool {
		return []frames.Tool{{Name: "advertised", Parameters: json.RawMessage(`{}`)}}
	})

	h.worker.OnActivated(h.ctx, llmworker.ActivationArgs{
		Messages: []map[string]any{{"role": "user", "content": "bonjour"}},
	}.ToMap())

	if !waitFor(3*time.Second, func() bool {
		return slices.Contains(h.sink.appended(), "bonjour")
	}) {
		t.Errorf("the pipeline saw %v, want the activation's message appended", h.sink.appended())
	}
	if !h.sink.sawTools() {
		t.Error("the model was never told which tools it may call")
	}
}

// TestActivationCanAppendWithoutAnswering checks a caller can put messages into
// the conversation without the model answering them.
func TestActivationCanAppendWithoutAnswering(t *testing.T) {
	var h *harness
	h = newHarness(t, &h, nil, nil)

	no := false
	h.worker.OnActivated(h.ctx, llmworker.ActivationArgs{
		Messages: []map[string]any{{"role": "user", "content": "quietly"}},
		RunLLM:   &no,
	}.ToMap())

	if !waitFor(3*time.Second, func() bool { return h.sink.lastAppendRan() != nil }) {
		t.Fatal("the activation's message never arrived")
	}
	if ran := h.sink.lastAppendRan(); ran == nil || *ran {
		t.Error("the model was asked to answer, want the append left unanswered")
	}
}

// TestActivationArgsRoundTrip checks the arguments survive the shape they travel
// in on the bus, which is how one worker activates another.
func TestActivationArgsRoundTrip(t *testing.T) {
	no := false
	args := llmworker.ActivationArgs{
		Messages: []map[string]any{{"role": "user", "content": "hello"}},
		RunLLM:   &no,
	}

	got := llmworker.ActivationArgsFrom(args.ToMap())
	if len(got.Messages) != 1 || got.Messages[0]["content"] != "hello" {
		t.Errorf("messages = %+v, want the one that was set", got.Messages)
	}
	if got.RunLLM == nil || *got.RunLLM {
		t.Errorf("run_llm = %v, want false", got.RunLLM)
	}
}
