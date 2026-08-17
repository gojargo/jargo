package llm_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// fakeToolGen requests one tool call on its first turn, then answers with text.
type fakeToolGen struct {
	mu   sync.Mutex
	turn int
}

func (g *fakeToolGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *fakeToolGen) GenerateWithTools(_ context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	g.mu.Lock()
	turn := g.turn
	g.turn++
	g.mu.Unlock()
	if turn == 0 {
		return sink.Tool(frames.ToolCall{
			ID:   "call_1",
			Name: "get_weather",
			Args: json.RawMessage(`{"location":"Paris"}`),
		})
	}
	return sink.Text("It is sunny.")
}

func TestBaseRunsToolLoop(t *testing.T) {
	gen := &fakeToolGen{}
	svc := llm.New("FakeToolLLM", gen)

	var gotArgs string
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		gotArgs = string(p.Arguments)
		return p.Result(ctx, "sunny, 20C", nil)
	})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}})
	convo.AddUserMessage("weather in Paris?")

	// The assistant aggregator commits the tool turn and re-triggers generation,
	// so the tool loop only completes end to end with it in the pipeline.
	pair := aggregators.New(convo)

	var mu sync.Mutex
	var got []string
	ends := 0
	done := make(chan struct{}, 1)
	// The assistant aggregator consumes the function-call frames, so watch them
	// where anything else would: between the LLM and the aggregator, which is
	// where an RTVI processor sits.
	probe := newProbe(func(f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.LLMFullResponseStartFrame:
			got = append(got, "start")
		case *frames.FunctionCallsStartedFrame:
			got = append(got, "calls-started")
		case *frames.FunctionCallInProgressFrame:
			got = append(got, "in-progress:"+fr.ToolName)
		case *frames.FunctionCallResultFrame:
			got = append(got, "result:"+fr.Result)
		case *frames.LLMTextFrame:
			got = append(got, "text:"+fr.Text)
		case *frames.LLMFullResponseEndFrame:
			ends++
			if ends == 2 {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe, pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("tool loop did not complete")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	// The response-end frames are counted rather than placed: the tool calls run
	// off the frame loop, so the first end and the call's own frames race.
	if ends != 2 {
		t.Fatalf("response-end frames = %d, want 2", ends)
	}
	if !slices.Contains(got, "calls-started") {
		t.Fatalf("frames = %v, want the calls-started announcement", got)
	}
	// calls-started is a system frame, so where it lands among the queued frames
	// of the turn is deliberately not fixed. What must hold is the order of the
	// frames that carry the turn: the call starts, reports, and only then does the
	// continuation run and answer.
	want := []string{
		"start",
		"in-progress:get_weather",
		"result:sunny, 20C",
		"start",
		"text:It is sunny.",
	}
	ordered := without(got, "calls-started")
	if len(ordered) != len(want) {
		t.Fatalf("frames = %v, want %v", ordered, want)
	}
	for i := range want {
		if ordered[i] != want[i] {
			t.Fatalf("frame %d = %q, want %q (all: %v)", i, ordered[i], want[i], ordered)
		}
	}
	if gotArgs != `{"location":"Paris"}` {
		t.Fatalf("handler args = %q, want %q", gotArgs, `{"location":"Paris"}`)
	}
	// The aggregator is the sole writer; the turn is recorded as user, assistant
	// tool-use, tool result, assistant reply — with every tool call balanced.
	msgs := convo.Messages()
	if len(msgs) != 4 {
		t.Fatalf("context messages = %+v, want 4", msgs)
	}
	if msgs[1].Role != frames.RoleAssistant || len(msgs[1].ToolCalls) != 1 {
		t.Fatalf("message 1 = %+v, want an assistant tool-use", msgs[1])
	}
	if len(msgs[2].ToolResults) != 1 {
		t.Fatalf("message 2 = %+v, want a tool-result message", msgs[2])
	}
	if !toolCallsBalanced(convo) {
		t.Fatalf("context has an unbalanced tool call: %+v", msgs)
	}
}

// TestToolHandlerDoesNotBlockFrames is the regression test for a handler run on
// the frame loop: a tool that waits on the network must not hold up the frames
// queued behind it, which is how a bot covers the wait with speech.
func TestToolHandlerDoesNotBlockFrames(t *testing.T) {
	gen := &fakeToolGen{}
	svc := llm.New("FakeToolLLM", gen)

	release := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return p.Result(ctx, "sunny, 20C", nil)
	})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	convo.AddUserMessage("weather?")

	spoken := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.TTSSpeakFrame); ok {
			select {
			case spoken <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	task.QueueFrame(frames.NewTTSSpeakFrame("Je regarde la météo."))

	select {
	case <-spoken:
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("speech queued behind a running tool handler never reached the transport")
	}
	close(release)

	task.StopWhenDone()
	<-runDone
}

// balanceCheckGen records whether the context is balanced (every tool-use message
// is followed by its tool-result message) at the moment the continuation runs.
type balanceCheckGen struct {
	mu              sync.Mutex
	turn            int
	sawContinuation bool
	balanced        bool
}

func (g *balanceCheckGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *balanceCheckGen) GenerateWithTools(_ context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	g.mu.Lock()
	turn := g.turn
	g.turn++
	g.mu.Unlock()
	if turn == 0 {
		return sink.Tool(frames.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)})
	}
	g.mu.Lock()
	g.sawContinuation = true
	// The aggregator appends the tool result immediately before re-triggering, so
	// a correct continuation always sees it as the last message. An internal loop
	// that re-read the context early would see the tool_use still dangling.
	g.balanced = endsWithToolResult(convo)
	g.mu.Unlock()
	return sink.Text("done")
}

// delayResults delays each tool-result frame before forwarding it, widening the
// window between the model requesting a tool and the aggregator committing its
// result. The continuation must still read a balanced context.
type delayResults struct {
	*processor.Base
	d time.Duration
}

func newDelayResults(d time.Duration) *delayResults {
	p := &delayResults{d: d}
	p.Base = processor.New("DelayResults", p)
	return p
}

func (p *delayResults) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.FunctionCallResultFrame); ok {
		select {
		case <-time.After(p.d):
		case <-ctx.Done():
		}
	}
	return p.PushFrame(ctx, f, dir)
}

// TestToolContinuationSeesToolResult is the regression test for the tool
// continuation race: even when result delivery lags, the continuation runs only
// after the aggregator has written the result, so the context stays balanced.
func TestToolContinuationSeesToolResult(t *testing.T) {
	gen := &balanceCheckGen{}
	svc := llm.New("FakeToolLLM", gen)
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "sunny", nil)
	})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	convo.AddUserMessage("weather?")

	pair := aggregators.New(convo)

	done := make(chan struct{}, 1)
	ends := 0
	var mu sync.Mutex
	pipe := pipeline.New(svc, newDelayResults(50*time.Millisecond), pair.Assistant())
	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			mu.Lock()
			ends++
			if ends == 2 {
				select {
				case done <- struct{}{}:
				default:
				}
			}
			mu.Unlock()
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("tool loop did not complete")
	}
	task.StopWhenDone()
	<-runDone

	gen.mu.Lock()
	defer gen.mu.Unlock()
	if !gen.sawContinuation {
		t.Fatal("continuation never ran")
	}
	if !gen.balanced {
		t.Fatal("continuation read an unbalanced context (tool_use without tool_result)")
	}
}

// without returns got with every occurrence of drop removed.
func without(got []string, drop string) []string {
	out := make([]string, 0, len(got))
	for _, g := range got {
		if g != drop {
			out = append(out, g)
		}
	}
	return out
}

// toolCallsBalanced reports whether every assistant tool-use message is
// immediately followed by a message carrying tool results.
func toolCallsBalanced(convo *frames.LLMContext) bool {
	msgs := convo.Messages()
	for i, m := range msgs {
		if len(m.ToolCalls) > 0 {
			if i+1 >= len(msgs) || len(msgs[i+1].ToolResults) == 0 {
				return false
			}
		}
	}
	return true
}

// endsWithToolResult reports whether the last context message carries tool
// results — true once the aggregator has committed the turn's results.
func endsWithToolResult(convo *frames.LLMContext) bool {
	msgs := convo.Messages()
	return len(msgs) > 0 && len(msgs[len(msgs)-1].ToolResults) > 0
}

// cancelOnInterruptionGen requests one tool call and nothing else, so a test can
// hold the call open and interrupt it.
type cancelOnInterruptionGen struct{}

func (cancelOnInterruptionGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return nil
}

func (cancelOnInterruptionGen) GenerateWithTools(_ context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	return sink.Tool(frames.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)})
}

// onceToolGen requests the tool on its first run and answers in words after
// that, the way a model does once it has the result. A generator that asks for
// the same call every time would cascade for as long as anything re-runs it.
type onceToolGen struct{ runs atomic.Int64 }

func (*onceToolGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *onceToolGen) GenerateWithTools(_ context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	if g.runs.Add(1) > 1 {
		return sink.Text("it did not finish")
	}
	return sink.Tool(frames.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)})
}

// toolConvo builds a context advertising one tool, with a user turn to answer.
func toolConvo(name string) *frames.LLMContext {
	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name: name, Description: "a tool", Parameters: json.RawMessage(`{"type":"object"}`),
	}})
	convo.AddUserMessage("go on then")
	return convo
}

// TestInterruptionCancelsTheCall checks a barge-in cancels an ordinary call: the
// handler's context is canceled, the cancellation is announced, and a handler
// that reports anyway is ignored, since the conversation already records the
// call as canceled.
func TestInterruptionCancelsTheCall(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	running := make(chan struct{})
	canceled := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		close(running)
		<-ctx.Done()
		close(canceled)
		// A handler that ignores its cancellation and reports anyway must not
		// reach the conversation.
		return p.Result(context.Background(), "too late", nil)
	})

	var mu sync.Mutex
	var results, cancels int
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f.(type) {
		case *frames.FunctionCallResultFrame:
			results++
		case *frames.FunctionCallCancelFrame:
			cancels++
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))
	select {
	case <-running:
	case <-time.After(3 * time.Second):
		t.Fatal("the handler never ran")
	}

	task.QueueFrame(frames.NewInterruptionFrame())
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("the interruption did not cancel the call's context")
	}

	// Give the ignored result every chance to arrive before checking it did not.
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if cancels == 0 {
		t.Error("the cancellation was never announced")
	}
	if results != 0 {
		t.Errorf("result frames = %d, want none: the call was canceled", results)
	}
}

// TestAsyncToolSurvivesInterruption checks a tool registered to survive a
// barge-in keeps running through one and still reports.
func TestAsyncToolSurvivesInterruption(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	running := make(chan struct{})
	release := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		close(running)
		select {
		case <-release:
		case <-ctx.Done():
			return ctx.Err()
		}
		return p.Result(ctx, "sunny", nil)
	}, llm.WithCancelOnInterruption(false))

	result := make(chan string, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))
	select {
	case <-running:
	case <-time.After(3 * time.Second):
		t.Fatal("the handler never ran")
	}

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)
	close(release)

	select {
	case got := <-result:
		if got != "sunny" {
			t.Errorf("result = %q, want the tool's own result", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("an asynchronous call must survive the barge-in and still report")
	}

	task.StopWhenDone()
	<-runDone
}

// TestMissingFunctionAnswersTheCall checks a call to a tool with no registered
// handler still completes, with a terminal result naming the function. Leaving
// it unanswered would hang the turn on a call that can never report.
func TestMissingFunctionAnswersTheCall(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	result := make(chan string, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The tool is advertised but no handler was ever registered for it.
	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))

	select {
	case got := <-result:
		if !strings.Contains(got, "get_weather") || !strings.Contains(got, "not currently available") {
			t.Errorf("result = %q, want a terminal result naming the function", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("a call to an unregistered tool was never answered")
	}

	task.StopWhenDone()
	<-runDone
}

// probe reports every frame passing through it and forwards it untouched. It
// stands where an RTVI processor would, between the LLM and the assistant
// aggregator, which is where the function-call frames can be seen: the
// aggregator consumes them, so they never reach the end of the pipeline.
type probe struct {
	*processor.Base
	seen func(frames.Frame)
}

func newProbe(seen func(frames.Frame)) *probe {
	p := &probe{seen: seen}
	p.Base = processor.New("Probe", p)
	return p
}

func (p *probe) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir == processor.Downstream {
		p.seen(f)
	}
	return p.PushFrame(ctx, f, dir)
}

// TestFunctionCallTimeout checks a call that overruns its bound is canceled
// rather than abandoned: the handler is canceled so it can run its cleanup, the
// conversation records the call as canceled, and the cancellation asks for
// inference so the model can say the call did not complete.
func TestFunctionCallTimeout(t *testing.T) {
	gen := &onceToolGen{}
	svc := llm.New("FakeToolLLM", gen, llm.WithFunctionCallTimeout(50*time.Millisecond))

	release := make(chan struct{})
	rolledBack := make(chan struct{}, 1)
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		select {
		case <-release:
			return p.Result(ctx, "far too late", nil)
		case <-ctx.Done():
			select {
			case rolledBack <- struct{}{}:
			default:
			}
			return ctx.Err()
		}
	})

	canceled := make(chan *frames.FunctionCallCancelFrame, 4)
	results := make(chan *frames.FunctionCallResultFrame, 4)
	probe := newProbe(func(f frames.Frame) {
		switch fr := f.(type) {
		case *frames.FunctionCallCancelFrame:
			select {
			case canceled <- fr:
			default:
			}
		case *frames.FunctionCallResultFrame:
			select {
			case results <- fr:
			default:
			}
		}
	})
	convo := toolConvo("get_weather")
	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(svc, probe, pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case fr := <-canceled:
		// Nothing else follows up on a deadline, so it has to run the model.
		if !fr.RunLLM {
			t.Error("the deadline did not ask for inference, so the call is never answered")
		}
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("the overrunning call was never given up on")
	}

	select {
	case <-rolledBack:
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("the handler was never canceled, so it could not roll anything back")
	}

	// The conversation records it as canceled, not as still running: the user
	// turn, the call, then the placeholder settled to the canceled marker.
	if !waitForContext(convo, func(msgs []frames.Message) bool {
		return len(msgs) >= 3 && len(msgs[2].ToolResults) == 1 &&
			msgs[2].ToolResults[0].Content == "CANCELLED" //nolint:misspell // the literal written to the conversation
	}) {
		t.Errorf("messages = %+v, want the call recorded as canceled", convo.Messages())
	}

	// The cancellation asked for inference, so the model gets a turn to say the
	// call did not complete.
	if !waitFor(3*time.Second, func() bool { return gen.runs.Load() >= 2 }) {
		t.Errorf("the model ran %d times, want a second run answering the cancellation", gen.runs.Load())
	}

	// A handler that outlives its deadline cannot settle the call late.
	select {
	case fr := <-results:
		t.Errorf("result %q reached the pipeline after the deadline settled the call", fr.Result)
	default:
	}

	close(release)
	task.StopWhenDone()
	<-runDone
}

// TestFunctionCallTimeoutPerFunction checks a per-function bound overrides the
// service's, so one slow tool need not loosen the bound for every other.
func TestFunctionCallTimeoutPerFunction(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{},
		llm.WithFunctionCallTimeout(20*time.Millisecond))

	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		time.Sleep(80 * time.Millisecond)
		return p.Result(ctx, "sunny", nil)
	}, llm.WithTimeout(3*time.Second))

	result := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))

	select {
	case got := <-result:
		if got != "sunny" {
			t.Errorf("result = %q, want the tool's own: its bound is the looser one", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the call never reported")
	}

	task.StopWhenDone()
	<-runDone
}

// TestCatchAllFunction checks the handler registered under the empty name takes
// a call no named handler claims, and runs under the name the model used.
func TestCatchAllFunction(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	name := make(chan string, 1)
	svc.RegisterFunction("", func(ctx context.Context, p llm.FunctionCallParams) error {
		select {
		case name <- p.FunctionName:
		default:
		}
		return p.Result(ctx, "handled", nil)
	})
	if !svc.HasFunction("anything_at_all") {
		t.Error("a catch-all should claim any call")
	}

	result := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))

	select {
	case got := <-result:
		if got != "handled" {
			t.Errorf("result = %q, want the catch-all's", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the catch-all never ran")
	}
	if got := <-name; got != "get_weather" {
		t.Errorf("FunctionName = %q, want the name the model used", got)
	}

	task.StopWhenDone()
	<-runDone
}

// TestUnregisterFunction checks a withdrawn handler stops claiming calls, which
// then fall through to the missing-function answer.
func TestUnregisterFunction(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "sunny", nil)
	})
	if !svc.HasFunction("get_weather") {
		t.Fatal("the handler was just registered")
	}
	if !svc.UnregisterFunction("get_weather") {
		t.Error("UnregisterFunction should report that it removed one")
	}
	if svc.HasFunction("get_weather") {
		t.Error("the handler was withdrawn")
	}
	if svc.UnregisterFunction("get_weather") {
		t.Error("UnregisterFunction should report false the second time")
	}
}

// TestFunctionCallEvents checks the application is told which calls a response
// started and which an interruption canceled.
func TestFunctionCallEvents(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	started := make(chan []frames.ToolCall, 1)
	canceled := make(chan []frames.ToolCall, 1)
	svc.OnFunctionCallsStarted(func(_ context.Context, calls []frames.ToolCall) {
		select {
		case started <- calls:
		default:
		}
	})
	svc.OnFunctionCallsCanceled(func(_ context.Context, calls []frames.ToolCall) {
		select {
		case canceled <- calls:
		default:
		}
	})

	running := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		close(running)
		<-ctx.Done()
		return ctx.Err()
	})

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))

	select {
	case calls := <-started:
		if len(calls) != 1 || calls[0].Name != "get_weather" {
			t.Errorf("started = %+v, want the one call", calls)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the started callback never ran")
	}

	<-running
	task.QueueFrame(frames.NewInterruptionFrame())

	select {
	case calls := <-canceled:
		if len(calls) != 1 || calls[0].Name != "get_weather" {
			t.Errorf("canceled = %+v, want the one call", calls)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the canceled callback never ran")
	}

	task.StopWhenDone()
	<-runDone
}

// waitForContext polls the conversation until cond holds or a deadline passes.
func waitForContext(convo *frames.LLMContext, cond func([]frames.Message) bool) bool {
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond(convo.Messages()) {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond(convo.Messages())
}

// twoCallGen requests two tool calls in one response, which is what the runner
// options are about.
type twoCallGen struct{}

func (twoCallGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (twoCallGen) GenerateWithTools(_ context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	if err := sink.Tool(frames.ToolCall{ID: "c1", Name: "tool", Args: json.RawMessage(`{}`)}); err != nil {
		return err
	}
	return sink.Tool(frames.ToolCall{ID: "c2", Name: "tool", Args: json.RawMessage(`{}`)})
}

// TestSequentialFunctionCalls checks that with sequential running no two
// handlers overlap, so tools sharing something that does not take concurrent use
// stay safe.
func TestSequentialFunctionCalls(t *testing.T) {
	svc := llm.New("FakeToolLLM", twoCallGen{}, llm.WithSequentialFunctionCalls())

	var mu sync.Mutex
	running, maxRunning := 0, 0
	done := make(chan struct{}, 2)
	svc.RegisterFunction("tool", func(ctx context.Context, p llm.FunctionCallParams) error {
		mu.Lock()
		running++
		if running > maxRunning {
			maxRunning = running
		}
		mu.Unlock()

		time.Sleep(30 * time.Millisecond)

		mu.Lock()
		running--
		mu.Unlock()
		if err := p.Result(ctx, "ok", nil); err != nil {
			return err
		}
		done <- struct{}{}
		return nil
	})

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("tool")))
	for range 2 {
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("both calls should run, one after the other")
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if maxRunning != 1 {
		t.Errorf("handlers running at once = %d, want 1", maxRunning)
	}

	task.StopWhenDone()
	<-runDone
}

// TestUngroupedFunctionCalls checks that without grouping each call carries no
// group id, so each result re-runs generation on its own.
func TestUngroupedFunctionCalls(t *testing.T) {
	svc := llm.New("FakeToolLLM", twoCallGen{}, llm.WithUngroupedFunctionCalls())
	svc.RegisterFunction("tool", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "ok", nil)
	})

	groups := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallInProgressFrame); ok {
			select {
			case groups <- fr.GroupID:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("tool")))
	for range 2 {
		select {
		case got := <-groups:
			if got != "" {
				t.Errorf("GroupID = %q, want none: the calls are not grouped", got)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("both calls should start")
		}
	}

	task.StopWhenDone()
	<-runDone
}

// TestGroupedFunctionCallsShareAGroup checks the default: the calls of one
// response carry the same group id, which is what lets the aggregator answer the
// batch with a single inference.
func TestGroupedFunctionCallsShareAGroup(t *testing.T) {
	svc := llm.New("FakeToolLLM", twoCallGen{})
	svc.RegisterFunction("tool", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "ok", nil)
	})

	groups := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallInProgressFrame); ok {
			select {
			case groups <- fr.GroupID:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("tool")))
	var seen []string
	for range 2 {
		select {
		case got := <-groups:
			seen = append(seen, got)
		case <-time.After(3 * time.Second):
			t.Fatal("both calls should start")
		}
	}
	if seen[0] == "" || seen[0] != seen[1] {
		t.Errorf("group ids = %q, want one shared id", seen)
	}

	task.StopWhenDone()
	<-runDone
}

// asyncToolGen requests the named tool once, then answers with text. It stands
// in for a service that converts through an adapter, which is where the tools
// the service implements itself are added.
type asyncToolGen struct {
	// adapter is what the base adds its built-in tools to, and what this
	// generator reads back to see what the model was actually offered.
	adapter adapter.Base
	mu      sync.Mutex
	turn    int
	name    string
	// tools records what the model was offered on the first inference.
	tools []frames.Tool
	// system records the instruction the service composed, read the way a real
	// provider reads it when it builds a request.
	system string
	// svc is the service this generator belongs to, set once it is built.
	svc *llm.Base
}

// LLMAdapter implements llm.AdapterHolder.
func (g *asyncToolGen) LLMAdapter() llm.BuiltinToolHolder { return &g.adapter }

func (g *asyncToolGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *asyncToolGen) GenerateWithTools(
	_ context.Context, convo *frames.LLMContext, sink llm.Sink,
) error {
	g.mu.Lock()
	turn := g.turn
	g.turn++
	if turn == 0 {
		// What an adapter would send, which is where a built-in tool now lives,
		// and the instruction a provider passes beside the conversation.
		g.tools = g.adapter.WithBuiltins(convo.ToolsSchema()).Standard
		g.system = g.svc.SystemInstruction()
	}
	g.mu.Unlock()
	if turn == 0 {
		return sink.Tool(frames.ToolCall{ID: "c1", Name: g.name, Args: json.RawMessage(`{}`)})
	}
	return sink.Text("done")
}

// TestAsyncToolCancellationOffersTheBuiltInTool checks the built-in tool and its
// instructions appear only while an asynchronous tool is registered, and only
// when the service was built to offer them.
func TestAsyncToolCancellationOffersTheBuiltInTool(t *testing.T) {
	hasCancel := func(tools []frames.Tool) bool {
		for _, tool := range tools {
			if tool.Name == llm.CancelToolName("watch") {
				return true
			}
		}
		return false
	}

	t.Run("offered alongside an async tool", func(t *testing.T) {
		gen := &asyncToolGen{name: "watch"}
		svc := llm.New("FakeToolLLM", gen, llm.WithAsyncToolCancellation())
		gen.svc = svc
		svc.RegisterFunction("watch", func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "watching", nil)
		}, llm.WithCancelOnInterruption(false))

		runToolTurn(t, svc, "watch")

		gen.mu.Lock()
		defer gen.mu.Unlock()
		if !hasCancel(gen.tools) {
			t.Errorf("tools = %+v, want the built-in cancel tool among them", gen.tools)
		}
		if !strings.Contains(gen.system, "ASYNC TOOL CANCELLATION") {
			t.Errorf("system = %q, want the cancellation instructions", gen.system)
		}
	})

	t.Run("not offered without an async tool", func(t *testing.T) {
		gen := &asyncToolGen{name: "lookup"}
		svc := llm.New("FakeToolLLM", gen, llm.WithAsyncToolCancellation())
		gen.svc = svc
		// Registered the ordinary way, so it is canceled on interruption.
		svc.RegisterFunction("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "found", nil)
		})

		runToolTurn(t, svc, "lookup")

		gen.mu.Lock()
		defer gen.mu.Unlock()
		if hasCancel(gen.tools) {
			t.Errorf("tools = %+v, want no cancel tool: nothing runs in the background", gen.tools)
		}
		if strings.Contains(gen.system, "ASYNC TOOL CANCELLATION") {
			t.Error("the instructions should not be added when the tool is not offered")
		}
	})

	t.Run("not offered unless asked for", func(t *testing.T) {
		gen := &asyncToolGen{name: "watch"}
		svc := llm.New("FakeToolLLM", gen)
		gen.svc = svc
		svc.RegisterFunction("watch", func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "watching", nil)
		}, llm.WithCancelOnInterruption(false))

		runToolTurn(t, svc, "watch")

		gen.mu.Lock()
		defer gen.mu.Unlock()
		if hasCancel(gen.tools) {
			t.Errorf("tools = %+v, want none: cancellation was not enabled", gen.tools)
		}
	})
}

// TestCancelAsyncToolCallAbandonsTheCall checks the model can abandon a call
// that outlives an interruption: its context is canceled, the cancellation is
// announced, and anything it reports afterwards is ignored.
func TestCancelAsyncToolCallAbandonsTheCall(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{}, llm.WithAsyncToolCancellation())

	running := make(chan struct{})
	canceled := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		close(running)
		<-ctx.Done()
		close(canceled)
		return p.Result(context.Background(), "far too late", nil)
	}, llm.WithCancelOnInterruption(false))

	var mu sync.Mutex
	var results []string
	cancels := 0
	probe := newProbe(func(f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.FunctionCallResultFrame:
			results = append(results, fr.Result)
		case *frames.FunctionCallCancelFrame:
			cancels++
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))
	select {
	case <-running:
	case <-time.After(3 * time.Second):
		t.Fatal("the async handler never ran")
	}

	// The model asks for it back.
	if err := svc.CancelAsyncToolCall(context.Background(), "call_1"); err != nil {
		t.Fatalf("CancelAsyncToolCall: %v", err)
	}
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("the request did not cancel the call's context")
	}

	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if cancels == 0 {
		t.Error("the cancellation was never announced")
	}
	for _, r := range results {
		if r == "far too late" {
			t.Error("a result reported after cancellation must not reach the conversation")
		}
	}
}

// runToolTurn drives one tool-calling turn through svc and waits for it to
// settle, so a test can inspect what the model was shown.
func runToolTurn(t *testing.T, svc *llm.Base, tool string) {
	t.Helper()
	convo := toolConvo(tool)
	pair := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(svc, pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	if !waitForContext(convo, func(msgs []frames.Message) bool { return len(msgs) >= 3 }) {
		t.Fatal("the tool turn never settled")
	}
	task.StopWhenDone()
	<-runDone
}

// TestToolCarriedHandlerIsRegistered checks a tool that carries its own handler
// needs no separate registration: advertising it is enough.
func TestToolCarriedHandlerIsRegistered(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}})
	convo.AddUserMessage("weather?")

	result := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case got := <-result:
		if got != "sunny" {
			t.Errorf("result = %q, want the tool's own handler to have run", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the handler the tool carries never ran")
	}

	task.StopWhenDone()
	<-runDone
}

// TestToolCarriedHandlerIsDroppedWhenWithdrawn checks a handler that came from a
// toolset goes when the toolset stops advertising it, so what the model can call
// and what answers stay the same set.
func TestToolCarriedHandlerIsDroppedWhenWithdrawn(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})

	weather := frames.Tool{
		Name: "get_weather", Description: "weather", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}
	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{weather})
	convo.AddUserMessage("weather?")

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	if !waitFor(3*time.Second, func() bool { return svc.HasFunction("get_weather") }) {
		t.Fatal("advertising the tool should have registered the handler it carries")
	}

	// The toolset stops advertising it.
	convo.SetTools(nil)
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	if !waitFor(3*time.Second, func() bool { return !svc.HasFunction("get_weather") }) {
		t.Error("withdrawing the tool should have taken its handler with it")
	}

	task.StopWhenDone()
	<-runDone
}

// TestRegisteredHandlerWinsOverToolCarried checks a handler registered by hand
// is neither replaced by a tool's own nor dropped when the tool is withdrawn.
// Registering is the deliberate act, and it is the application's to undo.
func TestRegisteredHandlerWinsOverToolCarried(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, "registered by hand", nil)
	})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name: "get_weather", Description: "weather", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "carried by the tool", nil)
		},
	}})
	convo.AddUserMessage("weather?")

	result := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			select {
			case result <- fr.Result:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case got := <-result:
		if got != "registered by hand" {
			t.Errorf("result = %q, want the hand-registered handler to win", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no handler ran")
	}

	// Withdrawing the tool must not take the hand-registered handler with it.
	convo.SetTools(nil)
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	time.Sleep(100 * time.Millisecond)
	if !svc.HasFunction("get_weather") {
		t.Error("a hand-registered handler is the application's to remove")
	}

	task.StopWhenDone()
	<-runDone
}

// waitFor polls cond until it holds or the timeout elapses.
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
