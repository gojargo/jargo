package llm_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
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
	task := pipeline.NewTask(pipeline.New(svc, pair.Assistant()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
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
		},
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
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TTSSpeakFrame); ok {
				select {
				case spoken <- struct{}{}:
				default:
				}
			}
		},
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
	task := pipeline.NewTask(pipe, pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
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
		},
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
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch f.(type) {
			case *frames.FunctionCallResultFrame:
				results++
			case *frames.FunctionCallCancelFrame:
				cancels++
			}
		},
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
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
				select {
				case result <- fr.Result:
				default:
				}
			}
		},
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
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
				select {
				case result <- fr.Result:
				default:
				}
			}
		},
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
