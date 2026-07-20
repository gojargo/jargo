package llm_test

import (
	"context"
	"encoding/json"
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
	svc.RegisterFunction("get_weather", func(_ context.Context, args json.RawMessage) (string, error) {
		gotArgs = string(args)
		return "sunny, 20C", nil
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
				got = append(got, "end")
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
	want := []string{
		"start",
		"calls-started",
		"in-progress:get_weather",
		"result:sunny, 20C",
		"end",
		"start",
		"text:It is sunny.",
		"end",
	}
	if len(got) != len(want) {
		t.Fatalf("frames = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame %d = %q, want %q (all: %v)", i, got[i], want[i], got)
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
	svc.RegisterFunction("get_weather", func(context.Context, json.RawMessage) (string, error) {
		return "sunny", nil
	})

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{Name: "get_weather", Description: "weather", Parameters: json.RawMessage(`{"type":"object"}`)}})
	convo.AddUserMessage("weather?")

	pair := aggregators.New(convo)

	done := make(chan struct{}, 1)
	ends := 0
	var mu sync.Mutex
	task := pipeline.NewTask(pipeline.New(svc, newDelayResults(50*time.Millisecond), pair.Assistant()), pipeline.TaskParams{
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
