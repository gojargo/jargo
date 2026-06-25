package llm_test

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
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

	var mu sync.Mutex
	var got []string
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
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
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}})
	convo.AddUserMessage("weather in Paris?")
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
	// The LLM base must not write to the context; only the user message is there.
	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser {
		t.Fatalf("context messages = %+v, want only the user message", msgs)
	}
}
