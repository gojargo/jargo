package aggregators_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// drainAssistant runs an assistant aggregator over the given frames and returns
// the resulting context messages.
func drainAssistant(t *testing.T, convo *frames.LLMContext, fs ...frames.Frame) {
	t.Helper()
	pair := aggregators.New(convo)
	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, f := range fs {
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

func TestAssistantAggregatorWritesToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame("", calls),
		frames.NewFunctionCallResultFrame("c1", "get_weather", "sunny", false),
		frames.NewLLMTextFrame("It is sunny."),
		frames.NewLLMFullResponseEndFrame(),
	)

	msgs := convo.Messages()
	if len(msgs) != 3 {
		t.Fatalf("messages = %+v, want 3", msgs)
	}
	if msgs[0].Role != frames.RoleAssistant || len(msgs[0].ToolCalls) != 1 || msgs[0].ToolCalls[0].ID != "c1" {
		t.Fatalf("msg[0] = %+v, want assistant tool_use c1", msgs[0])
	}
	if msgs[1].Role != frames.RoleUser || len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].Content != "sunny" {
		t.Fatalf("msg[1] = %+v, want user tool_result sunny", msgs[1])
	}
	if msgs[2].Role != frames.RoleAssistant || msgs[2].Text != "It is sunny." {
		t.Fatalf("msg[2] = %+v, want assistant final text", msgs[2])
	}
}

func TestAssistantAggregatorParallelToolResults(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "a"}, {ID: "c2", Name: "b"}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame("", calls),
		frames.NewFunctionCallResultFrame("c1", "a", "ra", false),
		frames.NewFunctionCallResultFrame("c2", "b", "rb", false),
		frames.NewLLMFullResponseEndFrame(),
	)

	msgs := convo.Messages()
	// assistant(tool_use x2), then one user(tool_result x2). No final text.
	if len(msgs) != 2 {
		t.Fatalf("messages = %+v, want 2", msgs)
	}
	if len(msgs[0].ToolCalls) != 2 {
		t.Fatalf("tool calls = %+v, want 2", msgs[0].ToolCalls)
	}
	if len(msgs[1].ToolResults) != 2 {
		t.Fatalf("tool results = %+v, want 2 in one message", msgs[1].ToolResults)
	}
}

func TestAssistantAggregatorInterruptionBalancesToolUse(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather"}}
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewFunctionCallsStartedFrame("", calls))
	// The interruption arrives before any result, while the tool is "running".
	time.Sleep(300 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	// assistant(tool_use c1) then a synthetic user(tool_result c1, error) so the
	// tool_use is not left dangling for the next turn.
	if len(msgs) != 2 {
		t.Fatalf("messages = %+v, want 2 (tool_use + synthetic result)", msgs)
	}
	if len(msgs[1].ToolResults) != 1 || !msgs[1].ToolResults[0].IsError || msgs[1].ToolResults[0].ID != "c1" {
		t.Fatalf("msg[1] = %+v, want synthetic error result for c1", msgs[1])
	}
}
