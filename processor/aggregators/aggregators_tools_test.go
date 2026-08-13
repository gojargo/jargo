package aggregators_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// group is the batch identifier the LLM service stamps on every call it starts
// from one response.
const group = "g1"

// inProgress builds the frame that starts an ordinary tool call: one the model
// waits for, and that a barge-in cancels.
func inProgress(id, name string, args json.RawMessage) *frames.FunctionCallInProgressFrame {
	return frames.NewFunctionCallInProgressFrame(id, name, args, true, group)
}

// drainAssistant runs an assistant aggregator over the given frames, then stops.
func drainAssistant(t *testing.T, convo *frames.LLMContext, fs ...frames.Frame) {
	t.Helper()
	runAssistant(t, convo, func(task *pipeline.Worker) {
		for _, f := range fs {
			task.QueueFrame(f)
		}
	})
}

func TestAssistantAggregatorWritesToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame(calls),
		inProgress("c1", "get_weather", json.RawMessage(`{"location":"Paris"}`)),
		frames.NewFunctionCallResultFrame("c1", "get_weather", nil, "sunny"),
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

// TestAssistantAggregatorAnswersCallBeforeItReports checks the window that used
// to hold an invalid conversation: between the model requesting a tool and the
// tool reporting, the call must already have a message answering it, so an
// inference in that window sends the model a valid conversation.
func TestAssistantAggregatorAnswersCallBeforeItReports(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather"}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame(calls),
		inProgress("c1", "get_weather", nil),
	)

	msgs := convo.Messages()
	if len(msgs) != 2 {
		t.Fatalf("messages = %+v, want the call and a placeholder answering it", msgs)
	}
	if len(msgs[0].ToolCalls) != 1 || msgs[0].ToolCalls[0].ID != "c1" {
		t.Fatalf("msg[0] = %+v, want assistant tool_use c1", msgs[0])
	}
	if len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].ID != "c1" {
		t.Fatalf("msg[1] = %+v, want a placeholder result for c1", msgs[1])
	}
	if got := msgs[1].ToolResults[0].Content; got != "IN_PROGRESS" {
		t.Fatalf("placeholder content = %q, want IN_PROGRESS", got)
	}
}

// TestAssistantAggregatorResultStaysWithItsCall is the regression test for the
// interleaving that broke a session for good.
//
// The user aggregator sits upstream of the LLM and writes to the same context, so
// a new user turn can land while a tool is still running. A result appended at
// the tail would then sit after that user message, separated from the call it
// answers, and every later inference would carry the mis-ordered pair. Updating
// the placeholder in place is what keeps them together.
func TestAssistantAggregatorResultStaysWithItsCall(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather"}}

	runAssistant(t, convo, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewFunctionCallsStartedFrame(calls))
		task.QueueFrame(inProgress("c1", "get_weather", nil))

		// Wait for the call to be recorded, then let a new user turn land the way
		// the user aggregator would while the tool is still running.
		if !waitFor(3*time.Second, func() bool { return len(convo.Messages()) == 2 }) {
			t.Fatal("the tool call was never written to the context")
		}
		convo.AddUserMessage("actually, what time is it?")

		task.QueueFrame(frames.NewFunctionCallResultFrame("c1", "get_weather", nil, "sunny"))
	})

	msgs := convo.Messages()
	if len(msgs) != 3 {
		t.Fatalf("messages = %+v, want 3", msgs)
	}
	if len(msgs[0].ToolCalls) != 1 || msgs[0].ToolCalls[0].ID != "c1" {
		t.Fatalf("msg[0] = %+v, want the assistant tool_use", msgs[0])
	}
	if len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].Content != "sunny" {
		t.Fatalf("msg[1] = %+v, want the result next to the call that made it", msgs[1])
	}
	if msgs[2].Role != frames.RoleUser || msgs[2].Text != "actually, what time is it?" {
		t.Fatalf("msg[2] = %+v, want the user turn after the settled pair", msgs[2])
	}
}

// TestAssistantAggregatorParallelToolResults checks a batch of calls: each is
// written with its own answer, and each result settles into its own placeholder
// rather than into whichever came first.
func TestAssistantAggregatorParallelToolResults(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "a"}, {ID: "c2", Name: "b"}}
	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame(calls),
		inProgress("c1", "a", nil),
		inProgress("c2", "b", nil),
		// The second call reports first, which is what a per-call placeholder has
		// to survive: an appended result would answer the wrong call.
		frames.NewFunctionCallResultFrame("c2", "b", nil, "rb"),
		frames.NewFunctionCallResultFrame("c1", "a", nil, "ra"),
		frames.NewLLMFullResponseEndFrame(),
	)

	msgs := convo.Messages()
	if len(msgs) != 4 {
		t.Fatalf("messages = %+v, want each call followed by its own result", msgs)
	}
	for i, want := range []struct{ id, result string }{{"c1", "ra"}, {"c2", "rb"}} {
		call, res := msgs[i*2], msgs[i*2+1]
		if len(call.ToolCalls) != 1 || call.ToolCalls[0].ID != want.id {
			t.Fatalf("msg[%d] = %+v, want the tool_use for %s", i*2, call, want.id)
		}
		if len(res.ToolResults) != 1 || res.ToolResults[0].ID != want.id {
			t.Fatalf("msg[%d] = %+v, want the result for %s", i*2+1, res, want.id)
		}
		if got := res.ToolResults[0].Content; got != want.result {
			t.Fatalf("result for %s = %q, want %q", want.id, got, want.result)
		}
	}
}

// TestAssistantAggregatorInterruptionLeavesThePairBalanced checks that a
// barge-in needs no tool balancing of its own: the call already has a message
// answering it, and the cancellation marks that message where it sits.
func TestAssistantAggregatorInterruptionLeavesThePairBalanced(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "get_weather"}}

	runAssistant(t, convo, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewFunctionCallsStartedFrame(calls))
		task.QueueFrame(inProgress("c1", "get_weather", nil))

		// Barge in once the call is running, which is when a barge-in can happen
		// at all. The cancel frame is a system frame and would otherwise overtake
		// the call it is canceling.
		if !waitFor(3*time.Second, func() bool { return len(convo.Messages()) == 2 }) {
			t.Fatal("the tool call was never written to the context")
		}
		task.QueueFrame(frames.NewInterruptionFrame())
		task.QueueFrame(frames.NewFunctionCallCancelFrame("c1", "get_weather"))
	})

	msgs := convo.Messages()
	if len(msgs) != 2 {
		t.Fatalf("messages = %+v, want the pair and nothing appended by the barge-in", msgs)
	}
	if len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].ID != "c1" {
		t.Fatalf("msg[1] = %+v, want the placeholder for c1", msgs[1])
	}
	//nolint:misspell // the literal written to the conversation
	if got := msgs[1].ToolResults[0].Content; got != "CANCELLED" {
		t.Fatalf("placeholder content = %q, want the canceled marker", got)
	}
}

// TestAssistantAggregatorAsyncToolReportsThroughDeveloperMessages checks a tool
// registered to survive an interruption: the model does not wait for it, so it
// gets a started message rather than a placeholder, and each result it reports is
// appended where the conversation has got to rather than replacing anything.
func TestAssistantAggregatorAsyncToolReportsThroughDeveloperMessages(t *testing.T) {
	convo := frames.NewLLMContext("system")
	calls := []frames.ToolCall{{ID: "c1", Name: "watch_flight"}}
	notFinal := false
	update := frames.NewFunctionCallResultFrame("c1", "watch_flight", nil, "boarding")
	update.Properties = &frames.FunctionCallResultProperties{IsFinal: &notFinal}

	drainAssistant(t, convo,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewFunctionCallsStartedFrame(calls),
		// cancelOnInterruption false: the call outlives the turn that made it.
		frames.NewFunctionCallInProgressFrame("c1", "watch_flight", nil, false, group),
		update,
		frames.NewFunctionCallResultFrame("c1", "watch_flight", nil, "departed"),
	)

	msgs := convo.Messages()
	if len(msgs) != 4 {
		t.Fatalf("messages = %+v, want the call, its started message, and both results", msgs)
	}
	started, ok := frames.ParseAsyncToolMessage(msgs[1])
	if !ok || started.Kind != frames.AsyncToolStarted || started.ToolCallID != "c1" {
		t.Fatalf("msg[1] = %+v, want the async-tool started message", msgs[1])
	}
	for i, want := range []struct {
		kind   frames.AsyncToolKind
		result string
	}{{frames.AsyncToolIntermediate, "boarding"}, {frames.AsyncToolFinal, "departed"}} {
		m := msgs[2+i]
		if m.Role != frames.RoleDeveloper {
			t.Fatalf("msg[%d] = %+v, want a developer message", 2+i, m)
		}
		got, ok := frames.ParseAsyncToolMessage(m)
		if !ok || got.Kind != want.kind || got.Result != want.result {
			t.Fatalf("msg[%d] = %+v, want a %s result %q", 2+i, got, want.kind, want.result)
		}
	}
}
