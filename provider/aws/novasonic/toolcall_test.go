package novasonic

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Nova Sonic asks for a tool with a toolUse event and takes the result back as a
// content block of its own. These tests cover both halves and the declaration
// that makes the call possible at all.

// eventBody digs the named event's body out of one wire message, failing rather
// than panicking when it is not shaped the way the test expects.
func eventBody(t *testing.T, msg map[string]any, name string) map[string]any {
	t.Helper()
	envelope, ok := msg["event"].(map[string]any)
	if !ok {
		t.Fatalf("message = %v, want an event envelope", msg)
	}
	body, ok := envelope[name].(map[string]any)
	if !ok {
		t.Fatalf("event = %v, want a %s", envelope, name)
	}
	return body
}

// The tools are declared in the prompt start, which is the only place Nova Sonic
// takes them. Each carries its schema as a string rather than as an object.
func TestPromptStartDeclaresTheTools(t *testing.T) {
	s, _ := newTestService(t)
	s.promptName = "p1"
	s.tools = []frames.Tool{{
		Name:        "get_weather",
		Description: "look up the weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
	}}

	body := eventBody(t, s.promptStart(), "promptStart")

	config, ok := body["toolConfiguration"].(map[string]any)
	if !ok {
		t.Fatalf("promptStart = %v, want a toolConfiguration", body)
	}
	if _, ok := body["toolUseOutputConfiguration"]; !ok {
		t.Error("the prompt start did not say how tool output should come back")
	}
	tools, _ := config["tools"].([]map[string]any)
	if len(tools) != 1 {
		t.Fatalf("declared %d tools, want 1", len(tools))
	}
	spec, _ := tools[0]["toolSpec"].(map[string]any)
	if spec["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", spec["name"])
	}
	schema, _ := spec["inputSchema"].(map[string]any)
	if _, isString := schema["json"].(string); !isString {
		t.Errorf("inputSchema = %v, want the schema as a string", schema)
	}
}

// A session with no tools declares none, rather than an empty configuration the
// model would have to make sense of.
func TestPromptStartWithoutToolsDeclaresNone(t *testing.T) {
	s, _ := newTestService(t)
	s.promptName = "p1"

	body := eventBody(t, s.promptStart(), "promptStart")

	if _, ok := body["toolConfiguration"]; ok {
		t.Error("a session with no tools declared a tool configuration")
	}
}

// The point of the whole exchange: the model asks for a tool and the handler the
// tool carries runs, with the arguments the model wrote.
func TestToolUseRunsTheHandler(t *testing.T) {
	s, _ := newTestService(t)

	ran := make(chan string, 1)
	convo := frames.NewLLMContext("system")
	convo.SetTools([]frames.Tool{{
		Name:       "get_weather",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			select {
			case ran <- string(p.Arguments):
			default:
			}
			return p.Result(ctx, "sunny", nil)
		},
	}})
	s.convo = convo
	s.SyncToolHandlers(context.Background(), convo.Tools())

	s.handle(outputEvent{ToolUse: &toolUse{
		ToolName:  "get_weather",
		ToolUseID: "call_1",
		Content:   `{"location":"Paris"}`,
	}})

	select {
	case args := <-ran:
		if !strings.Contains(args, "Paris") {
			t.Errorf("handler got arguments %q, want the ones the model wrote", args)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the tool the model asked for never ran")
	}
}

// A call arriving before any conversation has nowhere to put its result, so it
// is not run rather than run and lost.
func TestAToolUseWithoutAConversationIsNotRun(t *testing.T) {
	s, _ := newTestService(t)

	// No conversation has reached the service.
	s.handle(outputEvent{ToolUse: &toolUse{ToolName: "get_weather", ToolUseID: "call_1"}})

	if s.resultSeen("call_1") {
		t.Error("a call with nowhere to answer was treated as answered")
	}
}

// A result is sent as its own content block: opened naming the call it answers,
// then the result, then closed.
func TestAToolResultIsSentAsItsOwnBlock(t *testing.T) {
	events := toolResultEvents("p1", "c1", "call_1", "sunny")

	if len(events) != 3 {
		t.Fatalf("sent %d events, want the block opened, filled and closed", len(events))
	}

	start := eventBody(t, events[0], "contentStart")
	if start["type"] != "TOOL" || start["role"] != "TOOL" {
		t.Errorf("contentStart = %v, want a TOOL block", start)
	}
	config, ok := start["toolResultInputConfiguration"].(map[string]any)
	if !ok || config["toolUseId"] != "call_1" {
		t.Errorf("contentStart = %v, want it to name the call it answers", start)
	}

	result := eventBody(t, events[1], "toolResult")
	if result[keyContent] != "sunny" {
		t.Errorf("toolResult content = %v, want what the handler returned", result[keyContent])
	}
	if result[keyContentName] != "c1" || result[keyPromptName] != "p1" {
		t.Errorf("toolResult = %v, want it in the block that was opened", result)
	}

	end := eventBody(t, events[2], "contentEnd")
	if end[keyContentName] != "c1" {
		t.Errorf("contentEnd = %v, want it to close the block that was opened", end)
	}
}

// A call still running has a placeholder standing in for its result. It is not
// an answer, so it is neither sent nor recorded: the real one has still to come.
func TestAnInProgressResultIsNotSettled(t *testing.T) {
	s, _ := newTestService(t)
	convo := frames.NewLLMContext("system")
	convo.AddToolResult(frames.ToolResult{
		ID: "call_1", Name: "get_weather", Content: frames.ToolResultInProgress,
	})

	s.processCompletedCalls(context.Background(), convo, true)

	if s.resultSeen("call_1") {
		t.Error("the placeholder was recorded as the call's result")
	}
}

// A result already reported is not reported again: the conversation is handed to
// the service whenever it changes, and every change carries every result it
// holds.
func TestAResultIsRecordedOnce(t *testing.T) {
	s, _ := newTestService(t)
	convo := frames.NewLLMContext("system")
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "get_weather", Content: "sunny"})

	ctx := context.Background()
	s.processCompletedCalls(ctx, convo, true)
	if !s.resultSeen("call_1") {
		t.Fatal("the result was not recorded as settled")
	}
	// A second pass over the same conversation finds nothing new to send.
	s.processCompletedCalls(ctx, convo, true)
}

// The tools a conversation brings reach the session, which for this API means
// opening it again: the prompt start is the only place they can be declared.
func TestToolsFromTheConversationAreHeldForThePromptStart(t *testing.T) {
	s, _ := newTestService(t)
	convo := frames.NewLLMContext("system")
	convo.SetTools([]frames.Tool{{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)}})

	// No session is open, so reopening is a no-op and only the state it would
	// have been opened with is observable.
	s.handleContext(context.Background(), convo)

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.tools) != 1 || s.tools[0].Name != "get_weather" {
		t.Errorf("tools = %v, want the conversation's own", s.tools)
	}
}
