package frames_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestAsyncToolMessagesRoundTrip checks the three stages of the protocol survive
// the trip through a conversation message and back, which is what a realtime
// service relies on when it spots these while walking the context.
func TestAsyncToolMessagesRoundTrip(t *testing.T) {
	tests := []struct {
		name    string
		message frames.Message
		want    frames.AsyncToolMessage
		role    frames.Role
	}{
		{
			name:    "started",
			message: frames.NewAsyncToolStartedMessage("c1"),
			want:    frames.AsyncToolMessage{Kind: frames.AsyncToolStarted, ToolCallID: "c1", Status: "running"},
			role:    frames.RoleUser,
		},
		{
			name:    "intermediate",
			message: frames.NewAsyncToolIntermediateMessage("c1", "boarding"),
			want: frames.AsyncToolMessage{
				Kind: frames.AsyncToolIntermediate, ToolCallID: "c1",
				Status: "running", Result: "boarding", HasResult: true,
			},
			role: frames.RoleDeveloper,
		},
		{
			name:    "final",
			message: frames.NewAsyncToolFinalMessage("c1", "departed"),
			want: frames.AsyncToolMessage{
				Kind: frames.AsyncToolFinal, ToolCallID: "c1",
				Status: "finished", Result: "departed", HasResult: true,
			},
			role: frames.RoleDeveloper,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.message.Role != tc.role {
				t.Errorf("role = %q, want %q", tc.message.Role, tc.role)
			}
			got, ok := frames.ParseAsyncToolMessage(tc.message)
			if !ok {
				t.Fatalf("message not recognized as an async-tool message: %+v", tc.message)
			}
			if got.Description == "" {
				t.Error("every stage carries a description telling the model what is happening")
			}
			got.Description = "" // compared separately; the text itself is not the contract
			if got != tc.want {
				t.Errorf("parsed = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// TestAsyncToolStartedMessageAnswersItsCall checks the started message is a
// tool-result message keyed by the call id. That is what makes it answer the
// tool call it belongs to, so the conversation is valid from the moment an
// asynchronous call begins.
func TestAsyncToolStartedMessageAnswersItsCall(t *testing.T) {
	m := frames.NewAsyncToolStartedMessage("c1")
	if len(m.ToolResults) != 1 || m.ToolResults[0].ID != "c1" {
		t.Fatalf("message = %+v, want a tool result keyed by the call id", m)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(m.ToolResults[0].Content), &payload); err != nil {
		t.Fatalf("content is not JSON: %v", err)
	}
	if payload["type"] != "async_tool" || payload["status"] != "running" {
		t.Errorf("payload = %+v, want a running async_tool payload", payload)
	}
	if _, ok := payload["result"]; ok {
		t.Error("a started message carries no result; the task has not produced one")
	}
}

// TestParseAsyncToolMessageRejectsOthers checks ordinary conversation messages
// are not mistaken for the protocol.
func TestParseAsyncToolMessageRejectsOthers(t *testing.T) {
	others := []frames.Message{
		{Role: frames.RoleUser, Text: "hello"},
		{Role: frames.RoleAssistant, Text: "hi"},
		{Role: frames.RoleDeveloper, Text: "not json"},
		{Role: frames.RoleDeveloper, Text: `{"type":"something_else","tool_call_id":"c1","status":"running"}`},
		{Role: frames.RoleDeveloper, Text: `{"type":"async_tool","status":"running"}`},
		{Role: frames.RoleDeveloper, Text: `{"type":"async_tool","tool_call_id":"c1","status":"elsewhere"}`},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "c1", Content: "sunny"}}},
	}
	for _, m := range others {
		if _, ok := frames.ParseAsyncToolMessage(m); ok {
			t.Errorf("message parsed as async-tool but is not one: %+v", m)
		}
	}
}

// TestAsyncToolDescriptionsExplainTheProtocol checks each stage says what it is,
// since the model reads these with no other knowledge of the protocol.
func TestAsyncToolDescriptionsExplainTheProtocol(t *testing.T) {
	started, _ := frames.ParseAsyncToolMessage(frames.NewAsyncToolStartedMessage("c1"))
	if !strings.Contains(started.Description, "status=finished") {
		t.Errorf("started description should say how the last result is recognized: %q", started.Description)
	}
	intermediate, _ := frames.ParseAsyncToolMessage(frames.NewAsyncToolIntermediateMessage("c1", "r"))
	if !strings.Contains(intermediate.Description, "still running") {
		t.Errorf("intermediate description should say the task is still running: %q", intermediate.Description)
	}
	final, _ := frames.ParseAsyncToolMessage(frames.NewAsyncToolFinalMessage("c1", "r"))
	if !strings.Contains(final.Description, "No further results") {
		t.Errorf("final description should say nothing more is coming: %q", final.Description)
	}
}
