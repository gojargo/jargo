package realtime

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "openai_realtime" {
		t.Errorf("id = %q, want %q", got, "openai_realtime")
	}
}

// TestToProviderToolsFormat checks the Realtime API's flattened tool shape: the
// function fields sit on the tool rather than nested under a "function" key.
func TestToProviderToolsFormat(t *testing.T) {
	out := (&Adapter{}).ToProviderToolsFormat([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look it up",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	if len(out) != 1 {
		t.Fatalf("tools = %+v, want one", out)
	}
	if out[0][keyType] != toolTypeFunction || out[0][keyName] != "get_weather" {
		t.Errorf("tool = %+v, want a flattened function tool", out[0])
	}
	if out[0][keyDescription] != "Look it up" {
		t.Errorf("description = %v, want the tool's own", out[0][keyDescription])
	}
}

// TestToProviderToolsFormatOmitsEmptyFields checks a tool with no description or
// schema states neither, rather than sending them empty.
func TestToProviderToolsFormatOmitsEmptyFields(t *testing.T) {
	out := (&Adapter{}).ToProviderToolsFormat([]frames.Tool{{Name: "now"}})
	if _, ok := out[0][keyDescription]; ok {
		t.Errorf("tool = %+v, want no description key", out[0])
	}
	if _, ok := out[0][keyParameters]; ok {
		t.Errorf("tool = %+v, want no parameters key", out[0])
	}
}

// TestSessionParamsDefaultsTheChoice checks a conversation that states no choice
// is sent the default rather than an empty one, which would leave the model
// without an answer to whether it may call a tool.
func TestSessionParamsDefaultsTheChoice(t *testing.T) {
	p := (&Adapter{}).SessionParams([]frames.Tool{{Name: "now"}}, "")
	if p.ToolChoice != string(frames.ToolChoiceAuto) {
		t.Errorf("choice = %q, want %q", p.ToolChoice, frames.ToolChoiceAuto)
	}
}

// TestSessionIsNilWithoutTools checks a conversation advertising no tools
// renders no function-calling block at all.
func TestSessionIsNilWithoutTools(t *testing.T) {
	if got := (&Adapter{}).SessionParams(nil, frames.ToolChoiceAuto).Session(); got != nil {
		t.Errorf("session = %+v, want nil", got)
	}
}

// TestSessionCarriesToolsAndChoice checks both halves reach the session payload.
func TestSessionCarriesToolsAndChoice(t *testing.T) {
	got := (&Adapter{}).
		SessionParams([]frames.Tool{{Name: "now"}}, frames.ToolChoiceRequired).
		Session()
	tools, ok := got[keyTools].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("session = %+v, want the toolset on it", got)
	}
	if got[keyToolChoice] != string(frames.ToolChoiceRequired) {
		t.Errorf("choice = %v, want %q", got[keyToolChoice], frames.ToolChoiceRequired)
	}
}

// TestLLMInvocationParamsReadsTheConversation checks the conversation's own
// toolset and choice are what a session is built from.
func TestLLMInvocationParamsReadsTheConversation(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.SetTools([]frames.Tool{{Name: "now"}})
	convo.SetToolChoice(frames.ToolChoiceRequired)

	p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	if len(p.Tools) != 1 || p.Tools[0][keyName] != "now" {
		t.Errorf("tools = %+v, want the conversation's own", p.Tools)
	}
	if p.ToolChoice != string(frames.ToolChoiceRequired) {
		t.Errorf("choice = %q, want the conversation's own", p.ToolChoice)
	}
}

// TestMessagesForLoggingIsEmpty checks a realtime session renders no message
// list: the model hears the conversation rather than being sent it.
func TestMessagesForLoggingIsEmpty(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("hello")
	if got := (&Adapter{}).MessagesForLogging(convo); got != nil {
		t.Errorf("logged = %+v, want nothing", got)
	}
}
