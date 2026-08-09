package mistral

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// answeredConvo is a conversation in which call_a has been made and answered
// and call_b has not been made at all.
func answeredConvo() *frames.LLMContext {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "call_a", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "call_a", Name: "get_weather", Content: "sunny"})
	return convo
}

func toolCall(id, name string) frames.ToolCall {
	return frames.ToolCall{ID: id, Name: name, Args: json.RawMessage(`{}`)}
}

// TestDropAnsweredCalls checks the completion that answers a tool result does
// not run that tool a second time, while a call the conversation has no result
// for is left alone.
func TestDropAnsweredCalls(t *testing.T) {
	convo := answeredConvo()

	kept := dropAnsweredCalls(convo, []frames.ToolCall{
		toolCall("call_a", "get_weather"),
		toolCall("call_b", "get_time"),
	})
	if len(kept) != 1 || kept[0].ID != "call_b" {
		t.Errorf("kept = %+v, want only the call the conversation has no result for", kept)
	}

	if kept := dropAnsweredCalls(convo, []frames.ToolCall{toolCall("call_a", "get_weather")}); len(kept) != 0 {
		t.Errorf("kept = %+v, want nothing: every call was answered already", kept)
	}

	fresh := []frames.ToolCall{toolCall("call_c", "get_time")}
	if kept := dropAnsweredCalls(frames.NewLLMContext(""), fresh); len(kept) != 1 {
		t.Errorf("kept = %+v, want the call: an empty conversation has answered nothing", kept)
	}

	// An id is what pairs a call to its result. Without one there is nothing to
	// match, so the call runs rather than being swallowed by the first result.
	if kept := dropAnsweredCalls(convo, []frames.ToolCall{toolCall("", "get_time")}); len(kept) != 1 {
		t.Errorf("kept = %+v, want a call with no id to run", kept)
	}
}
