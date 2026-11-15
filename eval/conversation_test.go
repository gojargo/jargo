package eval_test

import (
	"reflect"
	"slices"
	"testing"

	"github.com/gojargo/jargo/eval"
)

// cancelledKey is the key a tool_call message says whether it was canceled
// under.
const cancelledKey = "cancelled" //nolint:misspell // the conversation file's spelling

func TestConversationABotTurnIsWhatItSaidBetweenUserTurnsWithItsCallsBeforeIt(t *testing.T) {
	c := eval.NewConversation()
	c.AddAssistantMessage("Hello!")
	c.AddUserMessage("A table at six, please.")
	c.AddAssistantMessage("Let me check.")
	c.AddToolCall("check_availability", map[string]any{"time": "6"})
	c.AddAssistantMessage("Six is free.")
	c.EndBotTurn()
	want := []map[string]any{
		{"role": "assistant", "content": "Hello!"},
		{"role": "user", "content": "A table at six, please."},
		{"role": "tool_call", "name": "check_availability", "arguments": map[string]any{"time": "6"}, cancelledKey: false},
		{"role": "assistant", "content": "Let me check. Six is free."},
	}
	if got := c.Messages(); !reflect.DeepEqual(got, want) {
		t.Fatalf("messages = %v, want %v", got, want)
	}
}

func TestConversationATurnInWhichNothingWasSaidIsNotATurn(t *testing.T) {
	c := eval.NewConversation()
	c.AddUserMessage("Hi.")
	c.AddToolCall("lookup", map[string]any{})
	c.AddAssistantMessage("  ")
	c.AddUserMessage("Hello?")
	c.AddUserMessage("")
	c.EndBotTurn()
	var roles []string
	for _, m := range c.Messages() {
		role, _ := m["role"].(string)
		roles = append(roles, role)
	}
	if want := []string{"user", "tool_call", "user"}; !slices.Equal(roles, want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
}

func TestConversationACancelMarksTheLatestCallByThatNameStillStanding(t *testing.T) {
	c := eval.NewConversation()
	c.AddToolCall("book", map[string]any{"time": "6"})
	c.AddToolCall("book", map[string]any{"time": "7"})
	canceledFlags := func() []bool {
		var out []bool
		for _, m := range c.Messages() {
			canceled, _ := m[cancelledKey].(bool)
			out = append(out, canceled)
		}
		return out
	}
	c.CancelToolCall("book")
	if got := canceledFlags(); !slices.Equal(got, []bool{false, true}) {
		t.Fatalf("canceled = %v", got)
	}
	c.CancelToolCall("book")
	if got := canceledFlags(); !slices.Equal(got, []bool{true, true}) {
		t.Fatalf("canceled = %v", got)
	}
	c.CancelToolCall("lookup")
	if got := canceledFlags(); !slices.Equal(got, []bool{true, true}) {
		t.Fatalf("canceled = %v", got)
	}
}

func TestConversationTheUserSpokeLastUntilTheBotSaysSomething(t *testing.T) {
	c := eval.NewConversation()
	if c.UserSpokeLast() {
		t.Fatal("an empty conversation has no last speaker")
	}
	c.AddUserMessage("Bye.")
	if !c.UserSpokeLast() {
		t.Fatal("the user spoke last")
	}
	// A call is not something said.
	c.AddToolCall("end", map[string]any{})
	if !c.UserSpokeLast() {
		t.Fatal("a call is not something said")
	}
	c.AddAssistantMessage("Goodbye!")
	if c.UserSpokeLast() {
		t.Fatal("the bot spoke last")
	}
}
