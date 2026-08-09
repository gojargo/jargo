package adapter_test

import (
	"errors"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// names returns the name of each tool, which is what these tests compare.
func names(tools []frames.Tool) []string {
	out := make([]string, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Name)
	}
	return out
}

// wantNames fails unless the tools carry exactly the given names in order.
func wantNames(t *testing.T, tools []frames.Tool, want ...string) {
	t.Helper()
	got := names(tools)
	if len(got) != len(want) {
		t.Fatalf("tools = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tool %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestBuiltinsFollowTheConversationsOwn checks a tool the service implements is
// sent after the ones the application advertised, so what the application asked
// for reaches the model in the order it stated.
func TestBuiltinsFollowTheConversationsOwn(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "cancel"}})

	wantNames(t, b.WithBuiltins([]frames.Tool{{Name: "watch"}}), "watch", "cancel")
}

// TestBuiltinsKeepTheirOrder checks two built-in tools are sent the same way
// twice running. A request that reorders them would rewrite a cached prompt
// prefix on every turn and never read one back.
func TestBuiltinsKeepTheirOrder(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "first"}})
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "second"}})

	for range 3 {
		wantNames(t, b.WithBuiltins(nil), "first", "second")
	}
}

// TestSetBuiltinReplacesWithoutReordering checks re-registering a tool updates
// it in place rather than moving it to the end.
func TestSetBuiltinReplacesWithoutReordering(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "first"}})
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "second"}})
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "first", Description: "updated"}})

	tools := b.WithBuiltins(nil)
	wantNames(t, tools, "first", "second")
	if tools[0].Description != "updated" {
		t.Errorf("description = %q, want the re-registered tool's", tools[0].Description)
	}
}

// TestRemoveBuiltin checks a withdrawn tool stops being sent, and that
// withdrawing one that was never registered says so.
func TestRemoveBuiltin(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "cancel"}})

	if !b.RemoveBuiltin("cancel") {
		t.Error("RemoveBuiltin said there was no such tool, want it removed")
	}
	wantNames(t, b.WithBuiltins([]frames.Tool{{Name: "watch"}}), "watch")

	if b.RemoveBuiltin("cancel") {
		t.Error("RemoveBuiltin said it removed a tool that was not registered")
	}
}

// TestWithBuiltinsLeavesTheConversationAlone checks the merge does not write
// into the slice it was given, which the conversation still holds.
func TestWithBuiltinsLeavesTheConversationAlone(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "cancel"}})

	tools := []frames.Tool{{Name: "watch"}}
	b.WithBuiltins(tools)
	wantNames(t, tools, "watch")
}

// TestSystemWithBuiltins checks the instructions of an offered tool are appended
// to the prompt, so the model is told how to use what it is being sent.
func TestSystemWithBuiltins(t *testing.T) {
	var b adapter.Base
	if got := b.SystemWithBuiltins("be brief"); got != "be brief" {
		t.Errorf("system = %q, want it unchanged with no built-in tools", got)
	}

	b.SetBuiltin(adapter.Builtin{
		Tool:         frames.Tool{Name: "cancel"},
		Instructions: "how to cancel",
	})
	if got := b.SystemWithBuiltins("be brief"); got != "be brief\n\nhow to cancel" {
		t.Errorf("system = %q, want the instructions appended", got)
	}
	if got := b.SystemWithBuiltins(""); got != "how to cancel" {
		t.Errorf("system = %q, want the instructions to stand alone", got)
	}
}

// TestSystemWithBuiltinsSkipsToolsThatSayNothing checks a built-in tool needing
// no explanation adds no blank run to the prompt.
func TestSystemWithBuiltinsSkipsToolsThatSayNothing(t *testing.T) {
	var b adapter.Base
	b.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "cancel"}})

	if got := b.SystemWithBuiltins("be brief"); got != "be brief" {
		t.Errorf("system = %q, want it unchanged", got)
	}
}

// TestConversionErrorCarriesItsCause checks the failure a conversion wrapped is
// still reachable, so what actually went wrong can be read off it.
func TestConversionErrorCarriesItsCause(t *testing.T) {
	cause := testError{}
	err := &adapter.ConversionError{Cause: cause}
	if !errors.Is(err.Unwrap(), cause) {
		t.Errorf("Unwrap = %v, want the cause", err.Unwrap())
	}
	if got := err.Error(); got == "" {
		t.Error("Error() is empty, want it to name the failure")
	}
}

// testError is a stand-in cause for the wrapping test.
type testError struct{}

func (testError) Error() string { return "boom" }
