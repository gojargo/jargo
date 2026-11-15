package main

import "testing"

// TestGoogleSearchIsOffered checks Google Search is offered in Gemini's own
// format, and is the only tool.
func TestGoogleSearchIsOffered(t *testing.T) {
	if len(tools().Standard) != 0 {
		t.Fatalf("standard tools = %v, want none", tools().Standard)
	}
	custom := tools().CustomFor("gemini")
	if len(custom) != 1 {
		t.Fatalf("custom tools = %v, want Google Search", custom)
	}
	if m, ok := custom[0].(map[string]any); !ok || m["google_search"] == nil {
		t.Errorf("custom tool = %v, want google_search", custom[0])
	}
}
