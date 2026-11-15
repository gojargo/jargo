package main

import "testing"

// TestSearchToolsAreOffered checks Google Search and code execution are
// offered in Gemini's own format, with no standard tools.
func TestSearchToolsAreOffered(t *testing.T) {
	if len(tools().Standard) != 0 {
		t.Fatalf("standard tools = %v, want none", tools().Standard)
	}
	custom := tools().CustomFor("gemini")
	if len(custom) != 2 {
		t.Fatalf("custom tools = %v, want Google Search and code execution", custom)
	}
	for i, name := range []string{"google_search", "code_execution"} {
		if m, ok := custom[i].(map[string]any); !ok || m[name] == nil {
			t.Errorf("custom tool %d = %v, want %s", i, custom[i], name)
		}
	}
}

// TestTruncate checks a long search result is cut to its first characters.
func TestTruncate(t *testing.T) {
	if got := truncate("héllo", 2); got != "hé" {
		t.Errorf("truncate = %q, want %q", got, "hé")
	}
	if got := truncate("hi", 200); got != "hi" {
		t.Errorf("truncate = %q, want %q", got, "hi")
	}
}
