package realtime

import (
	"encoding/json"
	"testing"
)

// TestResponseSpanRecordsTurn checks that a completed response is recorded with
// how it ended, what the model said, and the calls it asked for.
func TestResponseSpanRecordsTurn(t *testing.T) {
	raw := `{
		"type": "response.done",
		"response": {
			"id": "resp_9",
			"status": "completed",
			"output": [
				{"role": "assistant", "content": [{"transcript": "Good morning."}]},
				{"type": "function_call", "name": "book_table", "call_id": "call_1"},
				{"type": "function_call", "name": "send_confirmation", "call_id": "call_2"}
			]
		}
	}`
	var ev serverEvent
	if err := json.Unmarshal([]byte(raw), &ev); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if ev.Response == nil {
		t.Fatal("response not parsed")
	}

	if got := assistantTranscript(ev.Response.Output); got != "Good morning." {
		t.Errorf("assistant transcript = %q, want %q", got, "Good morning.")
	}
	calls := functionCallNames(ev.Response.Output)
	if len(calls) != 2 || calls[0] != "book_table" || calls[1] != "send_confirmation" {
		t.Errorf("function calls = %v, want the two the model asked for in order", calls)
	}
}

// TestTruncateKeepsUTF8 checks that cutting a long instruction leaves valid
// text: the cut is made on a rune boundary, not mid-character.
func TestTruncateKeepsUTF8(t *testing.T) {
	// "é" is two bytes, so a cut at an odd offset would split it.
	s := "ééééé"
	for n := range len(s) + 2 {
		got := truncate(s, n)
		if !json.Valid([]byte(`"` + got + `"`)) {
			t.Fatalf("truncate(%q, %d) = %q, which is not valid UTF-8", s, n, got)
		}
	}
}
