// Package wire holds the pieces of Gemini's wire format that the Gemini, Gemini
// Live and Interactions services share.
package wire

import (
	"encoding/json"
	"strings"
)

// TextBookmarkMatches reports whether a thought signature's text bookmark names
// a reply's text. Whitespace is normalized before comparing, and a prefix either
// way matches: the reply is cut short when the response was interrupted, and
// the bookmark holds only the text seen so far when the signature arrived early.
func TextBookmarkMatches(bookmark, text string) bool {
	want := strings.Join(strings.Fields(bookmark), " ")
	got := strings.Join(strings.Fields(text), " ")
	return got != "" && (strings.HasPrefix(want, got) || strings.HasPrefix(got, want))
}

// FunctionResponseDict shapes a tool result for Gemini's functionResponse,
// which requires an object. A result that decodes to a JSON object passes
// through as it stands. Any other JSON value (null, a scalar, an array) is
// wrapped decoded, as {"value": <value>}, and text that is not JSON at all, such
// as the "COMPLETED" a function returning nothing reports, is wrapped as it
// stands.
func FunctionResponseDict(content string) any {
	dec := json.NewDecoder(strings.NewReader(content))
	dec.UseNumber()
	var decoded any
	if dec.Decode(&decoded) != nil || dec.More() {
		return map[string]any{"value": content}
	}
	if _, ok := decoded.(map[string]any); ok {
		return json.RawMessage(content)
	}
	return map[string]any{"value": decoded}
}
