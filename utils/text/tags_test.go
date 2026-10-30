package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// TestParseStartEndTags checks the tag the text ends inside and where the next
// scan resumes: just past the open tag's start, or past the last tag closed.
func TestParseStartEndTags(t *testing.T) {
	a := text.StartEndTags{Start: "<a>", End: "</a>"}
	spell := text.StartEndTags{Start: "spell(", End: ")"}
	bold := text.StartEndTags{Start: "**", End: "**"}

	for _, tc := range []struct {
		name  string
		text  string
		tags  []text.StartEndTags
		index int
		open  *text.StartEndTags
		next  int
	}{
		{"empty", "", nil, 0, nil, 0},
		{"no tags", "Hello from Pipecat!", nil, 0, nil, 0},

		{"closed", "Hello from <a>Pipecat</a>!", []text.StartEndTags{a}, 0, nil, 25},
		{"open", "Hello from <a>Pipecat", []text.StartEndTags{a}, 0, &a, 14},
		{"open from an offset", "Hello from <a>Pipecat", []text.StartEndTags{a}, 6, &a, 14},
		{"closed call", "Hello from spell(Pipecat)!", []text.StartEndTags{spell}, 0, nil, 25},
		{"open call", "Hello from spell(Pipecat", []text.StartEndTags{spell}, 0, &spell, 17},

		{"two closed", "Hello from <a>Pipecat</a>! Hello <a>World</a>!", []text.StartEndTags{a}, 0, nil, 45},
		{"closed then open", "Hello from <a>Pipecat</a>! Hello <a>World", []text.StartEndTags{a}, 0, &a, 36},

		// A tag of any pair opens, not only the first pair's.
		{"second pair opens", "Hello spell(Pip", []text.StartEndTags{a, spell}, 0, &spell, 12},
		{"both pairs close", "<a>Hi</a> and spell(Pipecat) now", []text.StartEndTags{a, spell}, 0, nil, 28},

		{"one delimiter open", "It's **back", []text.StartEndTags{bold}, 0, &bold, 7},
		{"one delimiter closed", "It's **back up** now", []text.StartEndTags{bold}, 0, nil, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			open, next := text.ParseStartEndTags(tc.text, tc.tags, nil, tc.index)
			if (open == nil) != (tc.open == nil) || (open != nil && *open != *tc.open) || next != tc.next {
				t.Errorf("ParseStartEndTags(%q) = %v, %d; want %v, %d", tc.text, open, next, tc.open, tc.next)
			}
		})
	}
}
