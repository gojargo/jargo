package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// The two kinds of piece mix inside one turn, so each case here is a shape the
// conversation really produces: a model streaming a sentence with its spacing,
// a synthesizer reporting spoken words without it, and a turn that switches from
// one to the other partway through.
func TestConcatenate(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parts []text.Part
		want  string
	}{
		{"nothing", nil, ""},
		{
			"a run that carries its own spacing",
			[]text.Part{{Text: "Hello", IncludesInterPartSpaces: true}, {Text: " there", IncludesInterPartSpaces: true}},
			"Hello there",
		},
		{
			"a run that carries none",
			[]text.Part{{Text: "Hello"}, {Text: "there"}},
			"Hello there",
		},
		{
			"a transition where neither side brought a space",
			[]text.Part{{Text: "Hello", IncludesInterPartSpaces: true}, {Text: "there"}},
			"Hello there",
		},
		{
			"a transition where the left side already ends in one",
			[]text.Part{{Text: "Hello ", IncludesInterPartSpaces: true}, {Text: "there"}},
			"Hello there",
		},
		{
			"a transition where the right side already starts with one",
			[]text.Part{{Text: "Hello"}, {Text: " there", IncludesInterPartSpaces: true}},
			"Hello there",
		},
		{
			"empty pieces contribute nothing, not a space",
			[]text.Part{{Text: "Hello"}, {Text: ""}, {Text: "there"}},
			"Hello there",
		},
		{
			"the whole is trimmed",
			[]text.Part{{Text: " Hello there ", IncludesInterPartSpaces: true}},
			"Hello there",
		},
		{
			"a spoken run following a streamed sentence",
			[]text.Part{
				{Text: "I can", IncludesInterPartSpaces: true},
				{Text: " help.", IncludesInterPartSpaces: true},
				{Text: "One"},
				{Text: "moment."},
			},
			"I can help. One moment.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := text.Concatenate(tc.parts); got != tc.want {
				t.Fatalf("Concatenate() = %q, want %q", got, tc.want)
			}
		})
	}
}
