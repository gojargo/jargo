package kokoro

import (
	"reflect"
	"testing"
)

// Tests for the punctuation processing around phonemization, which loads
// nothing.

func TestPunctuationRemove(t *testing.T) {
	for in, want := range map[string]string{
		"a, b,c.":        "a b c",
		"abc de":         "abc de",
		"!d.d. dd??  d!": "d d dd d",
	} {
		if got := newPunctuation(defaultPunctuationMarks).remove([]string{in})[0]; got != want {
			t.Errorf("remove(%q) = %q, want %q", in, got, want)
		}
	}
}

// Restoring what was preserved gives the text back.
func TestPunctuationPreserveRestore(t *testing.T) {
	for _, in := range [][]string{
		{".a.b.c."},
		{"a, a?", "b, b"},
		{"a, a?", "b, b", "!"},
		{"a, a?", "!?", "b, b"},
		{"!?", "a, a?", "b, b"},
		{"a, a, a"},
		{"a, a?", "aaa bb", ".bb, b", "c", "!d.d. dd??  d!"},
		{`Truly replied, "Yes".`},
		{`hi; ho,"`},
		{"!?"},
		{"!'"},
		{"It is ! (I think so)"},
		{"This {is} right"},
		{"[He] is right"},
		{"1,5 and more"},
		{"19,99 euro"},
		{"hello, world"},
		{"3.14 is pi."},
		{"1.5."},
	} {
		p := newPunctuation(defaultPunctuationMarks)
		chunks, lines := p.preserve(in)
		if got := restore(chunks, lines, defaultSeparator, true, nil); !reflect.DeepEqual(got, in) {
			t.Errorf("restore(preserve(%q)) = %q", in, got)
		}
	}
}

// A final chunk is restored as is: phonemized, it ends with a word separator,
// or is empty and the mark before it is followed by one.
func TestPunctuationPreserveRestoreUnstripped(t *testing.T) {
	p := newPunctuation(".!;:,?")
	for in, want := range map[string]string{
		`hi; hi,"`: `hi; hi,"`,
		`hi; "hi,`: `hi; "hi, `,
		`"hi; hi,`: `"hi; hi, `,
	} {
		chunks, lines := p.preserve([]string{in})
		if got := restore(chunks, lines, defaultSeparator, false, nil)[0]; got != want {
			t.Errorf("restore(preserve(%q)) = %q, want %q", in, got, want)
		}
	}
}

// A mark that doubles as a decimal separator does not split a number: espeak
// would read two separate numbers and the value spoken would change. A
// separator next to a non-digit is ordinary punctuation and is still split.
func TestDecimalSeparatorNotSplit(t *testing.T) {
	for in, want := range map[string]string{
		"1,5":               "1,5",
		"3.14":              "3.14",
		"19,99 euro":        "19,99 euro",
		"version 2.5":       "version 2.5",
		"1,5, and more":     "1,5 and more",
		"I have 42.":        "I have 42",
		"42, 43":            "42 43",
		"line 42, column 7": "line 42 column 7",
	} {
		if got := newPunctuation(defaultPunctuationMarks).remove([]string{in})[0]; got != want {
			t.Errorf("remove(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPreserveParts(t *testing.T) {
	text := []string{"hello, my world!", "...", "ok", `"a (b) c", d`}
	chunks, lines := newPunctuation(defaultPunctuationMarks).preserve(text)
	if want := []string{"hello", "my world", "ok", "a", "b", "c", "d"}; !reflect.DeepEqual(chunks, want) {
		t.Errorf("chunks = %q, want %q", chunks, want)
	}
	c := func(n int) linePart { return linePart{chunk: n} }
	m := func(s string) linePart { return linePart{mark: s, isMark: true} }
	want := [][]linePart{
		{c(0), m(", "), c(1), m("!")},
		{m("...")},
		{c(2)},
		{m(`"`), c(3), m(" ("), c(4), m(") "), c(5), m(`", `), c(6)},
	}
	if !reflect.DeepEqual(lines, want) {
		t.Errorf("lines = %+v, want %+v", lines, want)
	}
	// The chunks separated by quotes or brackets only.
	if got := inlineGroups(lines); !reflect.DeepEqual(got, [][]int{{3, 4, 5}}) {
		t.Errorf("inlineGroups = %v, want [[3 4 5]]", got)
	}
}

func TestIsInline(t *testing.T) {
	for mark, want := range map[string]bool{
		"(": true, " « ": true, `") `: true, ", ": false, `", `: false, "-": false, " ": false, "": false,
	} {
		if got := isInline(mark); got != want {
			t.Errorf("isInline(%q) = %v, want %v", mark, got, want)
		}
	}
}

func TestRestoreContext(t *testing.T) {
	_, lines := newPunctuation(defaultPunctuationMarks).preserve([]string{"a (b c)"})
	phonemized := []string{"a ", "b c "}
	for _, tc := range []struct {
		context string
		want    string
	}{
		// No context, or the words cannot be aligned with the chunks.
		{"", "a (b c) "},
		{"x y ", "a (b c) "},
		{"x y z w ", "a (b c) "},
		// Aligned: the mark goes after the words of the first chunk.
		{"x y z ", "x (y z) "},
		{"x  y z ", "x (y z) "},
	} {
		var context []contextGroup
		if tc.context != "" {
			context = []contextGroup{{chunks: []int{0, 1}, phonemized: tc.context}}
		}
		if got := restore(phonemized, lines, defaultSeparator, false, context)[0]; got != tc.want {
			t.Errorf("restore with context %q = %q, want %q", tc.context, got, tc.want)
		}
	}

	// The words cannot be counted without a word separator.
	if got := restore([]string{"a", "bc"}, lines, separator{phone: "-"}, false,
		[]contextGroup{{chunks: []int{0, 1}, phonemized: "xyz"}})[0]; got != "a(bc)" {
		t.Errorf("restore without a word separator = %q, want a(bc)", got)
	}
	if got := restore([]string{"a|", "b|c|"}, lines, separator{word: "|", phone: " | "}, false,
		[]contextGroup{{chunks: []int{0, 1}, phonemized: "x|y|z|"}})[0]; got != "a|(b|c)|" {
		t.Errorf("restore with the word separator in the phone one = %q, want a|(b|c)|", got)
	}
}

// The chunks of a group must be separated by a single inline mark in a line,
// else the group is ignored.
func TestRestoreContextNotAGroup(t *testing.T) {
	_, lines := newPunctuation(defaultPunctuationMarks).preserve([]string{"a (b), c"})
	phonemized := []string{"a ", "b ", "c "}
	for _, group := range [][]int{{0, 2}, {1, 2}, {0, 5}} {
		got := restore(phonemized, lines, defaultSeparator, false,
			[]contextGroup{{chunks: group, phonemized: "x y "}})
		if got[0] != "a (b), c " {
			t.Errorf("restore with group %v = %q, want %q", group, got[0], "a (b), c ")
		}
	}
}

func TestRestoreKeepsInput(t *testing.T) {
	_, lines := newPunctuation(defaultPunctuationMarks).preserve([]string{"a, b!"})
	phonemized := []string{"a ", "b "}
	restore(phonemized, lines, defaultSeparator, false, nil)
	if !reflect.DeepEqual(phonemized, []string{"a ", "b "}) {
		t.Errorf("restore changed its input: %q", phonemized)
	}
}

func TestRestoreMarksAlone(t *testing.T) {
	got := restore(nil, [][]linePart{{{mark: "! ?", isMark: true}}}, separator{word: "\\1"}, true, nil)
	if got[0] != `!\1?` {
		t.Errorf("restore = %q, want %q", got[0], `!\1?`)
	}
}
