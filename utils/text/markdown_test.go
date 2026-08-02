package text

import (
	"fmt"
	"strings"
	"testing"
)

func newTestMarkdownFilter() *MarkdownFilter {
	return NewMarkdownFilter(DefaultMarkdownFilterOptions())
}

func TestMarkdownFilterBasicRemoval(t *testing.T) {
	input := "" +
		"            **Bold text** and *italic text*\n" +
		"            1. Numbered list item\n" +
		"            - Bullet point\n" +
		"            Some `inline code` here\n\n## Subtitle\n"
	want := "" +
		"            Bold text and italic text\n" +
		"            1. Numbered list item\n" +
		"            - Bullet point\n" +
		"            Some inline code here\nSubtitle"

	got := newTestMarkdownFilter().Filter(input)
	if strings.TrimSpace(got) != strings.TrimSpace(want) {
		t.Errorf("Filter() =\n%q\nwant\n%q", strings.TrimSpace(got), strings.TrimSpace(want))
	}
}

func TestMarkdownFilterPreservesSpaces(t *testing.T) {
	for _, in := range []string{
		"  Leading spaces",
		"Trailing spaces  ",
		"  Both ends  ",
		"  Multiple  spaces  between  words  ",
	} {
		got := newTestMarkdownFilter().Filter(in)
		if len(got) != len(in) {
			t.Errorf("Filter(%q) = %q, length %d want %d", in, got, len(got), len(in))
			continue
		}
		for i := range in {
			if in[i] == ' ' && got[i] != ' ' {
				t.Errorf("Filter(%q) = %q, space at %d not preserved", in, got, i)
			}
		}
	}
}

func TestMarkdownFilterRepeatedCharacters(t *testing.T) {
	cases := map[string]string{
		"Hello!!!!!World":      "HelloWorld",
		"Test####ing":          "Test####ing",
		"Normal text":          "Normal text",
		"!!!!!":                "",
		"Mixed!!!!!...../////": "Mixed",
		"Text^^^^test":         "Text^^^^test",
		"Text^^^^^test":        "Texttest",
		"Dots....here":         "Dots....here",
		"Dots.....here":        "Dotshere",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterKeepsRepeatedCharactersWhenOff(t *testing.T) {
	opts := DefaultMarkdownFilterOptions()
	opts.FilterRepeatedSequences = false
	if got := NewMarkdownFilter(opts).Filter("55555"); got != "55555" {
		t.Errorf("Filter() = %q, want %q", got, "55555")
	}
}

func TestMarkdownFilterNumberedList(t *testing.T) {
	input := "1. First item\n        2. Second item\n        3. Third item with **bold**"
	want := "1. First item\n        2. Second item\n        3. Third item with bold"
	got := newTestMarkdownFilter().Filter(input)
	if strings.TrimSpace(got) != strings.TrimSpace(want) {
		t.Errorf("Filter() =\n%q\nwant\n%q", strings.TrimSpace(got), strings.TrimSpace(want))
	}
}

func TestMarkdownFilterHTMLEntities(t *testing.T) {
	cases := map[string]string{
		"This &amp; that":              "This & that",
		"1 &lt; 2":                     "1 < 2",
		"2 &gt; 1":                     "2 > 1",
		"Line&nbsp;break":              "Line break",
		"Mixed &amp; &lt;entities&gt;": "Mixed & <entities>",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterAsterisks(t *testing.T) {
	cases := map[string]string{
		"**bold text**":         "bold text",
		"*italic text*":         "italic text",
		"**bold** and *italic*": "bold and italic",
		"multiple**bold**words": "multipleboldwords",
		"edge**cases***here*":   "edgecaseshere",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterNewlines(t *testing.T) {
	cases := map[string]string{
		"Line 1\n\nLine 2":    "Line 1\n Line 2",
		"Line 1\n   \nLine 2": "Line 1\n Line 2",
		"Text\n\n\nMore":      "Text\n More",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterStripsScheme(t *testing.T) {
	cases := map[string]string{
		"Please check http://example.com":       "Please check example.com",
		"Visit https://www.google.com for more": "Visit www.google.com for more",
		"No link here":                          "No link here",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterNumberedListMarkers(t *testing.T) {
	cases := map[string]string{
		"1. First\n2. Second":    "1. First\n2. Second",
		"  1. Indented":          "  1. Indented",
		"1. Item\nText\n2. Item": "1. Item\nText\n2. Item",
		"1.No space":             "1.No space",
		"12. Large number":       "12. Large number",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterInlineCode(t *testing.T) {
	cases := map[string]string{
		"`code`":              "code",
		"Text `code` more":    "Text code more",
		"``nested`code``":     "nested`code",
		"`code1` and `code2`": "code1 and code2",
		"No``space``between":  "Nospacebetween",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); got != want {
			t.Errorf("Filter(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMarkdownFilterRemovesTables(t *testing.T) {
	opts := DefaultMarkdownFilterOptions()
	opts.FilterTables = true
	// A short delimiter row survives the repeated-sequence rule and is parsed.
	input := "| Column 1 | Column 2 |\n|---|---|\n| Cell 1   | Cell 2   |"
	if got := NewMarkdownFilter(opts).Filter(input); strings.TrimSpace(got) != "" {
		t.Errorf("Filter() = %q, want empty", got)
	}

	// A delimiter row of five or more dashes is flattened by the repeated-sequence
	// rule before the text is parsed, and the renderer will not take a table
	// without a well-formed delimiter row. Turning that rule off leaves the row
	// intact and the table is dropped as it should be.
	long := "| Column 1 | Column 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |"
	opts.FilterRepeatedSequences = false
	if got := NewMarkdownFilter(opts).Filter(long); strings.TrimSpace(got) != "" {
		t.Errorf("Filter() with repeated sequences off = %q, want empty", got)
	}
}

func TestMarkdownFilterDisabled(t *testing.T) {
	in := "**bold** and *italic* with `code`"
	if got := NewMarkdownFilter(MarkdownFilterOptions{}).Filter(in); got != in {
		t.Errorf("disabled Filter() = %q, want %q", got, in)
	}
	if got := newTestMarkdownFilter().Filter(in); got != "bold and italic with code" {
		t.Errorf("enabled Filter() = %q, want %q", got, "bold and italic with code")
	}
}

func TestMarkdownFilterHeaderStripping(t *testing.T) {
	got := newTestMarkdownFilter().Filter("# Title\n\n## Subtitle\n")
	if want := "Title\nSubtitle"; strings.TrimSpace(got) != want {
		t.Errorf("Filter() = %q, want %q", strings.TrimSpace(got), want)
	}
}

func TestMarkdownFilterClosedATXHeaders(t *testing.T) {
	cases := map[string]string{
		"## Subtitle ##":            "Subtitle",
		"### Deep ###":              "Deep",
		"## Done ###":               "Done",
		"# Title\n\n## Subtitle ##": "Title\nSubtitle",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); strings.TrimSpace(got) != want {
			t.Errorf("Filter(%q) = %q, want %q", in, strings.TrimSpace(got), want)
		}
	}
}

func TestMarkdownFilterHeaderTrailingSpace(t *testing.T) {
	if got := newTestMarkdownFilter().Filter("## Trailing   "); got != "Trailing   " {
		t.Errorf("Filter() = %q, want %q", got, "Trailing   ")
	}
}

func TestMarkdownFilterHeaderLevels(t *testing.T) {
	for n := 1; n <= 6; n++ {
		in := fmt.Sprintf("%s Heading", strings.Repeat("#", n))
		if got := newTestMarkdownFilter().Filter(in); strings.TrimSpace(got) != "Heading" {
			t.Errorf("Filter(%q) = %q, want %q", in, strings.TrimSpace(got), "Heading")
		}
	}
}

func TestMarkdownFilterHashInContent(t *testing.T) {
	cases := map[string]string{
		"C# is great":        "C# is great",
		"## Issue #42 today": "Issue #42 today",
		"## C# matters":      "C# matters",
	}
	for in, want := range cases {
		if got := newTestMarkdownFilter().Filter(in); strings.TrimSpace(got) != want {
			t.Errorf("Filter(%q) = %q, want %q", in, strings.TrimSpace(got), want)
		}
	}
}
