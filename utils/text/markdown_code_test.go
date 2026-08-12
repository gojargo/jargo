package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// Tests for the state the Markdown filter carries between calls. A code block or
// a table spans several chunks of streamed text, so the filter has to remember
// it is inside one, and an interruption abandons whatever was half-read because
// the text that would have closed it is never coming.

// filterAll runs each chunk through f and returns what came out of each.
func filterAll(f *text.MarkdownFilter, chunks []string) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, f.Filter(c))
	}
	return out
}

// checkChunks fails the test for the first chunk whose output is not want[i].
func checkChunks(t *testing.T, got, want []string, in []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d outputs, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("chunk %d (%q): got %q, want %q", i, in[i], got[i], want[i])
		}
	}
}

// TestMarkdownFilterDropsStreamedCodeBlock checks that a fenced block arriving a
// chunk at a time is dropped whole: the fence opens the block, everything up to
// the closing fence is skipped, and the prose either side is spoken.
func TestMarkdownFilterDropsStreamedCodeBlock(t *testing.T) {
	in := []string{
		"Here is some code: ",
		"```python\n",
		"def f():\n",
		"    return 1\n",
		"```\n",
		" and that was it.",
	}
	want := []string{"Here is some code: ", "", "", "", " ", " and that was it."}

	f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterCode: true})
	checkChunks(t, filterAll(f, in), want, in)
}

// TestMarkdownFilterKeepsCodeWhenNotFiltering checks the other setting: with
// code filtering off the block is spoken as it was written, fences and all,
// which is what makes the fences visible to the tracking when it is on.
func TestMarkdownFilterKeepsCodeWhenNotFiltering(t *testing.T) {
	in := []string{"Here is some code: ", "```python\n", "def f():\n", "```\n"}
	want := []string{"Here is some code: ", "```python ", "def f(): ", "``` "}

	f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true})
	checkChunks(t, filterAll(f, in), want, in)
}

// TestMarkdownFilterCodeBlockWithinAChunk checks the two shapes that arrive
// inside a single chunk rather than across several.
func TestMarkdownFilterCodeBlockWithinAChunk(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			// Both fences are in the chunk, so the block is cut out and the
			// text either side of it joins up.
			name: "a whole block between text",
			in:   "Here: ```\nprint(1)\n``` done.",
			want: "Here: print(1) done.",
		},
		{
			// A span rather than a block: the backticks come off and what they
			// wrapped is spoken.
			name: "a span on one line",
			in:   "before ```code``` after",
			want: "before code after",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterCode: true})
			if got := f.Filter(tt.in); got != tt.want {
				t.Errorf("Filter(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestMarkdownFilterCodeBlockOpeningMidLine checks a fence that opens partway
// through a line: what came before it is spoken and the rest is held until the
// closing fence arrives.
func TestMarkdownFilterCodeBlockOpeningMidLine(t *testing.T) {
	in := []string{"say ```py\n", "code\n", "```\n", "end"}
	want := []string{"say", "", " ", "end"}

	f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterCode: true})
	checkChunks(t, filterAll(f, in), want, in)
}

// TestMarkdownFilterInterruptionAbandonsCodeBlock checks that an interruption
// drops the half-read block. The closing fence is never coming, so without this
// every later chunk would be swallowed as if it were still code.
func TestMarkdownFilterInterruptionAbandonsCodeBlock(t *testing.T) {
	f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterCode: true})

	if got := f.Filter("intro "); got != "intro " {
		t.Fatalf("Filter(intro) = %q, want %q", got, "intro ")
	}
	if got := f.Filter("```python\n"); got != "" {
		t.Fatalf("the opening fence produced %q, want it dropped", got)
	}
	if got := f.Filter("def f():\n"); got != "" {
		t.Fatalf("the code line produced %q, want it dropped", got)
	}

	// The interruption abandons the block, so the next text is spoken.
	f.HandleInterruption()
	if got := f.Filter("after the interruption"); got != "after the interruption" {
		t.Errorf("after an interruption, Filter() = %q, want the text through", got)
	}

	// And filtering resumes as normal once the interruption is dealt with.
	f.ResetInterruption()
	if got := f.Filter("more text"); got != "more text" {
		t.Errorf("after the reset, Filter() = %q, want the text through", got)
	}
}

// TestMarkdownFilterStreamedTable checks the same tracking for a table that
// arrives split across chunks, and that an interruption abandons it too.
func TestMarkdownFilterStreamedTable(t *testing.T) {
	t.Run("a table split across chunks", func(t *testing.T) {
		in := []string{"before <table><tr>", "<td>x</td>", "</tr></table> after"}
		want := []string{"before", "", "after"}

		f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterTables: true})
		checkChunks(t, filterAll(f, in), want, in)
	})

	t.Run("a whole table in one chunk", func(t *testing.T) {
		f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterTables: true})
		got := f.Filter("before <table><tr><td>x</td></tr></table> after")
		if got != "before  after" {
			t.Errorf("Filter() = %q, want %q", got, "before  after")
		}
	})

	t.Run("an interruption abandons a half-read table", func(t *testing.T) {
		f := text.NewMarkdownFilter(text.MarkdownFilterOptions{Enabled: true, FilterTables: true})
		if got := f.Filter("before <table><tr>"); got != "before" {
			t.Fatalf("Filter() = %q, want %q", got, "before")
		}

		f.HandleInterruption()
		if got := f.Filter("resumed text"); got != "resumed text" {
			t.Errorf("after an interruption, Filter() = %q, want the text through", got)
		}

		f.ResetInterruption()
		if got := f.Filter("more"); got != "more" {
			t.Errorf("after the reset, Filter() = %q, want the text through", got)
		}
	})
}

// TestMarkdownFilterIsInterruptibleFilter checks the filter satisfies the
// interface the TTS base reaches for when it has to tell a filter that speech
// was cut off.
func TestMarkdownFilterIsInterruptibleFilter(t *testing.T) {
	var _ text.InterruptibleFilter = text.NewMarkdownFilter(text.DefaultMarkdownFilterOptions())
}
