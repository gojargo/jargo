package text

import (
	"bytes"
	"regexp"
	"strings"
	"unicode"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/renderer/html"
)

// MarkdownFilterOptions configures a MarkdownFilter. The zero value filters
// nothing at all (Enabled is false); use DefaultMarkdownFilterOptions for the
// recommended set and toggle individual fields from there.
type MarkdownFilterOptions struct {
	// Enabled turns the filter on. When false, Filter returns its input as
	// written, which is the switch to reach for to see what the model actually
	// produced.
	Enabled bool
	// FilterCode drops fenced code blocks instead of speaking their contents.
	// Code blocks span several chunks of streamed text, so the filter tracks
	// whether it is inside one across calls.
	FilterCode bool
	// FilterTables drops table contents. It also turns on table parsing, without
	// which a table is only ever a run of pipes and dashes.
	//
	// Pair it with FilterRepeatedSequences off. That rule runs first and flattens
	// a delimiter row of five or more dashes ("|--------|") to bare pipes, and a
	// table is not recognized without a well-formed delimiter row. A short row
	// ("|---|") is under the threshold and comes through either way.
	FilterTables bool
	// FilterRepeatedSequences drops a run of five or more of the same character,
	// the shape of an ASCII rule or a keyboard smash rather than of speech.
	FilterRepeatedSequences bool
}

// DefaultMarkdownFilterOptions returns the recommended set: filtering on, with
// repeated sequences dropped, and code and tables spoken rather than skipped.
func DefaultMarkdownFilterOptions() MarkdownFilterOptions {
	return MarkdownFilterOptions{
		Enabled:                 true,
		FilterRepeatedSequences: true,
	}
}

// MarkdownFilter converts Markdown to plain text for speech, preserving the
// structure around it: leading and trailing spaces survive, so text streamed
// word by word still joins up, and numbered list markers are spoken.
//
// It is stateful. Code blocks and tables arrive split across several calls, so
// the filter remembers that it is inside one; an interruption abandons whatever
// was half-read. Build one with NewMarkdownFilter.
type MarkdownFilter struct {
	opts MarkdownFilterOptions
	md   goldmark.Markdown

	inCodeBlock bool
	inTable     bool
	interrupted bool
}

// NewMarkdownFilter builds a MarkdownFilter from opts.
func NewMarkdownFilter(opts MarkdownFilterOptions) *MarkdownFilter {
	var exts []goldmark.Extender
	if opts.FilterTables {
		exts = append(exts, extension.Table)
	}
	// Raw HTML is rendered rather than escaped, so the tag stripping below sees
	// the tags the conversion produced and whatever the model wrote alike.
	md := goldmark.New(
		goldmark.WithExtensions(exts...),
		goldmark.WithRendererOptions(html.WithUnsafe()),
	)
	return &MarkdownFilter{opts: opts, md: md}
}

// HandleInterruption abandons any code block or table the filter was reading
// through, since the text that would have closed it is never coming.
func (f *MarkdownFilter) HandleInterruption() {
	f.interrupted = true
	f.inCodeBlock = false
	f.inTable = false
}

// ResetInterruption returns the filter to normal operation once the interruption
// has been dealt with.
func (f *MarkdownFilter) ResetInterruption() { f.interrupted = false }

//nolint:gochecknoglobals // compiled once, immutable
var (
	// The § markers below stand in for whitespace and list numbering while the
	// text goes through the Markdown conversion, which would otherwise collapse
	// them. They are put back at the end.
	mdBlankBeforeHeader = regexp.MustCompile(`\n\s*\n(#{1,6}[ \t])`)
	mdATXHeader         = regexp.MustCompile(`(?m)^#{1,6}[ \t]+(.*?)(?:[ \t]+#+[ \t]*)?$`)
	mdBlankLine         = regexp.MustCompile(`(?m)^\s*\n`)
	mdInlineSpan        = regexp.MustCompile("`[^`\n]+`")
	mdNumberedItem      = regexp.MustCompile(`^(\d+\.)\s`)
	mdEdgeSpace         = regexp.MustCompile(`(?m)^( +)|\s+$`)
	mdMarkedPipe        = regexp.MustCompile(`§\| `)
	mdHTMLTag           = regexp.MustCompile(`<[^<]+?>`)
	mdDoubleStar        = regexp.MustCompile(`\*\*`)
	mdEdgeStar          = regexp.MustCompile(`(^|\s)\*|\*($|\s)`)
	mdPipe              = regexp.MustCompile(`\|`)
	mdSeparatorRow      = regexp.MustCompile(`(?m)^\s*[-:]+\s*$`)
	mdMarker            = regexp.MustCompile(`§`)
	mdScheme            = regexp.MustCompile(`https?://`)
	mdFence             = regexp.MustCompile("```")
	mdTable             = regexp.MustCompile(`(?is)<table>.*?</table>`)
	mdTableStart        = regexp.MustCompile(`(?is)<table>.*`)
	mdTableEnd          = regexp.MustCompile(`(?is).*</table>`)
)

// Filter converts one chunk of Markdown into the plain text to speak.
func (f *MarkdownFilter) Filter(text string) string {
	if !f.opts.Enabled {
		return text
	}

	// Headers are flattened here rather than left to the conversion: the blank
	// line collapse below puts a space in front of them, and a header is no
	// longer a header once it is indented. Drop the blank line before a header
	// first, keeping the marker, so the collapse has nothing to work on.
	out := mdBlankBeforeHeader.ReplaceAllString(text, "\n$1")
	// Then take off the leading marker and any closing one, leaving the header
	// text. The spaces around a closing marker go with it, so genuine trailing
	// whitespace survives on an open header and word-by-word streaming still
	// joins up.
	out = mdATXHeader.ReplaceAllString(out, "$1")

	// A newline becomes a space only where there is no text either side of it.
	out = mdBlankLine.ReplaceAllString(out, " ")

	out = stripInlineCode(out)

	if f.opts.FilterRepeatedSequences {
		out = dropRepeatedRuns(out)
	}

	// Numbered list items keep their number, behind a marker that survives the
	// conversion.
	out = mdNumberedItem.ReplaceAllString(out, "§NUM§$1 ")

	// Leading and trailing spaces go behind markers too, one per character, which
	// is what keeps word-by-word streaming readable.
	out = mdEdgeSpace.ReplaceAllStringFunc(out, func(m string) string {
		return strings.Repeat("§", len(m))
	})

	// A marker in front of a table row would stop the row being recognized.
	out = mdMarkedPipe.ReplaceAllString(out, "| ")

	var buf bytes.Buffer
	if err := f.md.Convert([]byte(out), &buf); err != nil {
		// The renderer only fails on a write error, which a bytes.Buffer does not
		// produce. Speak the text as it stands rather than dropping it.
		return text
	}
	// The renderer ends its output with a newline; nothing downstream expects one.
	out = strings.TrimSuffix(buf.String(), "\n")

	if f.opts.FilterTables {
		out = f.removeTables(out)
	}

	out = mdHTMLTag.ReplaceAllString(out, "")

	// Entities become the characters they name. The renderer resolves a
	// non-breaking space itself rather than passing the entity through, so the
	// character is mapped alongside the entity; a synthesizer given a literal
	// U+00A0 reads it as part of the word either side of it.
	out = strings.NewReplacer(
		"&nbsp;", " ",
		" ", " ",
		"&lt;", "<",
		"&gt;", ">",
		"&amp;", "&",
	).Replace(out)

	// Emphasis markers the conversion left behind, either side of a word.
	out = mdDoubleStar.ReplaceAllString(out, "")
	out = mdEdgeStar.ReplaceAllString(out, "${1}${2}")

	// Table pipes and separator rows, for the tables the conversion did not take.
	out = mdPipe.ReplaceAllString(out, "")
	out = mdSeparatorRow.ReplaceAllString(out, "")

	if f.opts.FilterCode {
		out = f.removeCodeBlocks(out)
	}

	out = strings.ReplaceAll(out, "§NUM§", "")
	out = mdMarker.ReplaceAllString(out, " ")

	// A scheme is spelled out letter by letter otherwise, and says nothing.
	return mdScheme.ReplaceAllString(out, "")
}

// stripInlineCode takes the backticks off an inline code span, leaving what is
// inside. A backtick on either side means a fence rather than a span, so the
// scan steps past it and carries on, which is how a doubled backtick is left
// alone.
func stripInlineCode(text string) string {
	var b strings.Builder
	pos := 0
	for pos < len(text) {
		loc := mdInlineSpan.FindStringIndex(text[pos:])
		if loc == nil {
			break
		}
		start, end := pos+loc[0], pos+loc[1]
		if (start > 0 && text[start-1] == '`') || (end < len(text) && text[end] == '`') {
			b.WriteString(text[pos : start+1])
			pos = start + 1
			continue
		}
		b.WriteString(text[pos:start])
		b.WriteString(text[start+1 : end-1])
		pos = end
	}
	b.WriteString(text[pos:])
	return b.String()
}

// dropRepeatedRuns removes a run of five or more of the same non-whitespace
// character. RE2 has no backreferences, so the run is found by scanning.
func dropRepeatedRuns(text string) string {
	var b strings.Builder
	runes := []rune(text)
	for i := 0; i < len(runes); {
		j := i + 1
		for j < len(runes) && runes[j] == runes[i] {
			j++
		}
		if j-i >= 5 && !unicode.IsSpace(runes[i]) {
			i = j
			continue
		}
		b.WriteString(string(runes[i:j]))
		i = j
	}
	return b.String()
}

// removeCodeBlocks drops fenced code, carrying the "inside a block" state across
// calls because a block rarely arrives whole.
func (f *MarkdownFilter) removeCodeBlocks(text string) string {
	if f.interrupted {
		f.inCodeBlock = false
		return text
	}

	loc := mdFence.FindStringIndex(text)

	if f.inCodeBlock {
		// Inside a block: text after a closing fence resumes, otherwise it is all
		// still code.
		if loc != nil {
			f.inCodeBlock = false
			return strings.TrimSpace(text[loc[1]:])
		}
		return ""
	}

	if loc == nil {
		return text
	}
	if loc[0] == 0 || strings.TrimSpace(text[:loc[0]]) == "" {
		// A block starts here; keep whatever came before it.
		f.inCodeBlock = true
		return strings.TrimSpace(text[:loc[0]])
	}
	// A fence in the middle of a line: a whole block is cut out, and a lone
	// opening one takes the rest with it.
	parts := mdFence.Split(text, -1)
	if len(parts) > 2 {
		return strings.TrimSpace(parts[0] + " " + parts[len(parts)-1])
	}
	f.inCodeBlock = true
	return strings.TrimSpace(parts[0])
}

// removeTables drops table markup, carrying the "inside a table" state across
// calls for the same reason code blocks need it.
func (f *MarkdownFilter) removeTables(text string) string {
	if f.interrupted {
		f.inTable = false
		return text
	}

	text = mdTable.ReplaceAllString(text, "")

	if f.inTable {
		if loc := mdTableEnd.FindStringIndex(text); loc != nil && loc[0] == 0 {
			f.inTable = false
			return strings.TrimSpace(text[loc[1]:])
		}
		return ""
	}

	if loc := mdTableStart.FindStringIndex(text); loc != nil {
		f.inTable = true
		return strings.TrimSpace(text[:loc[0]])
	}

	return strings.TrimSpace(text)
}
