package kokoro

import (
	"strings"
	"unicode"
)

// The punctuation marks considered by default:
// ;:,.!?¡¿ then the em dash, the ellipsis, "«»“” and the brackets.
const defaultPunctuationMarks = ";:,.!?\u00a1\u00bf\u2014\u2026\"\u00ab\u00bb\u201c\u201d(){}[]"

// decimalSeparators are the marks also used as a decimal separator, which are
// not punctuation when they sit between two digits.
const decimalSeparators = ",."

// inlineMarks are the marks that do not break an utterance: the words on both
// sides are pronounced together, as without punctuation. The other marks end
// an utterance. They are quotes and brackets: "«»“”‘’„‹›()[]{}.
//
// This comes from the behavior of espeak-ng 1.52 rather than its
// documentation: the marks ;:,.!?¡¿ and the em dash and ellipsis end a clause,
// phonemized apart, while quotes and brackets only insert a short pause.
const inlineMarks = "\"\u00ab\u00bb\u201c\u201d\u2018\u2019\u201e\u2039\u203a()[]{}"

// separator is the strings put between phonemes, syllables and words of a
// phonemized text.
type separator struct {
	word     string
	syllable string
	phone    string
}

// defaultSeparator separates the words with a space, and nothing else.
//
//nolint:gochecknoglobals // a read-only value
var defaultSeparator = separator{word: " "}

// linePart is one part of a line split at its punctuation: the index of a
// chunk of text, or a mark.
type linePart struct {
	chunk  int
	mark   string
	isMark bool
}

// contextGroup is a group of chunks separated by inline marks only, with the
// group phonemized as a whole.
type contextGroup struct {
	chunks     []int
	phonemized string
}

// punctuation hides the punctuation from the phonemization backend and
// restores it afterwards, or removes it. espeak ignores punctuation and drops
// it silently, so it is cut out of the text before phonemization and put back
// into the phonemes.
type punctuation struct {
	marks string
}

// newPunctuation considers the marks given, each a single character.
func newPunctuation(marks string) *punctuation {
	return &punctuation{marks: marks}
}

// isMarkAt reports whether the rune at index i of line is a mark. A mark that
// doubles as a decimal separator is not one when it sits between two digits,
// so a decimal number reaches the backend intact: split, "1,5" would be read
// as two separate numbers, which changes the value spoken.
func (p *punctuation) isMarkAt(line []rune, i int) bool {
	r := line[i]
	if !strings.ContainsRune(p.marks, r) {
		return false
	}
	if !strings.ContainsRune(decimalSeparators, r) {
		return true
	}
	afterDigit := i > 0 && isASCIIDigit(line[i-1])
	beforeDigit := i+1 < len(line) && isASCIIDigit(line[i+1])
	return !afterDigit || !beforeDigit
}

// isASCIIDigit reports whether r is 0 to 9.
func isASCIIDigit(r rune) bool { return r >= '0' && r <= '9' }

// isSpace reports whether r is whitespace the way str.isspace and \s read it,
// which includes the information separators U+001C to U+001F.
func isSpace(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) }

// trimSpace strips the whitespace isSpace reads from both ends of s.
func trimSpace(s string) string { return strings.TrimFunc(s, isSpace) }

// fields splits s around each run of the whitespace isSpace reads.
func fields(s string) []string { return strings.FieldsFunc(s, isSpace) }

// markSpans finds the runs of marks in line: each is zero or more spaces, then
// one or more marks and zero or more spaces, repeated as long as marks follow.
// The spans are rune offsets, start inclusive and end exclusive.
func (p *punctuation) markSpans(line []rune) [][2]int {
	var spans [][2]int
	i := 0
	for i < len(line) {
		end := p.matchAt(line, i)
		if end < 0 {
			i++
			continue
		}
		spans = append(spans, [2]int{i, end})
		if end == i {
			i++
		} else {
			i = end
		}
	}
	return spans
}

// matchAt is the end of the run of marks starting at start, or -1 when none
// starts there. A run needs a mark after its leading spaces.
func (p *punctuation) matchAt(line []rune, start int) int {
	end := -1
	i := start
	for {
		j := i
		for j < len(line) && isSpace(line[j]) {
			j++
		}
		k := j
		for k < len(line) && p.isMarkAt(line, k) {
			k++
		}
		if k == j {
			return end
		}
		for k < len(line) && isSpace(line[k]) {
			k++
		}
		end = k
		i = k
	}
}

// remove is the text with all its punctuation replaced by spaces, and trimmed.
func (p *punctuation) remove(text []string) []string {
	out := make([]string, len(text))
	for n, line := range text {
		runes := []rune(line)
		var b strings.Builder
		cursor := 0
		for _, span := range p.markSpans(runes) {
			b.WriteString(string(runes[cursor:span[0]]))
			b.WriteString(" ")
			cursor = span[1]
		}
		b.WriteString(string(runes[cursor:]))
		out[n] = trimSpace(b.String())
	}
	return out
}

// isInline reports whether mark does not break the utterance: quotes and
// brackets, whitespace aside. The other marks, and marks not known to be
// inline, end an utterance.
func isInline(mark string) bool {
	mark = strings.Join(fields(mark), "")
	if mark == "" {
		return false
	}
	for _, r := range mark {
		if !strings.ContainsRune(inlineMarks, r) {
			return false
		}
	}
	return true
}

// preserve removes the punctuation from text, allowing for its restoration.
// It returns the chunks of text between the marks, to be phonemized, and for
// each line its parts in order, a chunk index or a mark:
//
//	["hello, my world!", "ok"] -> ["hello", "my world", "ok"],
//		[[0, ", ", 1, "!"], [2]]
func (p *punctuation) preserve(text []string) ([]string, [][]linePart) {
	var chunks []string
	lines := make([][]linePart, 0, len(text))
	for _, line := range text {
		runes := []rune(line)
		var parts []linePart
		cursor := 0
		for _, span := range p.markSpans(runes) {
			if span[0] > cursor {
				parts = append(parts, linePart{chunk: len(chunks)})
				chunks = append(chunks, string(runes[cursor:span[0]]))
			}
			parts = append(parts, linePart{mark: string(runes[span[0]:span[1]]), isMark: true})
			cursor = span[1]
		}
		if cursor < len(runes) {
			parts = append(parts, linePart{chunk: len(chunks)})
			chunks = append(chunks, string(runes[cursor:]))
		}
		lines = append(lines, parts)
	}
	return chunks, lines
}

// inlineGroups lists the groups of consecutive chunks of a line separated by
// inline marks only. The backend phonemizes such a group as a whole as well,
// so that its words are pronounced in context.
func inlineGroups(lines [][]linePart) [][]int {
	var groups [][]int
	for _, parts := range lines {
		var group []int
		for num, part := range parts {
			if part.isMark {
				continue
			}
			if len(group) > 0 && num >= 2 && !parts[num-2].isMark && parts[num-2].chunk == group[len(group)-1] &&
				parts[num-1].isMark && isInline(parts[num-1].mark) {
				group = append(group, part.chunk)
			} else {
				if len(group) > 1 {
					groups = append(groups, group)
				}
				group = []int{part.chunk}
			}
		}
		if len(group) > 1 {
			groups = append(groups, group)
		}
	}
	return groups
}

// canAlign reports whether inline marks can be placed in context with sep:
// they are placed by counting the phonemized words, split on the word
// separator, which must not be empty nor found in the other separators.
func canAlign(sep separator) bool {
	return sep.word != "" && !strings.Contains(sep.phone, sep.word) && !strings.Contains(sep.syllable, sep.word)
}

// separatorWords is the words of a phonemized text, split on the word
// separator.
func separatorWords(text string, sep separator) []string {
	var words []string
	for w := range strings.SplitSeq(text, sep.word) {
		if w != "" {
			words = append(words, w)
		}
	}
	return words
}

// splitContext cuts a group phonemized as a whole at its inline marks, into
// one part per chunk, or reports false when the marks cannot be placed for
// sure: the words phonemized in context must be as many as the words of each
// chunk phonemized alone.
func splitContext(text []string, group []int, phonemized string, sep separator) ([]string, bool) {
	if !canAlign(sep) {
		return nil, false
	}
	counts := make([]int, len(group))
	total := 0
	for i, chunk := range group {
		counts[i] = len(separatorWords(text[chunk], sep))
		if counts[i] == 0 {
			return nil, false
		}
		total += counts[i]
	}
	if len(separatorWords(phonemized, sep)) != total {
		return nil, false
	}

	// Cut before the first word of each chunk but the first one.
	starts := map[int]bool{}
	acc := 0
	for _, c := range counts[:len(counts)-1] {
		acc += c
		starts[acc] = true
	}
	pieces := strings.Split(phonemized, sep.word)
	cuts := []int{0}
	nwords := 0
	for num, piece := range pieces {
		if piece != "" {
			if starts[nwords] {
				cuts = append(cuts, num)
			}
			nwords++
		}
	}
	cuts = append(cuts, len(pieces))
	parts := make([]string, 0, len(cuts)-1)
	for i := range len(cuts) - 1 {
		parts = append(parts, strings.Join(pieces[cuts[i]:cuts[i+1]], sep.word))
	}
	return parts, true
}

// restore puts the punctuation back into the phonemized chunks: the reverse
// of preserve. The internal spaces of the marks are replaced by the word
// separator. context gives the groups of chunks inlineGroups found, each
// phonemized as a whole; they replace the chunks phonemized alone when their
// marks can be placed.
func restore(text []string, lines [][]linePart, sep separator, strip bool, context []contextGroup) []string {
	inContext := placeContext(text, lines, sep, context)
	out := make([]string, 0, len(lines))
	for _, parts := range lines {
		out = append(out, restoreLine(text, parts, sep, strip, inContext))
	}
	return out
}

// placeContext is the chunks of the groups phonemized in context, by chunk,
// for the groups whose marks can be placed: chunks separated by a single
// inline mark each, whose words align with the chunks phonemized alone.
func placeContext(text []string, lines [][]linePart, sep separator, context []contextGroup) map[int]string {
	type place struct{ line, pos int }
	position := map[int]place{}
	for num, parts := range lines {
		for pos, part := range parts {
			if !part.isMark {
				position[part.chunk] = place{num, pos}
			}
		}
	}
	isGroup := func(chunks []int) bool {
		for i, chunk := range chunks {
			second, found := position[chunk]
			if !found {
				return false
			}
			if i == 0 {
				continue
			}
			first := position[chunks[i-1]]
			if second != (place{first.line, first.pos + 2}) {
				return false
			}
			if between := lines[first.line][first.pos+1]; !between.isMark || !isInline(between.mark) {
				return false
			}
		}
		return true
	}
	inContext := map[int]string{}
	for _, g := range context {
		if !isGroup(g.chunks) {
			continue
		}
		if parts, aligned := splitContext(text, g.chunks, g.phonemized, sep); aligned {
			for i, chunk := range g.chunks {
				inContext[chunk] = parts[i]
			}
		}
	}
	return inContext
}

// restoreLine is one line put back together from its parts.
func restoreLine(text []string, parts []linePart, sep separator, strip bool, inContext map[int]string) string {
	punctuated := ""
	mark, hasMark := "", false
	for _, part := range parts {
		if part.isMark {
			// Remove the last word separator before the mark, and replace its
			// internal spaces with the word separator.
			if sep.word != "" && strings.HasSuffix(punctuated, sep.word) {
				punctuated = punctuated[:len(punctuated)-len(sep.word)]
			}
			mark, hasMark = strings.ReplaceAll(part.mark, " ", sep.word), true
			punctuated += mark
			continue
		}
		chunk, ok := inContext[part.chunk]
		if !ok {
			chunk = text[part.chunk]
		}
		// A chunk phonemized as nothing does not follow the mark.
		if chunk != "" {
			punctuated += chunk
			mark, hasMark = "", false
		}
	}
	// A final mark is followed by a word separator, if not stripped.
	if hasMark && !strip && !strings.HasSuffix(mark, sep.word) {
		punctuated += sep.word
	}
	return punctuated
}
