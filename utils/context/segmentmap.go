package context

import (
	"regexp"
	"strings"
	"unicode"
)

// markupRE matches a well-formed markup tag, used to strip markup from a
// complete (non-truncated) text.
//
//nolint:gochecknoglobals // compiled once, read-only
var markupRE = regexp.MustCompile(`<[^>]+>`)

// stripCompleteMarkup removes well-formed <...> tags from a complete text. A
// lone '<' with no later '>' is left in place as real content.
func stripCompleteMarkup(text string) string {
	return markupRE.ReplaceAllString(text, "")
}

// iterCleanChars returns the (rune index, rune) pairs of text that lie outside
// markup. Markup is anything between '<' and '>'; an unclosed '<' swallows the
// rest of the string, matching how a streamed word-timestamp token can arrive
// mid-tag.
func iterCleanChars(runes []rune) []int {
	out := make([]int, 0, len(runes))
	inTag := false
	for i, r := range runes {
		switch {
		case inTag:
			if r == '>' {
				inTag = false
			}
		case r == '<':
			inTag = true
		default:
			out = append(out, i)
		}
	}
	return out
}

// stripMarkup removes markup from a possibly-truncated word-timestamp fragment.
// An unclosed '<' swallows the rest of text.
func stripMarkup(text string) string {
	runes := []rune(text)
	idx := iterCleanChars(runes)
	out := make([]rune, len(idx))
	for i, j := range idx {
		out[i] = runes[j]
	}
	return string(out)
}

// rawLenForCleanChars returns the raw rune offset into text just past its n-th
// markup-stripped character. It converts a match measured in markup-stripped
// space back to a raw offset. Returns the rune length of text when text has
// fewer than n non-markup chars.
func rawLenForCleanChars(text string, n int) int {
	if n <= 0 {
		return 0
	}
	runes := []rune(text)
	seen := 0
	for _, i := range iterCleanChars(runes) {
		seen++
		if seen == n {
			return i + 1
		}
	}
	return len(runes)
}

// textSegment is an immutable aligned chunk between original and TTS text.
// original is a chunk of the user-facing text; tts is the corresponding chunk in
// the transformed text sent to the synthesizer. originalStart/originalEnd are
// rune offsets into the original text.
type textSegment struct {
	origRunes []rune
	ttsRunes  []rune
	origStart int
	origEnd   int
}

func (s *textSegment) original() string { return string(s.origRunes) }
func (s *textSegment) tts() string      { return string(s.ttsRunes) }

// isTransformed reports whether the segment cannot be tracked by proportional
// character advancement: its alphanumeric content differs, a replacement changed
// the word count, or the TTS side carries markup.
func (s *textSegment) isTransformed() bool {
	ttsStr := s.tts()
	if ttsStr != stripCompleteMarkup(ttsStr) {
		return true
	}
	if normalize(s.original()) != normalize(ttsStr) {
		return true
	}
	return len(strings.Fields(s.original())) != len(strings.Fields(ttsStr))
}

func (s *textSegment) originalAlnumCount() int { return len([]rune(normalize(s.original()))) }

type hopKind int

const (
	hopPlaced    hopKind = iota // word fits within this segment; stop here
	hopCrosses                  // word runs past this segment; drain it, carry remainder
	hopExhausted                // no spoken content left here; drain it, keep the whole word
	hopNoMatch                  // word doesn't belong here; nudge past leading punctuation, stop
)

type hop struct {
	kind      hopKind
	segChars  int // raw runes consumed within this segment (PLACED / NO_MATCH nudge)
	wordChars int // runes trimmed off the front of the word (CROSSES)
}

type cand struct {
	text   string
	offset int
}

// TextSegmentMap maps cursor positions across three parallel texts as spoken
// words stream in: the transformed ttsText sent to the synthesizer, the
// user-facing originalText, and the llmText the model produced (which may add
// delimiters). It is built once by diffing ttsText against originalText into
// aligned segments; a single raw cursor over ttsText drives derived cursors over
// the original and LLM texts. Unchanged segments advance proportionally;
// transformed segments are held atomic and jump to the end of their original
// span once fully spoken.
type TextSegmentMap struct {
	ttsRunes  []rune
	origRunes []rune
	llmRunes  []rune
	segments  []textSegment

	segIdx        int
	segRawPos     int
	userFacingPos int
	llmPos        int
	lastCompleted *textSegment
	lastOverflow  string
}

// NewTextSegmentMap builds a segment map. ttsText is the post-transform text
// sent to the synthesizer, originalText is the user-facing pre-transform text,
// and llmText is the model-produced text (which may carry surrounding tags);
// pass "" for llmText to default it to originalText.
func NewTextSegmentMap(ttsText, originalText, llmText string) *TextSegmentMap {
	if llmText == "" {
		llmText = originalText
	}
	m := &TextSegmentMap{
		ttsRunes:  []rune(ttsText),
		origRunes: []rune(originalText),
		llmRunes:  []rune(llmText),
	}
	m.segments = buildSegments(ttsText, originalText)
	return m
}

// tokenizeWS splits text on runs of whitespace, keeping the separators, matching
// a capturing whitespace split so joining the tokens reconstructs text exactly.
func tokenizeWS(text string) []string {
	runes := []rune(text)
	var tokens []string
	start := 0
	i := 0
	for i < len(runes) {
		if unicode.IsSpace(runes[i]) {
			tokens = append(tokens, string(runes[start:i]))
			j := i
			for j < len(runes) && unicode.IsSpace(runes[j]) {
				j++
			}
			tokens = append(tokens, string(runes[i:j]))
			i = j
			start = j
		} else {
			i++
		}
	}
	tokens = append(tokens, string(runes[start:]))
	return tokens
}

// buildSegments builds aligned segments from a word-level diff. Each diff opcode
// (equal, replace, insert, delete) becomes a segment; segments whose normalized
// alphanumeric content differs are treated as transformed/atomic during cursor
// advancement.
func buildSegments(ttsText, originalText string) []textSegment {
	origTokens := tokenizeWS(originalText)
	ttsTokens := tokenizeWS(ttsText)

	var segments []textSegment
	origPos := 0
	for _, op := range getOpcodes(origTokens, ttsTokens) {
		origChunk := strings.Join(origTokens[op.i1:op.i2], "")
		ttsChunk := strings.Join(ttsTokens[op.j1:op.j2], "")
		origRunes := []rune(origChunk)
		origEnd := origPos + len(origRunes)
		segments = append(segments, textSegment{
			origRunes: origRunes,
			ttsRunes:  []rune(ttsChunk),
			origStart: origPos,
			origEnd:   origEnd,
		})
		origPos = origEnd
	}
	return segments
}

// leftTrimSpace returns runes with leading whitespace removed.
func leftTrimSpace(runes []rune) []rune {
	i := 0
	for i < len(runes) && unicode.IsSpace(runes[i]) {
		i++
	}
	return runes[i:]
}

// rtrimLen returns the rune length of runes after removing trailing whitespace.
func rtrimLen(runes []rune) int {
	i := len(runes)
	for i > 0 && unicode.IsSpace(runes[i-1]) {
		i--
	}
	return i
}

// runeSliceFrom returns s with its first k runes removed.
func runeSliceFrom(s string, k int) string {
	r := []rune(s)
	if k >= len(r) {
		return ""
	}
	return string(r[k:])
}

// literalHop tries a literal PLACED/CROSSES match of remainingWord against
// candidates, first as-is and then with the word's own trailing punctuation
// removed. When requireWordBoundary is set, a PLACED match is only accepted when
// it ends at a word boundary in the candidate.
func literalHop(candidates []cand, remainingWord string, requireWordBoundary bool) *hop {
	trimmed := stripTrailingPunctuation(remainingWord)
	words := []string{remainingWord}
	if trimmed != remainingWord {
		words = append(words, trimmed)
	}
	for _, word := range words {
		if word == "" {
			continue
		}
		wr := []rune(word)
		for _, c := range candidates {
			cr := []rune(c.text)
			switch {
			case strings.HasPrefix(c.text, word):
				landsMidWord := requireWordBoundary && len(wr) < len(cr) && isAlnum(cr[len(wr)])
				if !landsMidWord {
					return &hop{kind: hopPlaced, segChars: c.offset + len(wr)}
				}
			case c.text != "" && strings.HasPrefix(word, c.text):
				return &hop{kind: hopCrosses, wordChars: len(cr)}
			}
		}
	}
	return nil
}

// classifyHop decides where remainingWord goes against a segment's remaining raw
// text. It tries a literal match, then a case/accent-folded match, then a
// markup-stripped match; failing all three it drains an exhausted segment or
// nudges past leading punctuation.
func classifyHop(segmentRemaining, remainingWord string) hop {
	sr := []rune(segmentRemaining)
	strippedRunes := leftTrimSpace(sr)
	leadWs := len(sr) - len(strippedRunes)
	stripped := string(strippedRunes)

	candidates := []cand{{segmentRemaining, 0}}
	if leadWs > 0 {
		candidates = append(candidates, cand{stripped, leadWs})
	}
	if h := literalHop(candidates, remainingWord, false); h != nil {
		return *h
	}

	foldedWord := foldCaseAndAccents(remainingWord)
	folded := make([]cand, len(candidates))
	for i, c := range candidates {
		folded[i] = cand{foldCaseAndAccents(c.text), c.offset}
	}
	if h := literalHop(folded, foldedWord, true); h != nil {
		return *h
	}

	cleanWord := stripMarkup(remainingWord)
	if cleanWord != "" && strings.HasPrefix(stripMarkup(stripped), cleanWord) {
		rawLen := rawLenForCleanChars(stripped, len([]rune(cleanWord)))
		return hop{kind: hopPlaced, segChars: leadWs + rawLen}
	}

	if normalize(segmentRemaining) == "" {
		return hop{kind: hopExhausted}
	}

	nudge := 0
	for nudge < len(sr) && !isAlnum(sr[nudge]) {
		nudge++
	}
	return hop{kind: hopNoMatch, segChars: nudge}
}

// commitRawSpan advances the raw cursor to newPos within seg, moving the derived
// cursors. Unchanged segments advance proportionally by alphanumeric count;
// transformed segments hold the cursors until the segment fully completes, then
// jump to the end of its original span.
func (m *TextSegmentMap) commitRawSpan(seg *textSegment, newPos int) {
	if seg.isTransformed() {
		if normalize(string(seg.ttsRunes[newPos:])) == "" {
			newPos = len(seg.ttsRunes)
		}
	} else {
		nAlnum := len([]rune(normalize(string(seg.ttsRunes[m.segRawPos:newPos]))))
		if nAlnum > 0 {
			m.userFacingPos = advanceByAlnums(m.origRunes, m.userFacingPos, nAlnum)
		} else {
			m.userFacingPos = seg.origStart + rtrimLen(seg.ttsRunes[:newPos])
		}
		m.llmPos = advanceByAlnums(m.llmRunes, m.llmPos, nAlnum)
	}

	m.segRawPos = newPos

	if newPos >= len(seg.ttsRunes) {
		if seg.isTransformed() {
			m.userFacingPos = seg.origEnd
			m.llmPos = advanceByAlnums(m.llmRunes, m.llmPos, seg.originalAlnumCount())
		}
		m.lastCompleted = seg
		m.segIdx++
		m.segRawPos = 0
	}
}

// advanceRaw matches word against the remaining raw TTS text, hopping across
// segment boundaries as needed. A word running past the end of the TTS text is
// stored as overflow.
func (m *TextSegmentMap) advanceRaw(word string) {
	remaining := word
	for remaining != "" && m.segIdx < len(m.segments) {
		seg := &m.segments[m.segIdx]
		oldPos := m.segRawPos
		h := classifyHop(string(seg.ttsRunes[oldPos:]), remaining)

		switch h.kind {
		case hopNoMatch:
			m.segRawPos = oldPos + h.segChars
			return
		case hopPlaced:
			m.commitRawSpan(seg, oldPos+h.segChars)
			return
		default: // hopCrosses, hopExhausted
			m.commitRawSpan(seg, len(seg.ttsRunes))
			remaining = runeSliceFrom(remaining, h.wordChars)
		}
	}
	if remaining != "" {
		m.lastOverflow = remaining
	}
}

// AdvanceWord matches a raw word-timestamp token against the remaining TTS text
// and advances the cursors.
func (m *TextSegmentMap) AdvanceWord(word string) {
	m.lastCompleted = nil
	m.lastOverflow = ""
	if word != "" {
		m.advanceRaw(word)
	}
}

// WordBelongsCurrentSegment reports whether word plausibly continues the
// remaining TTS text. A false result signals that the synthesizer skipped a
// word-timestamp event and the caller should force-complete this slot.
func (m *TextSegmentMap) WordBelongsCurrentSegment(word string) bool {
	if word == "" {
		return true
	}
	if m.wordMatchesRemaining(word) {
		return true
	}
	if normalize(word) == "" {
		return m.symbolWordBelongs(word)
	}
	return false
}

// wordMatchesRemaining is a read-only replay of advanceRaw's segment walk.
func (m *TextSegmentMap) wordMatchesRemaining(word string) bool {
	if m.segIdx >= len(m.segments) {
		return false
	}
	segIdx := m.segIdx
	rawPos := m.segRawPos
	remaining := word
	for remaining != "" && segIdx < len(m.segments) {
		h := classifyHop(string(m.segments[segIdx].ttsRunes[rawPos:]), remaining)
		if h.kind == hopPlaced {
			return true
		}
		if h.kind == hopNoMatch {
			return false
		}
		remaining = runeSliceFrom(remaining, h.wordChars)
		segIdx++
		rawPos = 0
	}
	return true
}

// symbolWordBelongs reports whether a non-alphanumeric token (emoji,
// punctuation, symbol) belongs at the current position.
func (m *TextSegmentMap) symbolWordBelongs(word string) bool {
	pos := m.RawPos()
	searchStart := pos
	for searchStart > 0 {
		ch := m.ttsRunes[searchStart-1]
		if isAlnum(ch) || unicode.IsSpace(ch) || ch == '>' {
			break
		}
		searchStart--
	}
	if strings.Contains(string(m.ttsRunes[searchStart:]), word) {
		return true
	}
	if m.segIdx >= len(m.segments) {
		return false
	}
	p := pos
	for p < len(m.ttsRunes) && unicode.IsSpace(m.ttsRunes[p]) {
		p++
	}
	return p < len(m.ttsRunes) && !isAlnum(m.ttsRunes[p])
}

// UserFacingPos is the current rune offset into the original user-facing text.
func (m *TextSegmentMap) UserFacingPos() int { return m.userFacingPos }

// LLMPos is the current rune offset into the LLM text.
func (m *TextSegmentMap) LLMPos() int { return m.llmPos }

// RawPos is the current global rune offset into the TTS text.
func (m *TextSegmentMap) RawPos() int {
	pos := 0
	for i := 0; i < m.segIdx && i < len(m.segments); i++ {
		pos += len(m.segments[i].ttsRunes)
	}
	if m.segIdx < len(m.segments) {
		pos += m.segRawPos
	}
	return pos
}

// LastOverflow is the raw suffix of the last AdvanceWord that overflowed past
// the end of the TTS text, or "" when the last word fit.
func (m *TextSegmentMap) LastOverflow() string { return m.lastOverflow }

// IsComplete reports whether every segment's alphanumeric content has been
// accounted for. A frame whose remaining content is entirely punctuation/markup
// is already complete, except for whitespace-separated trailing punctuation that
// still arrives as its own token.
func (m *TextSegmentMap) IsComplete() bool {
	if m.segIdx >= len(m.segments) {
		return true
	}
	seg := &m.segments[m.segIdx]
	rem := string(seg.ttsRunes[m.segRawPos:])
	if normalize(rem) != "" {
		return false
	}
	if pendingSeparatedPunctuation(rem) {
		return false
	}
	for i := m.segIdx + 1; i < len(m.segments); i++ {
		if normalize(m.segments[i].tts()) != "" {
			return false
		}
	}
	return true
}

// pendingSeparatedPunctuation reports whether remaining is a whitespace-separated
// trailing punctuation token, which the synthesizer emits as its own event.
func pendingSeparatedPunctuation(remaining string) bool {
	sm := []rune(stripCompleteMarkup(remaining))
	if len(sm) == 0 || !unicode.IsSpace(sm[0]) {
		return false
	}
	content := strings.TrimSpace(string(sm))
	if content == "" {
		return false
	}
	return isPunct([]rune(content)[0])
}

// InTransformedSegment reports whether the cursor is partway through a
// transformed (atomic) segment.
func (m *TextSegmentMap) InTransformedSegment() bool {
	if m.segIdx >= len(m.segments) {
		return false
	}
	seg := &m.segments[m.segIdx]
	return seg.isTransformed() && m.segRawPos > 0
}

// LastCompletedSegment returns the original text of the segment completed by the
// last AdvanceWord call, and whether one completed.
func (m *TextSegmentMap) LastCompletedSegment() (original string, ok bool) {
	if m.lastCompleted == nil {
		return "", false
	}
	return m.lastCompleted.original(), true
}

// lastCompletedTransformed reports whether the last completed segment was a
// transformed one.
func (m *TextSegmentMap) lastCompletedTransformed() bool {
	return m.lastCompleted != nil && m.lastCompleted.isTransformed()
}

// hasLastCompleted reports whether the last AdvanceWord completed a segment.
func (m *TextSegmentMap) hasLastCompleted() bool { return m.lastCompleted != nil }

// Reset returns the map to its initial cursor state.
func (m *TextSegmentMap) Reset() {
	m.segIdx = 0
	m.segRawPos = 0
	m.userFacingPos = 0
	m.llmPos = 0
	m.lastCompleted = nil
	m.lastOverflow = ""
}
