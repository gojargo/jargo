package context

import (
	"strings"
	"unicode"
)

// textSegment is a piece of the utterance, paired with what the synthesizer was
// given in its place. The map is a list of these laid end to end over the whole
// utterance. In most of them the two sides are identical; the interesting ones
// are where a transform, a filter or a tag made them differ (see isTransformed).
//
// origStart and origEnd are rune offsets into the full original text. Cursors
// jump straight to origEnd once a rewritten piece is finished, since no position
// inside one means anything.
type textSegment struct {
	origRunes []rune
	ttsRunes  []rune
	origStart int
	origEnd   int
}

func (s *textSegment) original() string { return string(s.origRunes) }
func (s *textSegment) tts() string      { return string(s.ttsRunes) }

// isTransformed reports whether the two sides cannot be followed together, rune
// by rune. A segment like this is all or nothing: the cursors into the other
// texts wait at its start until every spoken word of it has arrived, then jump
// straight to its end.
//
// Any one of these makes it true: the letters and digits differ, as in "$42.50"
// against "forty two dollars"; the two sides have different numbers of words; or
// the TTS side has tags in it, even when the words match, because the cursor
// through the spoken text has tag runes to cross that the other texts do not.
//
// Only the shape of a tag matters, never its name.
func (s *textSegment) isTransformed() bool {
	ttsStr := s.tts()
	if ttsStr != stripCompleteMarkup(ttsStr) {
		return true
	}
	if alnumOnly(s.original()) != alnumOnly(ttsStr) {
		return true
	}
	return len(strings.Fields(s.original())) != len(strings.Fields(ttsStr))
}

// originalAlnumCount is how many letters and digits the original side of this
// segment has. It is what the cursors into the original and LLM texts spend,
// since those texts hold the original characters, not the spoken ones.
func (s *textSegment) originalAlnumCount() int { return len([]rune(alnumOnly(s.original()))) }

// hopKind is what happened when a word was offered to one segment. Trying one
// word against one segment is called a hop. Two of these answers end the
// attempt; the other two send the word on to the next segment.
type hopKind int

const (
	// hopPlaced means the word fits here: move to the end of the match and stop.
	hopPlaced hopKind = iota
	// hopCrosses means this segment holds only the beginning of the word: finish
	// the segment and take the rest of the word to the next one.
	hopCrosses
	// hopExhausted means there is nothing here that can be spoken, so no word
	// will ever match, an empty side of the diff or a lone <break/>: finish the
	// segment and try the whole word again on the next one.
	hopExhausted
	// hopNoMatch means the word does not belong here: step past punctuation at
	// the start of the segment and stop, without moving the cursors into the
	// other two texts.
	hopNoMatch
)

// hop is the outcome of offering one word to one segment. It is produced by
// classifyHop, collected by planHops and acted on by consumeWord.
type hop struct {
	kind hopKind
	// segChars is how many runes to move forward in this segment: the length of
	// the match for hopPlaced, or a step past leading punctuation for hopNoMatch.
	// The other two leave it at 0 because they finish the whole segment anyway.
	segChars int
	// wordChars is how many runes of the word this segment used up, and so how
	// many to drop before offering the rest to the next segment. Only hopCrosses
	// sets it; hopExhausted passes the word on whole.
	wordChars int
}

// cand is one starting point a word may be matched against, paired with how far
// into the segment's remaining text it starts.
type cand struct {
	text   string
	offset int
}

// TextSegmentMap answers "where are we?" in three versions of one utterance,
// word by word.
//
// A synthesizer reports the words it speaks. Each report has to be turned into a
// position, but into a position in three different strings, because the same
// utterance exists in three forms at once:
//
//   - ttsText, what was actually spoken, tags and all:
//     "Your balance is forty two dollars"
//   - originalText, what a client displays: "Your balance is $42.50"
//   - llmText, what the model wrote, so what the transcript should keep:
//     "Your balance is <b>$42.50</b>". Defaults to originalText.
//
// For a frame nothing rewrote, all three are the same string and every position
// is the same.
//
// The hard part is that a spoken word need not appear in the other two. The
// synthesizer says "dollars"; nothing in "$42.50" matches it. So the map is
// built once, by diffing ttsText against originalText into aligned segments,
// each either survived unchanged or rewritten whole.
//
// llmText is never compared against the others, and does not need to be. It
// holds the same letters and digits as originalText, in the same order, and
// differs only in what is wrapped around them: tags, delimiters, punctuation. So
// counting letters and digits is enough to keep it in step, and its cursor moves
// by that count.
//
// From then on one real cursor moves: RawPos, how far into ttsText the
// synthesizer has got. UserFacingPos and LLMPos follow it. Through an unchanged
// segment they keep pace, word for word. Through a rewritten one they wait:
// there is no honest position halfway through "$42.50" while "forty two dollars"
// is being spoken, so they hold and then jump to the end of the span in one step
// when the last of its words lands.
//
// Callers ask two things. WordBelongsCurrentSegment, does this token plausibly
// continue what is left to speak, and AdvanceWord, which consumes it. Both
// tolerate the ways synthesizers mangle tokens (added punctuation, changed case
// or diacritics, a fragment of a half-open tag) without the caller knowing
// anything about it; classifyHop holds that logic.
type TextSegmentMap struct {
	ttsRunes  []rune
	origRunes []rune
	llmRunes  []rune
	segments  []textSegment

	segIdx               int
	segRawPos            int
	userFacingPos        int
	llmPos               int
	lastCompleted        *textSegment
	lastOverflow         string
	lastLeadingDuplicate int
}

// NewTextSegmentMap lines the three texts up against each other. The comparison
// happens once, here; everything after this only moves cursors.
//
// ttsText is what was sent to the synthesizer, and so what incoming words are
// matched against; it may carry synthesis tags and rewritten values.
// originalText is the same content as a client displays it, before any
// rewriting, and is diffed against ttsText to build the segments. llmText is the
// same content as the model wrote it, which may add delimiters the other two
// never see; it rides its own cursor rather than being diffed. Pass "" for
// llmText to default it to originalText.
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

// buildSegments compares the two texts and cuts them into the segments the map
// walks. The diff lines them up a word at a time and reports each piece as
// equal, replaced, inserted or deleted, and every piece becomes one segment.
// Whitespace is kept as words of its own so that the positions stay exact.
//
// One extra step: a piece that came out equal is cut around any tag inside it,
// so a single tag does not turn the whole sentence into an all-or-nothing
// segment. Only equal pieces can be split this way. Both sides hold the same
// text there, so a single position cuts both; where the sides differ there is no
// position that means the same thing on each.
func buildSegments(ttsText, originalText string) []textSegment {
	origTokens := tokenizeWS(originalText)
	ttsTokens := tokenizeWS(ttsText)

	var segments []textSegment
	origPos := 0
	for _, op := range getOpcodes(origTokens, ttsTokens) {
		origChunk := strings.Join(origTokens[op.i1:op.i2], "")
		ttsChunk := strings.Join(ttsTokens[op.j1:op.j2], "")

		type part struct{ orig, tts string }
		var parts []part
		if op.tag == "equal" {
			for _, run := range splitMarkupRuns(origChunk) {
				parts = append(parts, part{run, run})
			}
		} else {
			parts = []part{{origChunk, ttsChunk}}
		}

		for _, p := range parts {
			origRunes := []rune(p.orig)
			origEnd := origPos + len(origRunes)
			segments = append(segments, textSegment{
				origRunes: origRunes,
				ttsRunes:  []rune(p.tts),
				origStart: origPos,
				origEnd:   origEnd,
			})
			origPos = origEnd
		}
	}
	return segments
}

// wordVariants returns word, then word with any punctuation at its end removed.
// A synthesizer can add punctuation the text it was given never had, reading a
// list item "my account" as a sentence and reporting "account.". Matching tries
// the word as it arrived first, then the trimmed form.
func wordVariants(word string) []string {
	trimmed := stripTrailingPunctuation(word)
	if trimmed == word {
		return []string{word}
	}
	return []string{word, trimmed}
}

// literalHop compares remainingWord to each candidate, rune for rune.
//
// Two things can match. If a candidate starts with the word, the word fits here
// (hopPlaced). If the word starts with a candidate, the candidate ran out first,
// so the rest of the word belongs to the next segment (hopCrosses).
//
// foldedHop calls this too, on folded copies of the same strings. Folding never
// changes a string's rune length, so a position found in a folded copy is the
// same position in the original.
//
// When requireWordBoundary is set, the word must end where a word ends in the
// candidate: either it used the candidate up, or the next rune is not a letter
// or digit. That stops "account" from matching the start of "Accountant". Only
// the folded pass asks for it, because folding away case makes that kind of
// accidental match much easier to hit.
func literalHop(candidates []cand, remainingWord string, requireWordBoundary bool) *hop {
	for _, word := range wordVariants(remainingWord) {
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

// leadingNonAlnumLen counts the runes at the start of text that are not letters
// or digits. For ", I can" it returns 2, the comma and the space.
//
// With stopAtMarkup, counting also stops at a '<', so the count never reaches
// inside a tag. Without it, "<break/>hello" would count past the '<', and a
// synthesizer that reports the tag's name "break" as a word would look like it
// had spoken it.
func leadingNonAlnumLen(text string, stopAtMarkup bool) int {
	runes := []rune(text)
	i := 0
	for i < len(runes) && !isAlnum(runes[i]) {
		if stopAtMarkup && runes[i] == '<' {
			break
		}
		i++
	}
	return i
}

// matchCandidates returns the three starting points a word may be matched
// against here. Synthesizers disagree about what they include in a word, so the
// same text is offered from three places, each paired with how far in it starts:
//
//  1. The text as it is, for one that reports " world" with its own leading
//     space.
//  2. Past any spaces.
//  3. Past everything that is not a letter or digit, for one that does not
//     repeat punctuation it already spoke: when "I" arrives for "Yeah, I can",
//     the ", " is still waiting here.
//
// Each start is further in than the last, so the closest match is tried first.
// Identical starting points are dropped.
func matchCandidates(segmentRemaining string) []cand {
	runes := []rune(segmentRemaining)
	leadWs := len(runes) - len(leftTrimSpace(runes))
	leadNonAlnum := leadingNonAlnumLen(segmentRemaining, true)

	candidates := make([]cand, 0, 3)
	seen := make(map[int]struct{}, 3)
	for _, offset := range [3]int{0, leadWs, leadNonAlnum} {
		if _, dup := seen[offset]; dup {
			continue
		}
		seen[offset] = struct{}{}
		candidates = append(candidates, cand{string(runes[offset:]), offset})
	}
	return candidates
}

// foldedHop matches remainingWord again, ignoring differences in how it is
// written. A synthesizer may report a word in lower case, without accents, or
// with plain quotes: "SQL" as "sql", "café" as "cafe", "don’t" as "don't".
// Folding both sides makes those the same.
//
// Folding swaps runes one for one and never adds or removes any, so the strings
// keep their rune length and a position found here means the same position in
// the original text.
//
// Because folding hides case, a short word could now match inside a longer one,
// "account" inside "Accountant", so a match here is only accepted if it ends
// where a word ends.
func foldedHop(candidates []cand, remainingWord string) *hop {
	folded := make([]cand, len(candidates))
	for i, c := range candidates {
		folded[i] = cand{foldForMatching(c.text), c.offset}
	}
	return literalHop(folded, foldForMatching(remainingWord), true)
}

// markupHop matches remainingWord again, with tags removed from both sides. A
// synthesizer may report a word wrapped in tags the text it was given did not
// have, or the other way round; removing tags from both sides lets the words
// themselves be compared.
//
// The match is found in text that has no tags, so its position has to be
// translated back to a position in the real text, tags included, by
// rawLenForCleanChars.
//
// Only hopPlaced comes out of this. A word that runs past the end of the segment
// is left to the two earlier passes.
func markupHop(segmentRemaining, remainingWord string) *hop {
	runes := []rune(segmentRemaining)
	strippedRunes := leftTrimSpace(runes)
	leadWs := len(runes) - len(strippedRunes)
	stripped := string(strippedRunes)
	haystack := stripMarkup(stripped)

	for _, candidate := range wordVariants(stripMarkup(remainingWord)) {
		if candidate != "" && strings.HasPrefix(haystack, candidate) {
			rawLen := rawLenForCleanChars(stripped, len([]rune(candidate)))
			return &hop{kind: hopPlaced, segChars: leadWs + rawLen}
		}
	}
	return nil
}

// classifyHop decides what remainingWord does to the text left in this segment.
// Everything here is plain string comparison: no tag names are understood, and
// nothing is remembered between calls.
//
// Three ways of matching are tried, each more forgiving than the one before:
// literalHop, then foldedHop, then markupHop. Any of them can report that the
// word fits here (hopPlaced) or that it runs past the end of the segment
// (hopCrosses).
//
// If none of them match, the answer depends on what is left in the segment.
// hopExhausted when nothing here can be spoken, only a tag such as <break/>, or
// trailing spaces and punctuation: the segment is finished so the word can try
// the next one. That is checked last, so a word that really does match something
// like a trailing emoji is found by the passes above first. hopNoMatch
// otherwise: the word belongs somewhere else, so the cursor only steps past
// punctuation at the start of the segment, never past anything that was actually
// spoken.
func classifyHop(segmentRemaining, remainingWord string) hop {
	candidates := matchCandidates(segmentRemaining)

	h := literalHop(candidates, remainingWord, false)
	if h == nil {
		h = foldedHop(candidates, remainingWord)
	}
	if h == nil {
		h = markupHop(segmentRemaining, remainingWord)
	}
	if h != nil {
		return *h
	}

	// Nothing left here that can be spoken: finish the segment so the word can
	// try the next one.
	if !hasAlnum(segmentRemaining) {
		return hop{kind: hopExhausted}
	}

	// Foreign token: nudge past leading punctuation only, then stop. Unlike the
	// skip candidates this does not stop at markup. It moves the raw cursor
	// rather than deciding a match, so there is no tag name it could mistake for
	// spoken content.
	return hop{kind: hopNoMatch, segChars: leadingNonAlnumLen(segmentRemaining, false)}
}

// advanceCursorsTo moves every cursor to newPos within seg, and finishes seg if
// it is reached. This is where the keep-pace-or-wait rule from the type comment
// is applied, and the only place the cursors into the other two texts move.
func (m *TextSegmentMap) advanceCursorsTo(seg *textSegment, newPos int) {
	if seg.isTransformed() {
		// Whatever is left is only a closing tag or the like, which no word event
		// will ever name. Take it now so the segment can finish. Unchanged
		// segments are not given this: a trailing emoji there is real output, and
		// its own event is still coming.
		if !hasAlnum(string(seg.ttsRunes[newPos:])) {
			newPos = len(seg.ttsRunes)
		}
	} else {
		m.keepDerivedCursorsInPace(seg, newPos)
	}

	m.segRawPos = newPos

	if newPos >= len(seg.ttsRunes) {
		if seg.isTransformed() {
			m.commitTransformedSpan(seg)
		}
		m.finishSegment(seg)
	}
}

// keepDerivedCursorsInPace moves the cursors into the other two texts by what
// this step just spoke. The count of letters and digits consumed here is what
// they move by.
func (m *TextSegmentMap) keepDerivedCursorsInPace(seg *textSegment, newPos int) {
	nAlnum := len([]rune(alnumOnly(string(seg.ttsRunes[m.segRawPos:newPos]))))
	if nAlnum > 0 {
		m.userFacingPos = advanceByAlnums(m.origRunes, m.userFacingPos, nAlnum)
	} else {
		// A token with no letters or digits to spend, punctuation set off by a
		// space as French writes it ("va ?", "Attention :"). There is no budget
		// to advance by, so step straight to where the raw cursor got to, and the
		// mark leaves the remaining text now rather than a word later. Both sides
		// are identical here, so that offset is exact.
		m.userFacingPos = seg.origStart + rtrimLen(seg.ttsRunes[:newPos])
	}
	m.llmPos = advanceByAlnums(m.llmRunes, m.llmPos, nAlnum)
}

// commitTransformedSpan jumps the other two cursors to the end of seg, now that
// it is done.
func (m *TextSegmentMap) commitTransformedSpan(seg *textSegment) {
	m.userFacingPos = seg.origEnd
	// The original's count, not the TTS side's: llmText holds "$42.50" (4
	// alnums), never the spoken "forty two dollars".
	m.llmPos = advanceByAlnums(m.llmRunes, m.llmPos, seg.originalAlnumCount())
}

// finishSegment records seg as finished and moves on to the next segment.
func (m *TextSegmentMap) finishSegment(seg *textSegment) {
	m.lastCompleted = seg
	m.segIdx++
	m.segRawPos = 0
}

// planHops works out what word would do, without moving anything. This decides;
// consumeWord acts. Keeping them apart is what stops them disagreeing, since
// canConsumeWord asks the same question and must get the same answer.
//
// Most words are placed by the first segment tried, and the walk stops there. It
// goes on when a segment cannot finish the job, because the word runs past it or
// it has nothing speakable left, and whatever remains of the word is offered to
// the next segment.
//
// It returns what each segment answered, in order, and whatever is left of the
// word once the segments run out. Anything left over is the word running past
// the end of this text.
func (m *TextSegmentMap) planHops(word string) ([]hop, string) {
	segIdx := m.segIdx
	rawPos := m.segRawPos
	remaining := word
	var hops []hop

	for remaining != "" && segIdx < len(m.segments) {
		h := classifyHop(string(m.segments[segIdx].ttsRunes[rawPos:]), remaining)
		hops = append(hops, h)

		// hopPlaced and hopNoMatch both end the walk; the other two carry on.
		if h.kind == hopPlaced || h.kind == hopNoMatch {
			return hops, ""
		}

		remaining = runeSliceFrom(remaining, h.wordChars)
		segIdx++
		rawPos = 0
	}

	return hops, remaining
}

// consumeWord moves the cursors according to what planHops decided. Anything
// left unplaced ran past the end of this text and is kept in lastOverflow, for
// the caller to hand to the next frame.
func (m *TextSegmentMap) consumeWord(word string) {
	hops, overflow := m.planHops(word)

	for _, h := range hops {
		seg := &m.segments[m.segIdx]

		switch h.kind {
		case hopNoMatch:
			// The word belongs somewhere else entirely (a synthesizer swapping a
			// symbol, say). Nudge the raw cursor past any leading punctuation so
			// the next word is not blocked by it, but leave the cursors that mean
			// something alone: nothing was really spoken here.
			m.segRawPos += h.segChars
		case hopPlaced:
			m.advanceCursorsTo(seg, m.segRawPos+h.segChars)
		default:
			// hopCrosses or hopExhausted: this segment is done either way, and the
			// next hop was classified against the one after it.
			m.advanceCursorsTo(seg, len(seg.ttsRunes))
		}
	}

	if overflow != "" {
		m.lastOverflow = overflow
	}
}

// AdvanceWord takes one spoken word and moves every cursor to where it ends.
//
// Afterwards LastCompletedSegment, LastOverflow and LastLeadingDuplicate
// describe what this particular word did; each is cleared at the start of the
// next call.
//
// The word may be a plain word, a word carrying its own spacing or punctuation,
// or a fragment of a half-open tag. Matching is textual, so the caller does not
// have to know which.
func (m *TextSegmentMap) AdvanceWord(word string) {
	m.lastCompleted = nil
	m.lastOverflow = ""
	m.lastLeadingDuplicate = 0

	if word != "" {
		m.lastLeadingDuplicate = m.leadingDuplicateLen(word)
		m.consumeWord(word)
	}
}

// leadingDuplicateLen counts runes at the start of word that were already
// spoken.
//
// In "Yeah, I can" the comma travels with "Yeah". If the synthesizer then
// reports the next word as ", I" instead of "I", the comma would be recorded
// twice, so this returns 2: the comma and its space.
//
// It returns 0 when that punctuation is new text instead (`"hello`), and when
// the word is nothing but punctuation, which is a mark arriving on its own. It
// must be called before the cursors move, while llmPos still sits at the end of
// the previous word.
func (m *TextSegmentMap) leadingDuplicateLen(word string) int {
	runes := []rune(word)
	i := 0
	for i < len(runes) && (unicode.IsSpace(runes[i]) || isPunct(runes[i])) {
		i++
	}
	if i >= len(runes) {
		return 0
	}
	mark := []rune(strings.TrimSpace(string(runes[:i])))
	if len(mark) == 0 {
		return 0
	}
	start := m.llmPos - len(mark)
	if start < 0 || string(m.llmRunes[start:m.llmPos]) != string(mark) {
		return 0
	}
	return i
}

// WordBelongsCurrentSegment reports whether word could be the next thing spoken
// here. It is AdvanceWord without the moving, so a caller can check first. A
// false answer means the synthesizer skipped ahead, and the word should go to
// the next frame instead.
//
// A word with no letters or digits gets a second chance from symbolWordBelongs,
// since there is nothing in it to match on.
func (m *TextSegmentMap) WordBelongsCurrentSegment(word string) bool {
	if word == "" {
		return true
	}
	if m.canConsumeWord(word) {
		return true
	}
	if !hasAlnum(word) {
		return m.symbolWordBelongs(word)
	}
	return false
}

// canConsumeWord asks whether this word would be placed, without placing it. It
// is true if some segment would take the word, or if the word simply runs off
// the end, and false if there are no segments left or a segment turns it down.
func (m *TextSegmentMap) canConsumeWord(word string) bool {
	if m.segIdx >= len(m.segments) {
		return false
	}
	hops, _ := m.planHops(word)
	return len(hops) == 0 || hops[len(hops)-1].kind != hopNoMatch
}

// symbolWordBelongs decides whether a word made only of punctuation or symbols
// belongs here. There is nothing in such a word to match on, so it gets two
// chances.
//
// First, look for the word itself in the text still to be spoken. The search
// starts a little before the cursor, because punctuation is often taken along
// with the word before it.
//
// Second, accept it as a stand-in. Some synthesizers report a different symbol
// than the one they were given, ElevenLabs reports "->" as "-", so the first
// check can never succeed. If words remain to be spoken and the next thing in
// the text is itself a symbol, treat the word as that symbol.
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

// UserFacingPos is how far into the user-facing text the spoken words have
// reached, as a rune offset.
func (m *TextSegmentMap) UserFacingPos() int { return m.userFacingPos }

// LLMPos is how far into the LLM's text the spoken words have reached, as a rune
// offset.
func (m *TextSegmentMap) LLMPos() int { return m.llmPos }

// RawPos is how far into the TTS text the synthesizer has spoken, counted from
// its start as a rune offset.
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

// LastOverflow is the end of the last word passed to AdvanceWord, if it did not
// fit. It is "" most of the time, and is set only when that word ran past the end
// of the TTS text with no segment left to take the rest, which means the
// leftover belongs to the next frame. It is always the tail of the word that was
// passed in, so the part that did fit is the word minus this many trailing
// runes.
func (m *TextSegmentMap) LastOverflow() string { return m.lastOverflow }

// LastLeadingDuplicate is how much of the last word's start was punctuation
// already spoken, in runes.
//
// It is the opposite end of the word from LastOverflow: that one is about a tail
// running past this text, this one about a head repeating punctuation the
// previous word already took. Cut both off to get the part of the word that
// belongs to this frame.
func (m *TextSegmentMap) LastLeadingDuplicate() int { return m.lastLeadingDuplicate }

// IsComplete reports whether every letter and digit in the text has been spoken.
//
// That is not the same as the cursor reaching the end. If all that is left is
// punctuation or tags, the text counts as finished even though those runes have
// not been walked over, because no word event is coming for them.
//
// There is one exception. Punctuation separated from its word by a space, as
// French writes "Comment ça va ?", does arrive as its own word event, so the
// text stays unfinished until it does. Punctuation stuck to the word itself, as
// in "you?", was already taken with the word.
func (m *TextSegmentMap) IsComplete() bool {
	if m.segIdx >= len(m.segments) {
		return true
	}
	seg := &m.segments[m.segIdx]
	rem := string(seg.ttsRunes[m.segRawPos:])
	if hasAlnum(rem) {
		return false
	}
	if pendingSeparatedPunctuation(rem) {
		return false
	}
	for i := m.segIdx + 1; i < len(m.segments); i++ {
		if hasAlnum(m.segments[i].tts()) {
			return false
		}
	}
	return true
}

// pendingSeparatedPunctuation reports whether all that is left is punctuation
// set off from its word by a space. Some languages write a space before a mark,
// as in "va ?" or "Bonjour !", and a synthesizer reports that mark as a word of
// its own, so the segment has to stay open until it arrives.
//
// Only real punctuation counts. A trailing emoji or arrow ("day! 😊", "→") never
// arrives as its own word, so it must not keep the segment open, and tags are
// removed first for the same reason. It is only called once no letters or digits
// are left.
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

// InTransformedSegment reports whether the cursor is partway through a rewritten
// segment.
func (m *TextSegmentMap) InTransformedSegment() bool {
	if m.segIdx >= len(m.segments) {
		return false
	}
	seg := &m.segments[m.segIdx]
	return seg.isTransformed() && m.segRawPos > 0
}

// LastCompletedSegment returns the original text of the segment finished by the
// last AdvanceWord call, and whether one finished.
func (m *TextSegmentMap) LastCompletedSegment() (original string, ok bool) {
	if m.lastCompleted == nil {
		return "", false
	}
	return m.lastCompleted.original(), true
}

// hasLastCompleted reports whether the last AdvanceWord finished a segment.
func (m *TextSegmentMap) hasLastCompleted() bool { return m.lastCompleted != nil }

// Reset puts every cursor back to the start of the text.
func (m *TextSegmentMap) Reset() {
	m.segIdx = 0
	m.segRawPos = 0
	m.userFacingPos = 0
	m.llmPos = 0
	m.lastCompleted = nil
	m.lastOverflow = ""
	m.lastLeadingDuplicate = 0
}
