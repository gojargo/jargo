package context

import (
	"log/slog"
	"strings"
	"unicode"
)

// WordCompletionTracker tracks whether all words of one aggregated text frame
// have been spoken, and maps each spoken word back to its span in the original
// text. It delegates cursor advancement to a TextSegmentMap built from the
// transformed ttsText: unchanged segments advance proportionally, transformed
// segments (e.g. "$42.50" spoken as "forty two dollars and fifty cents") are
// held atomic and jump to the end of the original span once fully spoken.
//
// When llmText is provided, the tracker additionally maps each spoken word back
// to its corresponding span there, so callers can attach the original written
// text to per-word frames and the conversation context receives properly-formed
// content rather than the cleaned words the synthesizer reports.
type WordCompletionTracker struct {
	ttsRunes        []rune
	userFacingRunes []rune
	userFacingPos   int

	hasLLM   bool
	llmRunes []rune
	llmPos   int

	overflowWord string
	overflowSet  bool
	llmConsumed  string
	llmSet       bool
	frameWord    string
	frameSet     bool

	forceCompleted bool
	segmentMap     *TextSegmentMap
}

// NewWordCompletionTracker builds a tracker for the frame being spoken. ttsText
// is the text sent to the synthesizer (may carry synthesis markup).
// userFacingText is the text as shown to the user; pass "" to default it to
// ttsText with markup stripped. llmText is the original model-produced text
// (with any delimiters); pass "" to disable original-text mapping.
func NewWordCompletionTracker(ttsText, userFacingText, llmText string) *WordCompletionTracker {
	if userFacingText == "" {
		userFacingText = stripCompleteMarkup(ttsText)
	}
	t := &WordCompletionTracker{
		ttsRunes:        []rune(ttsText),
		userFacingRunes: []rune(userFacingText),
		hasLLM:          llmText != "",
		llmRunes:        []rune(llmText),
		segmentMap:      NewTextSegmentMap(ttsText, userFacingText, llmText),
	}
	return t
}

// AddWord records a spoken word from a word-timestamp event and reports whether
// the frame is now fully spoken. Before advancing, it checks whether the word
// belongs to this frame; if not (the synthesizer dropped a word-timestamp
// event), the slot is force-completed: the remaining unspoken TTS text is stored
// as the frame word, all remaining original text is consumed, and the incoming
// word is routed as overflow for the next slot.
func (t *WordCompletionTracker) AddWord(word string) bool {
	t.overflowWord, t.overflowSet = "", false
	t.llmConsumed, t.llmSet = "", false
	t.frameWord, t.frameSet = "", false

	if t.forceCompleted || t.segmentMap.RawPos() >= len(t.ttsRunes) {
		return true
	}

	if !t.WordBelongsHere(word) {
		t.frameWord, t.frameSet = string(t.ttsRunes[t.segmentMap.RawPos():]), true
		t.userFacingPos = len(t.userFacingRunes)
		if t.hasLLM {
			t.llmConsumed, t.llmSet = string(t.llmRunes[t.llmPos:]), true
			t.llmPos = len(t.llmRunes)
			t.discardLLMSpanIfFrameWordMissing()
		}
		t.forceCompleted = true
		t.overflowWord, t.overflowSet = word, true
		return true
	}

	prevLLMPos := t.llmPos
	t.segmentMap.AdvanceWord(word)

	overflow := t.segmentMap.LastOverflow()
	if overflow != "" {
		wr := []rune(word)
		or := []rune(overflow)
		t.frameWord, t.frameSet = string(wr[:len(wr)-len(or)]), true
		t.overflowWord, t.overflowSet = overflow, true
	} else {
		t.frameWord, t.frameSet = word, true
	}

	t.userFacingPos = t.segmentMap.UserFacingPos()
	t.llmPos = t.segmentMap.LLMPos()

	if t.hasLLM {
		t.attributeLLMConsumed(word, prevLLMPos)
	}
	return t.IsComplete()
}

// attributeLLMConsumed sets llmConsumed to the original-text span the
// just-advanced word maps to.
func (t *WordCompletionTracker) attributeLLMConsumed(word string, prevLLMPos int) {
	completed := t.segmentMap.hasLastCompleted()
	completedTransformed := t.segmentMap.lastCompletedTransformed()

	switch {
	case t.IsComplete():
		t.llmConsumed, t.llmSet = string(t.llmRunes[prevLLMPos:]), true
		t.llmPos = len(t.llmRunes)
		if !completedTransformed {
			t.discardLLMSpanIfFrameWordMissing()
		}
	case t.segmentMap.InTransformedSegment():
		t.llmConsumed, t.llmSet = "", false
	case t.llmPos == prevLLMPos && !completed:
		start := t.llmPos
		for start < len(t.llmRunes) && unicode.IsSpace(t.llmRunes[start]) {
			start++
		}
		end := min(start+len([]rune(word)), len(t.llmRunes))
		t.llmConsumed, t.llmSet = string(t.llmRunes[start:end]), true
		t.llmPos = end
	default:
		t.llmConsumed, t.llmSet = string(t.llmRunes[prevLLMPos:t.llmPos]), true
		if !completedTransformed {
			t.discardLLMSpanIfFrameWordMissing()
		}
	}
}

// discardLLMSpanIfFrameWordMissing drops the attributed span when it does not
// contain the current frame word, guarding against the TTS and original texts
// drifting out of sync. The comparison is case- and connector-insensitive.
func (t *WordCompletionTracker) discardLLMSpanIfFrameWordMissing() {
	frameWord := stripTrailingPunctuation(t.frameWord)
	if frameWord == "" {
		return
	}
	foldedSpan := foldForComparison(t.llmConsumed)
	if strings.Contains(foldedSpan, foldForComparison(frameWord)) {
		return
	}

	// The word may lead with punctuation the word before it already took:
	// advancing sweeps the punctuation trailing a word into that word's span, so
	// a provider that reports it with the following word instead (", I" rather
	// than "Yeah,") presents it a second time. Drop the duplicate from the frame
	// word rather than the whole attribution, which would otherwise lose the
	// written form of a word that was spoken perfectly well.
	if trimmed := stripLeadingPunctuation(frameWord); trimmed != "" &&
		strings.Contains(foldedSpan, foldForComparison(trimmed)) {
		t.frameWord = stripLeadingPunctuation(t.frameWord)
		return
	}

	slog.Warn("tracker: the span attributed to a word does not contain it, discarding",
		"span", t.llmConsumed, "word", t.frameWord)
	t.llmConsumed, t.llmSet = "", false
}

// WordBelongsHere reports whether word plausibly belongs to the remaining TTS
// text of this frame. It is how a dropped word-timestamp event is detected: a
// word that does not match this frame's remaining content belongs to the next
// one, and this frame has to be force-completed.
func (t *WordCompletionTracker) WordBelongsHere(word string) bool {
	return t.segmentMap.WordBelongsCurrentSegment(word)
}

// Suppress reports whether the last word is mid-flight inside a transformed
// segment. When true, the per-word frame must not be written to the context;
// only the completing word of the segment carries the original text.
func (t *WordCompletionTracker) Suppress() bool {
	return t.segmentMap.InTransformedSegment()
}

// FrameWord returns the portion of the last word belonging to this frame,
// whitespace-trimmed, and whether one was recorded.
func (t *WordCompletionTracker) FrameWord() (string, bool) {
	if !t.frameSet {
		return "", false
	}
	return strings.TrimSpace(t.frameWord), true
}

// OverflowWord returns the raw suffix of the last word that overflows into the
// next frame, whitespace-trimmed, and whether there was overflow.
func (t *WordCompletionTracker) OverflowWord() (string, bool) {
	if !t.overflowSet {
		return "", false
	}
	return strings.TrimSpace(t.overflowWord), true
}

// RawText returns the original-text span consumed for the last added word,
// whitespace-trimmed, and whether one was recorded. It is never set when no
// original text was provided, or for an intermediate word of a transformed
// segment.
func (t *WordCompletionTracker) RawText() (string, bool) {
	if !t.llmSet {
		return "", false
	}
	return strings.TrimSpace(t.llmConsumed), true
}

// RemainingRawText returns the unspoken portion of the original text, trimmed.
// It is used to close out a frame so the context receives the full original
// text when the frame was not interrupted. Returns "" when nothing remains or no
// original text was provided.
func (t *WordCompletionTracker) RemainingRawText() string {
	if !t.hasLLM {
		return strings.TrimSpace(string(t.userFacingRunes[t.userFacingPos:]))
	}
	return strings.TrimSpace(string(t.llmRunes[t.llmPos:]))
}

// AccumulatedUserFacingText returns the user-facing text consumed so far.
func (t *WordCompletionTracker) AccumulatedUserFacingText() string {
	return string(t.userFacingRunes[:t.userFacingPos])
}

// RemainingUserFacingText returns the unspoken portion of the user-facing text.
// Leading whitespace is kept unless strip is set, so accumulated plus remaining
// reconstructs the original exactly.
func (t *WordCompletionTracker) RemainingUserFacingText(strip bool) string {
	remaining := string(t.userFacingRunes[t.userFacingPos:])
	if strip {
		return strings.TrimSpace(remaining)
	}
	return remaining
}

// AccumulatedTTSText returns the text sent to the synthesizer that has been
// consumed so far. Unlike FrameWord, which reflects only the last word, this is
// everything since construction or the last Reset.
func (t *WordCompletionTracker) AccumulatedTTSText() string {
	return string(t.ttsRunes[:t.segmentMap.RawPos()])
}

// RemainingTTSText returns the unspoken portion of the text sent to the
// synthesizer. Leading whitespace is kept unless strip is set.
func (t *WordCompletionTracker) RemainingTTSText(strip bool) string {
	remaining := string(t.ttsRunes[t.segmentMap.RawPos():])
	if strip {
		return strings.TrimSpace(remaining)
	}
	return remaining
}

// AccumulatedRawText returns the original text consumed so far, and whether any
// original text was provided.
func (t *WordCompletionTracker) AccumulatedRawText() (string, bool) {
	if !t.hasLLM {
		return "", false
	}
	return string(t.llmRunes[:t.llmPos]), true
}

// RemainingRawTextOnly returns the unspoken portion of the original text,
// trimmed, and whether any original text was provided. It differs from
// RemainingRawText, which falls back to the user-facing text when there is none.
func (t *WordCompletionTracker) RemainingRawTextOnly() (string, bool) {
	if !t.hasLLM {
		return "", false
	}
	return strings.TrimSpace(string(t.llmRunes[t.llmPos:])), true
}

// IsComplete reports whether this frame's TTS text has been fully accounted for.
func (t *WordCompletionTracker) IsComplete() bool {
	return t.forceCompleted || t.segmentMap.IsComplete()
}

// Reset returns the tracker to its initial state without changing the texts.
func (t *WordCompletionTracker) Reset() {
	t.userFacingPos = 0
	t.llmPos = 0
	t.overflowWord, t.overflowSet = "", false
	t.llmConsumed, t.llmSet = "", false
	t.frameWord, t.frameSet = "", false
	t.forceCompleted = false
	t.segmentMap.Reset()
}

// typographyFold maps common typographic punctuation variants to their ASCII
// equivalents.
//
//nolint:gochecknoglobals // read-only replacement table
var typographyFold = strings.NewReplacer(
	"‘", "'", "’", "'", "ʼ", "'",
	"“", `"`, "”", `"`,
	"–", "-", "—", "-",
)

// foldForComparison folds text for lenient span-containment comparisons: it
// applies typographic folding, lowercases, and collapses connector characters
// (spaces and hyphens), so case- or connector-only differences are not mistaken
// for a mismatch while other content is preserved.
func foldForComparison(text string) string {
	folded := strings.ToLower(typographyFold.Replace(text))
	var b strings.Builder
	b.Grow(len(folded))
	for _, r := range folded {
		if r == '-' || unicode.IsSpace(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}
