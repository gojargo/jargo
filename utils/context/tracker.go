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
	// llmPos is what has been attributed and llmSpokenPos what has been reported
	// spoken. They stop in different places, which is why there are two: a mark
	// stuck to the end of a word is attributed to it, while nothing has reported
	// speaking that mark.
	llmPos       int
	llmSpokenPos int

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

// AddWord records one word the synthesizer reported speaking, and reports
// whether the frame is now fully spoken.
//
// Three things can happen, in this order. The frame is already finished, and the
// word is ignored. The word does not match what is left to speak, so the
// synthesizer must have dropped an event: the frame is force-completed and this
// word is handed back as overflow. Otherwise the word advances the frame, and
// afterwards the accessors describe it: this frame's share of the word, the
// original text it stands for, and how much of the frame is now spoken.
func (t *WordCompletionTracker) AddWord(word string) bool {
	t.overflowWord, t.overflowSet = "", false
	t.llmConsumed, t.llmSet = "", false
	t.frameWord, t.frameSet = "", false

	// Every raw rune consumed, not IsComplete: a frame ending in an emoji
	// contributes no alphanumeric content, so it reads as complete before that
	// emoji's own event arrives, and that event is still wanted.
	if t.forceCompleted || t.segmentMap.RawPos() >= len(t.ttsRunes) {
		slog.Warn("tracker: a word arrived for a frame that is already complete", "word", word)
		return true
	}

	if !t.WordBelongsHere(word) {
		return t.forceComplete(word)
	}

	prevLLMPos := t.llmPos
	prevSpokenPos := t.llmSpokenPos
	prevRawPos := t.segmentMap.RawPos()
	t.segmentMap.AdvanceWord(word)

	overflow := t.segmentMap.LastOverflow()
	if overflow != "" {
		t.overflowWord, t.overflowSet = overflow, true
	}
	spoken := string(t.ttsRunes[prevRawPos:t.segmentMap.RawPos()])

	// Where the text just crossed holds more letters and digits than the word
	// reported, the synthesizer skipped some: markup does not count, because
	// synthesizers do not report tags, and only letters and digits are compared
	// so case, accents, punctuation and spacing make no difference.
	coversSkipped := len(alnumOnly(stripMarkup(spoken))) > len(alnumOnly(word))

	if coversSkipped {
		// The text passed over will never be reported on its own, so it travels
		// with the word that brings the tracker back in sync rather than being
		// dropped from the turn.
		t.frameWord, t.frameSet = spoken, true
	} else {
		// Neither end of the token is necessarily this frame's: the head can
		// repeat punctuation the previous word already carried, and the tail can
		// run into the next frame. The map measures both; keep what is between.
		// Without an original text there is no recorded span that could already
		// have carried the mark, so it is new text on this frame.
		wr := []rune(word)
		head := 0
		if t.hasLLM {
			head = t.segmentMap.LastLeadingDuplicate()
		}
		tail := len(wr)
		if overflow != "" {
			tail = len(wr) - len([]rune(overflow))
		}
		if head > tail {
			head = tail
		}
		t.frameWord, t.frameSet = string(wr[head:tail]), true
	}

	t.userFacingPos = t.segmentMap.UserFacingPos()
	t.llmPos = t.segmentMap.LLMPos()
	t.llmSpokenPos = t.segmentMap.LLMSpokenPos()

	if t.hasLLM {
		t.recordLLMSpan(word, prevLLMPos, prevSpokenPos)
	}

	complete := t.IsComplete()
	if complete {
		// Everything speakable has been spoken, so anything still left is text no
		// word will ever arrive for: a closing tag, or one sitting between the
		// last word and its punctuation. It belongs to this frame, so take it
		// rather than leave it out of the turn.
		t.userFacingPos = len(t.userFacingRunes)
	}
	return complete
}

// forceComplete ends this frame early because word does not belong to it.
//
// The synthesizer dropped one or more events, so the rest of this frame will
// never be reported. Rather than stall, it emits the unspoken remainder as this
// frame's word, so the conversation context still receives the full text, and
// hands word back as overflow for the next frame to try.
//
// The segment map is deliberately left where it is, since nothing here was
// actually spoken; forceCompleted answers for it from now on.
func (t *WordCompletionTracker) forceComplete(word string) bool {
	t.frameWord, t.frameSet = string(t.ttsRunes[t.segmentMap.RawPos():]), true
	t.userFacingPos = len(t.userFacingRunes)
	if t.hasLLM {
		// The whole remainder is this frame's by definition, tags included.
		t.llmConsumed, t.llmSet = string(t.llmRunes[t.llmSpokenPos:]), true
		t.llmPos = len(t.llmRunes)
		t.llmSpokenPos = len(t.llmRunes)
	}
	t.forceCompleted = true
	t.overflowWord, t.overflowSet = word, true
	return true
}

// TakeRemainingAsSpoken moves the cursors to the end, for a caller that emits
// the remainder itself. The accumulated and remaining views then agree with the
// text that went out.
func (t *WordCompletionTracker) TakeRemainingAsSpoken() {
	t.userFacingPos = len(t.userFacingRunes)
	if t.hasLLM {
		t.llmPos = len(t.llmRunes)
		t.llmSpokenPos = len(t.llmRunes)
	}
}

// recordLLMSpan records which part of the original text the word just added
// stands for.
//
// Usually that is simply the span the map's cursor moved over. Two cases reach
// further, and both leave llmPos ahead of the map's. When the word finished the
// frame, take everything to the end: the map stops at the last spoken rune, so a
// closing tag, which never arrives as its own event, is still outstanding and
// belongs to this word. When the cursor did not move, because the map placed the
// word without spending any budget (an emoji or symbol), take the word's own
// length, skipping spaces the previous word owns.
//
// A word inside a transformed segment records nothing, and is checked before
// that second case: the cursor is held there on purpose, so "did not move" would
// be misread as "spent nothing" and would walk the cursor through text the
// transform covers. Only the word completing the segment carries its original
// span.
func (t *WordCompletionTracker) recordLLMSpan(word string, prevLLMPos, prevSpokenPos int) {
	switch {
	case t.IsComplete():
		t.llmConsumed, t.llmSet = string(t.llmRunes[prevLLMPos:]), true
		t.llmPos = len(t.llmRunes)
		t.llmSpokenPos = len(t.llmRunes)
	case t.segmentMap.InTransformedSegment():
		t.llmConsumed, t.llmSet = "", false
	case t.llmSpokenPos == prevSpokenPos && t.llmPos == prevLLMPos && !t.segmentMap.hasLastCompleted():
		start := t.llmPos
		for start < len(t.llmRunes) && unicode.IsSpace(t.llmRunes[start]) {
			start++
		}
		end := min(start+len([]rune(word)), len(t.llmRunes))
		t.llmConsumed, t.llmSet = string(t.llmRunes[start:end]), true
		t.llmPos = end
	default:
		t.llmConsumed, t.llmSet = string(t.llmRunes[prevLLMPos:t.llmPos]), true
	}
}

// WordBelongsHere reports whether word plausibly belongs to the remaining TTS
// text of this frame. It is how a dropped word-timestamp event is detected: a
// word that does not match this frame's remaining content belongs to the next
// one, and this frame has to be force-completed.
func (t *WordCompletionTracker) WordBelongsHere(word string) bool {
	return t.segmentMap.WordBelongsCurrentSegment(word)
}

// Suppress reports whether the last word must not be written to the
// conversation context. Two kinds of word answer to this, both of which the
// context already has covered, or will have.
//
// One step inside a rewritten span: "$42.50" is spoken as five words, none of
// which the transcript should contain, and the word that finishes the span
// carries "$42.50" for all of them.
//
// A word with no span of its own: a synthesizer reporting "," on its own, after
// "Yeah" took the comma into its span, has nothing left to record. The context
// falls back to the spoken text when a word carries no span, which would store
// the mark a second time.
//
// The word is still emitted either way: the synthesizer spoke it, and a consumer
// reading the word stream should see it. Without an original text there are no
// spans at all, so nothing is suppressed and every word is recorded from its
// spoken text as usual.
func (t *WordCompletionTracker) Suppress() bool {
	return t.segmentMap.InTransformedSegment() ||
		(t.hasLLM && strings.TrimSpace(t.llmConsumed) == "")
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
	if !t.llmSet || t.llmConsumed == "" {
		return "", false
	}
	trimmed := strings.TrimSpace(t.llmConsumed)
	if trimmed == "" {
		return "", false
	}
	return trimmed, true
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
	return string(t.llmRunes[:t.llmSpokenPos]), true
}

// RemainingRawTextOnly returns the unspoken portion of the original text,
// trimmed, and whether any original text was provided. It differs from
// RemainingRawText, which falls back to the user-facing text when there is none.
func (t *WordCompletionTracker) RemainingRawTextOnly() (string, bool) {
	if !t.hasLLM {
		return "", false
	}
	return strings.TrimSpace(string(t.llmRunes[t.llmSpokenPos:])), true
}

// IsComplete reports whether this frame's TTS text has been fully accounted for.
func (t *WordCompletionTracker) IsComplete() bool {
	return t.forceCompleted || t.segmentMap.IsComplete()
}

// Reset returns the tracker to its initial state without changing the texts.
func (t *WordCompletionTracker) Reset() {
	t.userFacingPos = 0
	t.llmPos = 0
	t.llmSpokenPos = 0
	t.overflowWord, t.overflowSet = "", false
	t.llmConsumed, t.llmSet = "", false
	t.frameWord, t.frameSet = "", false
	t.forceCompleted = false
	t.segmentMap.Reset()
}
