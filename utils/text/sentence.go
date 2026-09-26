package text

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/gojargo/jargo/internal/sentencex"
)

// sentenceEndingPunctuation is every mark that can end a sentence, across the
// scripts a voice agent is likely to speak.
//
//nolint:gochecknoglobals // read-only lookup table
var sentenceEndingPunctuation = map[rune]bool{
	// Latin script (most European languages, Filipino).
	'.': true, '!': true, '?': true, ';': true, '…': true,
	// East Asian (Chinese, Japanese, Korean).
	'。': true, '？': true, '！': true, '；': true, '．': true, '｡': true,
	// Indic scripts.
	'।': true, '॥': true,
	// Arabic script (Arabic, Persian, Urdu, Pashto).
	'؟': true, '؛': true, '۔': true, '؏': true,
	// Myanmar.
	'၊': true, '။': true,
	// Khmer.
	'។': true, '៕': true,
	// Lao and Tibetan.
	'໌': true, '༎': true, '།': true,
	// Armenian.
	'։': true, '՜': true, '՞': true,
	// Ethiopic (Amharic).
	'።': true, '፧': true, '፨': true,
}

// latinSentenceEndingPunctuation is the punctuation the sentence rules
// disambiguate, since a period also appears in abbreviations and decimals.
//
//nolint:gochecknoglobals // read-only lookup table
var latinSentenceEndingPunctuation = map[rune]bool{
	'.': true, '!': true, '?': true, ';': true, '…': true,
}

// IsSentenceEnding reports whether r can end a sentence.
func IsSentenceEnding(r rune) bool { return sentenceEndingPunctuation[r] }

// isUnambiguousEnding reports whether r ends a sentence without needing the
// rules' judgement, which is what makes it usable for a script they do not
// cover.
func isUnambiguousEnding(r rune) bool {
	return sentenceEndingPunctuation[r] && !latinSentenceEndingPunctuation[r]
}

// ResolveSentenceTokenizerLanguage normalizes a language code for the sentence
// rules' fallback map: lowercased, reduced to its base code ("pt-BR" and "pt_BR"
// become "pt"). Empty and "auto" use English. A code the rules do not know is
// resolved by the rules themselves.
func ResolveSentenceTokenizerLanguage(language string) string {
	normalized := strings.ToLower(strings.TrimSpace(language))
	normalized, _, _ = strings.Cut(strings.ReplaceAll(normalized, "_", "-"), "-")
	if normalized == "" || normalized == "auto" {
		return "en"
	}
	return normalized
}

// sentTokenize splits text into sentences, keeping the source offsets up to the
// whitespace between them: the whitespace separating two sentences belongs to
// the one that follows, so trailing whitespace stays in a buffer until that
// sentence is emitted.
func sentTokenize(language, text string) []string {
	var (
		sentences []string
		pending   string
	)
	for _, span := range sentencex.Segment(language, text) {
		pending += span
		if strings.TrimSpace(span) != "" {
			sentence := strings.TrimRightFunc(pending, unicode.IsSpace)
			sentences = append(sentences, sentence)
			pending = pending[len(sentence):]
		}
	}
	return sentences
}

// openingQuotes are the quote marks a quoted word can open with.
const openingQuotes = "\"'“‘«‹„‟「『"

// quotedWordEnd finds a quoted word ending in a period at the end of text: one
// or more opening quotes not preceded by a word character, then word characters
// or periods, then a final period. It reports the byte span of the quotes.
func quotedWordEnd(text string) (start, end int, ok bool) {
	if !strings.HasSuffix(text, ".") {
		return 0, 0, false
	}
	// Walk back over the word, which is word characters and periods, then over
	// the quotes before it.
	i := len(text) - 1
	for i > 0 {
		r, size := utf8.DecodeLastRuneInString(text[:i])
		if r != '.' && !isWordRune(r) {
			break
		}
		i -= size
	}
	if i == len(text)-1 {
		return 0, 0, false // no word character before the final period
	}
	end = i
	for i > 0 {
		r, size := utf8.DecodeLastRuneInString(text[:i])
		if !strings.ContainsRune(openingQuotes, r) {
			break
		}
		i -= size
	}
	if i == end {
		return 0, 0, false
	}
	if i > 0 {
		if r, _ := utf8.DecodeLastRuneInString(text[:i]); isWordRune(r) {
			// The first quote follows a word character, so the match starts at
			// the next quote, which follows a quote; with only one, there is
			// none.
			_, size := utf8.DecodeRuneInString(text[i:])
			if i += size; i == end {
				return 0, 0, false
			}
		}
	}
	return i, end, true
}

// isWordRune reports whether r is a word character: a letter, a number or an
// underscore.
func isWordRune(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r) }

// matchEndOfSentence is what the aggregators find a boundary with. It is a
// variable so a test can count and steer the checks the lookahead makes.
//
//nolint:gochecknoglobals // swapped by tests
var matchEndOfSentence = MatchEndOfSentence

// MatchEndOfSentence returns the byte offset just past the end of the first
// complete sentence in text, in the given language, or 0 when text does not
// yet hold one. An empty language uses English.
//
// The rules' answer is verified before it is trusted: asked about a fragment
// they return it as one sentence, and LLM output arrives a token at a time, so
// a lone sentence counts only when the text ends on sentence-ending
// punctuation.
func MatchEndOfSentence(text, language string) int {
	text = strings.TrimRightFunc(text, unicode.IsSpace)
	if text == "" {
		return 0
	}
	code := ResolveSentenceTokenizerLanguage(language)
	sentences := sentTokenize(code, text)
	if len(sentences) == 0 {
		return 0
	}
	first := sentences[0]

	tokenizerText := text
	// With an unfinished quote such as 'She said, "Dr. S', the rules can split
	// after '"Dr.'. A proposed split after a quoted word ending in a period is
	// checked again, letting the rules decide whether that word is an
	// abbreviation.
	if len(sentences) > 1 {
		if start, end, ok := quotedWordEnd(first); ok {
			// Only the opening quotes are replaced, with spaces of the same byte
			// width, so the rules see the abbreviation without the quote and the
			// offsets still refer to the text, which keeps its quotes.
			tokenizerText = text[:start] + strings.Repeat(" ", end-start) + text[end:]
			sentences = sentTokenize(code, tokenizerText)
			first = sentences[0]
		}
	}

	// A single span can be an incomplete fragment; require terminal punctuation.
	if len(sentences) == 1 && first == tokenizerText {
		if last, ok := lastRune(text); ok && sentenceEndingPunctuation[last] {
			return len(text)
		}
		// Punctuation that needs no disambiguation delimits a sentence in a
		// script the rules do not cover.
		for i, r := range text {
			if isUnambiguousEnding(r) {
				return i + utf8.RuneLen(r)
			}
		}
		return 0
	}

	if len(sentences) > 1 {
		return len(first)
	}
	return 0
}

// lastRune returns the final rune of text.
func lastRune(text string) (rune, bool) {
	if text == "" {
		return 0, false
	}
	r, _ := utf8.DecodeLastRuneInString(text)
	return r, true
}
