package text

import (
	"regexp"
	"strings"
	"unicode"
)

// The classes the phrase pattern is built from. Go's \b and \s only know ASCII,
// so a phrase in another script, or one ending in an accented letter, would
// never sit on a boundary; these spell out the Unicode word characters and
// whitespace instead.
const (
	phraseWordClass    = `[\p{L}\p{N}_]`
	phraseNonWordClass = `[^\p{L}\p{N}_]`
	phraseSpaceClass   = `[\t\n\v\f\r\x{1c}-\x{1f}\x{85}\p{Z}]`
)

// CompilePhrasePattern compiles a case-insensitive pattern that finds a spoken
// phrase in text.
//
// The words of the phrase may be separated by any amount of whitespace
// (including none). The match must start and end on a word boundary, except next
// to characters of scripts written without spaces (Chinese, Japanese, Thai),
// where speech-to-text output has no boundaries between words: in
// "你好小助手今天天气怎么样" there is no boundary between "手" and "今", because
// both are word characters.
//
// The pattern is for telling whether, and roughly where, the phrase occurs: a
// boundary is matched by the character on its far side, so a match can take in
// one character before and after the phrase. An empty phrase matches any text
// holding a word character, which is where a word boundary is.
func CompilePhrasePattern(phrase string) *regexp.Regexp {
	words := strings.FieldsFunc(phrase, isPhraseSpace)
	if len(words) == 0 {
		return regexp.MustCompile(phraseWordClass)
	}
	quoted := make([]string, len(words))
	for i, w := range words {
		quoted[i] = regexp.QuoteMeta(w)
	}
	first := []rune(words[0])[0]
	last := []rune(words[len(words)-1])
	return regexp.MustCompile(`(?i)` + startBoundary(first) +
		strings.Join(quoted, phraseSpaceClass+`*`) + endBoundary(last[len(last)-1]))
}

// startBoundary is what must come before a phrase starting with r for the
// phrase to start on a word boundary.
func startBoundary(r rune) string {
	switch {
	case isUnspacedScript(r):
		return ""
	case isPhraseWord(r):
		return `(?:^|` + phraseNonWordClass + `)`
	default:
		return phraseWordClass
	}
}

// endBoundary is what must come after a phrase ending with r for the phrase to
// end on a word boundary.
func endBoundary(r rune) string {
	switch {
	case isUnspacedScript(r):
		return ""
	case isPhraseWord(r):
		return `(?:$|` + phraseNonWordClass + `)`
	default:
		return phraseWordClass
	}
}

// isPhraseWord reports whether r is a word character: a letter, a number or an
// underscore.
func isPhraseWord(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_'
}

// isPhraseSpace reports whether r separates the words of a phrase.
func isPhraseSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

// isUnspacedScript reports whether r belongs to a script written without spaces
// between words: Thai, Japanese kana, and the CJK ideographs.
func isUnspacedScript(r rune) bool {
	switch {
	case r >= 0x0e00 && r <= 0x0e7f, // Thai
		r >= 0x3040 && r <= 0x30ff,   // Hiragana and Katakana
		r >= 0x3400 && r <= 0x4dbf,   // CJK Extension A
		r >= 0x4e00 && r <= 0x9fff,   // CJK Unified Ideographs
		r >= 0xf900 && r <= 0xfaff,   // CJK Compatibility Ideographs
		r >= 0x20000 && r <= 0x2fa1f: // CJK Extensions B onward
		return true
	}
	return false
}
