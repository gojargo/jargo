// Package context provides the text-alignment utilities that let a TTS service
// map the words it actually spoke back to the original written text, so the
// conversation context can be truncated to exactly what was spoken when the bot
// is interrupted.
//
// The three cooperating types are:
//
//   - TextSegmentMap diffs the transformed text sent to a synthesizer against
//     the original written text and, as spoken words stream in, advances a
//     cursor through the original text. Unchanged spans advance proportionally;
//     transformed spans (e.g. "$42.50" spoken as "forty two dollars and fifty
//     cents") are held atomic and jump in one step once fully spoken.
//   - WordCompletionTracker wraps a map for one aggregated text frame, reporting
//     when the frame has been fully spoken and which span of original text each
//     spoken word maps to.
//   - MergePunctTokens normalizes raw word-timestamp streams.
//
// All matching is purely textual: callers feed raw word-timestamp tokens and
// need not parse markup or punctuation themselves.
package context

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// isAlnum reports whether r is a letter or a number, matching the alphanumeric
// test used throughout the cursor arithmetic.
func isAlnum(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsNumber(r)
}

// isPunct reports whether r is a Unicode punctuation character (category P*).
func isPunct(r rune) bool { return unicode.IsPunct(r) }

// stripTrailingPunctuation removes any trailing run of Unicode punctuation from
// text.
func stripTrailingPunctuation(text string) string {
	r := []rune(text)
	i := len(r)
	for i > 0 && isPunct(r[i-1]) {
		i--
	}
	return string(r[:i])
}

// typographyFold maps typographic punctuation variants to their ASCII
// equivalents. Models write the typographic forms and a synthesizer may report
// the ASCII ones in its word-timestamp events, or the reverse. Every entry maps
// one rune to one rune, which is what lets foldForMatching keep its 1:1 rune
// contract.
//
//nolint:gochecknoglobals // read-only replacement table
var typographyFold = strings.NewReplacer(
	"‘", "'", // U+2018 LEFT SINGLE QUOTATION MARK
	"’", "'", // U+2019 RIGHT SINGLE QUOTATION MARK
	"ʼ", "'", // U+02BC MODIFIER LETTER APOSTROPHE
	"“", `"`, // U+201C LEFT DOUBLE QUOTATION MARK
	"”", `"`, // U+201D RIGHT DOUBLE QUOTATION MARK
	"–", "-", // U+2013 EN DASH
	"—", "-", // U+2014 EM DASH
)

// foldTypography replaces typographic punctuation variants with their ASCII
// equivalents, keeping the rune length of text unchanged.
func foldTypography(text string) string { return typographyFold.Replace(text) }

// foldAccentedChar lowercases r, reducing it to its base letter when it carries
// a combining accent. NFD decomposition splits an accented character into a base
// letter plus a combining mark (category Mn); dropping the mark keeps the base.
// It always returns exactly one rune, so callers can rely on a 1:1 rune mapping.
func foldAccentedChar(r rune) rune {
	d := []rune(norm.NFD.String(string(r)))
	if len(d) >= 2 && unicode.Is(unicode.Mn, d[1]) {
		return unicode.ToLower(d[0])
	}
	return unicode.ToLower(r)
}

// foldForMatching folds away surface variation between two spellings of the
// same text, one rune for one rune.
//
// Unlike alnumOnly it never removes or merges characters: punctuation, spaces
// and markup pass through unchanged and each output rune corresponds to the
// same-index input rune, so a rune offset computed against the folded text
// applies unchanged to the original. That is what lets it be dropped in front of
// a position-based literal comparison.
//
// It folds case, accents and typographic punctuation ("’" to "'", "–" to "-"),
// the variations a synthesizer may introduce between the text it was sent and
// the words it reports back. It is deliberately narrow: each folded rune is
// listed in typographyFold rather than applying a blanket Unicode compatibility
// normalization, which would silently fold thousands of runes (CJK
// compatibility ideographs, halfwidth katakana, math alphanumerics) that no
// service is known to substitute.
func foldForMatching(text string) string {
	out := make([]rune, 0, len(text))
	for _, r := range text {
		if unicode.IsLetter(r) {
			out = append(out, foldAccentedChar(r))
		} else {
			out = append(out, r)
		}
	}
	return foldTypography(string(out))
}

// alnumOnly strips XML/HTML-style tags then keeps only lowercase alphanumeric
// characters, folding accents to their base letter. Non-Latin scripts pass
// through one rune each, keeping the reduced rune length in step with the raw
// alnum counts used by advanceByAlnums.
func alnumOnly(text string) string {
	stripped := stripCompleteMarkup(text)
	out := make([]rune, 0, len(stripped))
	for _, r := range stripped {
		if !isAlnum(r) {
			continue
		}
		if unicode.IsLetter(r) {
			out = append(out, foldAccentedChar(r))
		} else {
			out = append(out, unicode.ToLower(r))
		}
	}
	return string(out)
}

// hasAlnum reports whether text holds anything alphanumeric once markup is
// stripped. It is the predicate form of alnumOnly, for the question the callers
// actually ask: is there anything left to speak here? A tag's letters do not
// count, so "<break/>" is empty by this measure.
func hasAlnum(text string) bool { return alnumOnly(text) != "" }

// advanceByAlnums returns the rune position in runes after advancing past n
// alphanumeric characters from startPos. Tags (<...>) are skipped entirely and
// do not count against the budget, so the returned span includes the full tag.
// After the n alnum characters, it advances further past any immediately
// following punctuation, stopping before the next space, alnum char, or tag.
func advanceByAlnums(runes []rune, startPos, n int) int {
	pos := startPos
	count := 0
	for pos < len(runes) && count < n {
		switch {
		case runes[pos] == '<':
			end := indexRune(runes, '>', pos)
			if end != -1 {
				pos = end + 1
			} else {
				pos++
			}
		case isAlnum(runes[pos]):
			count++
			pos++
		default:
			pos++
		}
	}
	for pos < len(runes) {
		if runes[pos] == '<' || isAlnum(runes[pos]) || unicode.IsSpace(runes[pos]) {
			break
		}
		pos++
	}
	return pos
}

// indexRune returns the index of the first r in runes at or after from, or -1.
func indexRune(runes []rune, r rune, from int) int {
	for i := from; i < len(runes); i++ {
		if runes[i] == r {
			return i
		}
	}
	return -1
}
