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

// foldCaseAndAccents lowercases letters and strips accents, preserving every
// other character 1:1 by rune. Unlike normalize it never removes or merges
// characters, so a rune offset computed against the folded text applies
// unchanged to the original.
func foldCaseAndAccents(text string) string {
	out := make([]rune, 0, len(text))
	for _, r := range text {
		if unicode.IsLetter(r) {
			out = append(out, foldAccentedChar(r))
		} else {
			out = append(out, r)
		}
	}
	return string(out)
}

// normalize strips XML/HTML-style tags then keeps only lowercase alphanumeric
// characters, folding accents to their base letter. Non-Latin scripts pass
// through one rune each, keeping the normalized rune length in step with the raw
// alnum counts used by advanceByAlnums.
func normalize(text string) string {
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
