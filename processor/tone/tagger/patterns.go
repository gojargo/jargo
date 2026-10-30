package tagger

import (
	"unicode"

	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

// The model was trained on hints found with patterns written for Python's re
// module, whose \w, \d and \b read Unicode and whose case-insensitive letters
// match a few letters beyond ASCII. Go's regexp reads them as ASCII, and has
// no lookaround, so each pattern is matched here by hand, trying its
// alternatives and backtracking in the order re does. Each matcher names the
// pattern it follows.

// isWord is \w: a letter, a number or an underscore.
func isWord(r rune) bool { return r == '_' || shared.IsAlnum(r) }

// isDigit is \d: a decimal digit of any script.
func isDigit(r rune) bool { return unicode.IsDigit(r) }

func isUpperASCII(r rune) bool { return r >= 'A' && r <= 'Z' }

func isDigitASCII(r rune) bool { return r >= '0' && r <= '9' }

func isAlnumASCII(r rune) bool {
	return isUpperASCII(r) || isDigitASCII(r) || (r >= 'a' && r <= 'z')
}

// boundary is \b at i: a word character on one side of it and not the other.
func boundary(s []rune, i int) bool {
	before := i > 0 && isWord(s[i-1])
	after := i < len(s) && isWord(s[i])
	return before != after
}

// details is where the details someone might write down are in s, as
// finditer finds them:
//
//	\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b
//	|(?<![\d$.,])\d(?:[\d\- ]*\d){4,}(?![\d.,]\d)
//	|\b(?:[A-Za-z0-9][ -]){3,}[A-Za-z0-9]\b
//	|[\w.+-]+@[\w-]+(?:\.[\w-]+)+
//
// a code with letters and digits, five or more digits (with dashes or spaces),
// characters spelled out one by one, an email. The matches do not overlap,
// each found from where the one before it ended.
func details(s []rune) [][2]int {
	var found [][2]int
	for p := 0; p < len(s); {
		end, ok := detailAt(s, p)
		if !ok {
			p++
			continue
		}
		found = append(found, [2]int{p, end})
		p = end
	}
	return found
}

// detailAt is the end of the detail starting at p, trying the alternatives in
// order.
func detailAt(s []rune, p int) (int, bool) {
	for _, alternative := range []func([]rune, int) (int, bool){codeAt, digitsAt, spelledAt, emailAt} {
		if end, ok := alternative(s, p); ok {
			return end, true
		}
	}
	return 0, false
}

// codeAt matches \b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b at p.
func codeAt(s []rune, p int) (int, bool) {
	if !boundary(s, p) {
		return 0, false
	}
	run := 0
	for p+run < len(s) && (isUpperASCII(s[p+run]) || isDigitASCII(s[p+run])) {
		run++
	}
	// Each lookahead's [A-Z0-9]* gives back characters until the class after
	// it matches: within the run or, for \d, just past it.
	digit, upper := false, false
	for i := p; i <= p+run && i < len(s); i++ {
		digit = digit || isDigit(s[i])
		upper = upper || (i < p+run && isUpperASCII(s[i]))
	}
	if !digit || !upper {
		return 0, false
	}
	for n := run; n >= 4; n-- {
		if boundary(s, p+n) {
			return p + n, true
		}
	}
	return 0, false
}

// digitsAt matches (?<![\d$.,])\d(?:[\d\- ]*\d){4,}(?![\d.,]\d) at p.
//
// The repeated group backtracks through every way of splitting the digits
// after the first into four or more groups, and the lookahead depends only on
// where the match ends, so the end re settles on is the furthest digit, with
// four or more digits after the first up to it, that the lookahead lets end
// the match.
func digitsAt(s []rune, p int) (int, bool) {
	if p > 0 && (isDigit(s[p-1]) || s[p-1] == '$' || s[p-1] == '.' || s[p-1] == ',') {
		return 0, false
	}
	if p >= len(s) || !isDigit(s[p]) {
		return 0, false
	}
	var digits []int
	for i := p + 1; i < len(s) && (isDigit(s[i]) || s[i] == '-' || s[i] == ' '); i++ {
		if isDigit(s[i]) {
			digits = append(digits, i)
		}
	}
	for k := len(digits) - 1; k >= 3; k-- {
		if end := digits[k] + 1; !numberFollows(s, end) {
			return end, true
		}
	}
	return 0, false
}

// numberFollows is (?=[\d.,]\d) at i.
func numberFollows(s []rune, i int) bool {
	return i+1 < len(s) && (isDigit(s[i]) || s[i] == '.' || s[i] == ',') && isDigit(s[i+1])
}

// spelledAt matches \b(?:[A-Za-z0-9][ -]){3,}[A-Za-z0-9]\b at p.
func spelledAt(s []rune, p int) (int, bool) {
	if !boundary(s, p) {
		return 0, false
	}
	groups := 0
	for p+2*groups+1 < len(s) && isAlnumASCII(s[p+2*groups]) &&
		(s[p+2*groups+1] == ' ' || s[p+2*groups+1] == '-') {
		groups++
	}
	for n := groups; n >= 3; n-- {
		last := p + 2*n
		if last < len(s) && isAlnumASCII(s[last]) && boundary(s, last+1) {
			return last + 1, true
		}
	}
	return 0, false
}

// emailAt matches [\w.+-]+@[\w-]+(?:\.[\w-]+)+ at p.
func emailAt(s []rune, p int) (int, bool) {
	// Each greedy run gives back characters until '@' or '.' follows it, which
	// only its full length can have: a shorter one is followed by one of its
	// own characters.
	for n := runOf(s, p, isLocal); n >= 1; n-- {
		at := p + n
		if at >= len(s) || s[at] != '@' {
			continue
		}
		for h := runOf(s, at+1, isDomain); h >= 1; h-- {
			if end, parts := domainParts(s, at+1+h); parts > 0 {
				return end, true
			}
		}
	}
	return 0, false
}

func isLocal(r rune) bool { return isWord(r) || r == '.' || r == '+' || r == '-' }

func isDomain(r rune) bool { return isWord(r) || r == '-' }

// runOf is how many characters from i are in the class.
func runOf(s []rune, i int, in func(rune) bool) int {
	n := 0
	for i+n < len(s) && in(s[i+n]) {
		n++
	}
	return n
}

// domainParts matches (?:\.[\w-]+)* at i, greedy: where it ends, and how many
// parts it matched.
func domainParts(s []rune, i int) (end, parts int) {
	end = i
	for end+1 < len(s) && s[end] == '.' && isDomain(s[end+1]) {
		end += 1 + runOf(s, end+1, isDomain)
		parts++
	}
	return end, parts
}

// apologizes reports whether \b(sorry|apologi[sz]e), case-insensitive, is
// found in s.
func apologizes(s string) bool {
	return search(s, false, "sorry", "apologise", "apologize") //nolint:misspell // both spellings are matched
}

// asksToSlowOrRepeat reports whether
//
//	\b(slow(er)? down|slower|say (that|it) again|repeat|didn't (catch|get) (that|it)
//	|one more time|come again)\b
//
// case-insensitive, is found in s. A search only asks whether some position and
// alternative match, so the alternatives are spelled out in full.
func asksToSlowOrRepeat(s string) bool {
	return search(s, true,
		"slower down", "slow down", "slower", "say that again", "say it again", "repeat",
		"didn't catch that", "didn't catch it", "didn't get that", "didn't get it",
		"one more time", "come again")
}

// search reports whether one of the phrases, case-insensitive, starts at a word
// boundary of s, and with closed, ends at one.
func search(s string, closed bool, phrases ...string) bool {
	text := []rune(s)
	for p := range text {
		if !boundary(text, p) {
			continue
		}
		for _, phrase := range phrases {
			if end, ok := foldedAt(text, p, phrase); ok && (!closed || boundary(text, end)) {
				return true
			}
		}
	}
	return false
}

// foldedAt matches the lower-case ASCII phrase at p, case-insensitive, and
// returns where it ends.
func foldedAt(s []rune, p int, phrase string) (int, bool) {
	i := p
	for _, c := range phrase {
		if i >= len(s) || !foldEqual(c, s[i]) {
			return 0, false
		}
		i++
	}
	return i, true
}

// foldEqual reports whether r matches the lower-case ASCII c, case-insensitive:
// c in either case and, for i, k and s, the letters whose case maps onto
// them, dotted and dotless I (U+0130, U+0131), the Kelvin sign (U+212A) and
// the long s (U+017F).
func foldEqual(c, r rune) bool {
	if r == c || (c >= 'a' && c <= 'z' && r == c-'a'+'A') {
		return true
	}
	switch c {
	case 'i':
		return r == 'İ' || r == 'ı'
	case 'k':
		return r == 'K'
	case 's':
		return r == 'ſ'
	}
	return false
}

// lower is s lower-cased as Python's str.lower does in the ASCII letters and
// digits it yields, which is all the tagger reads of it: U+0130 lowers to two
// code points, i and a combining dot, where Go's unicode.ToLower gives one.
func lower(s string) []rune {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		if r == 'İ' {
			out = append(out, 'i', '̇')
			continue
		}
		out = append(out, unicode.ToLower(r))
	}
	return out
}

//nolint:gochecknoglobals // a read-only table
var digitWords = map[string]string{
	"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
	"five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

// normalize is the letters and digits of text alone, spoken digits as digits:
// "seven Q two" is "7q2". It joins each run of [a-z0-9] in the lower-cased
// text.
func normalize(text string) string {
	var out []rune
	lowered := lower(text)
	for i := 0; i < len(lowered); {
		if !isLowerAlnumASCII(lowered[i]) {
			i++
			continue
		}
		j := i
		for j < len(lowered) && isLowerAlnumASCII(lowered[j]) {
			j++
		}
		word := string(lowered[i:j])
		if digit, ok := digitWords[word]; ok {
			word = digit
		}
		out = append(out, []rune(word)...)
		i = j
	}
	return string(out)
}

func isLowerAlnumASCII(r rune) bool { return (r >= 'a' && r <= 'z') || isDigitASCII(r) }
