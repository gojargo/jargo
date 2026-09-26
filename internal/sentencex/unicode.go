package sentencex

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The segmentation rules were written against Rust's definitions of the
// Unicode character properties and of its string operations. The helpers in
// this file reproduce those definitions on top of Go's unicode tables, so the
// rest of the package can keep the rules exactly as written.

// isAlphabetic reports the Unicode Alphabetic property: Lu, Ll, Lt, Lm, Lo, Nl
// and Other_Alphabetic, with Other_Lowercase and Other_Uppercase folded in as
// the derived Lowercase and Uppercase properties require.
func isAlphabetic(r rune) bool {
	return unicode.IsLetter(r) ||
		unicode.Is(unicode.Nl, r) ||
		unicode.Is(unicode.Other_Alphabetic, r) ||
		unicode.Is(unicode.Other_Lowercase, r) ||
		unicode.Is(unicode.Other_Uppercase, r)
}

// isAlphanumeric reports whether r is Alphabetic or a number (Nd, Nl, No).
func isAlphanumeric(r rune) bool {
	return isAlphabetic(r) || unicode.IsNumber(r)
}

// isUppercase reports the Unicode Uppercase property: Lu and Other_Uppercase.
func isUppercase(r rune) bool {
	return unicode.Is(unicode.Lu, r) || unicode.Is(unicode.Other_Uppercase, r)
}

// isLowercase reports the Unicode Lowercase property: Ll and Other_Lowercase.
func isLowercase(r rune) bool {
	return unicode.Is(unicode.Ll, r) || unicode.Is(unicode.Other_Lowercase, r)
}

// isWhitespace reports the Unicode White_Space property, which is exactly the
// set unicode.IsSpace tests.
func isWhitespace(r rune) bool {
	return unicode.IsSpace(r)
}

// isWordChar reports whether r is a word character in the Unicode sense a
// regular expression's \w uses: Alphabetic, a mark, a decimal digit, a
// connector punctuation or a join control.
func isWordChar(r rune) bool {
	return isAlphabetic(r) ||
		unicode.IsMark(r) ||
		unicode.Is(unicode.Nd, r) ||
		unicode.Is(unicode.Pc, r) ||
		unicode.Is(unicode.Join_Control, r)
}

// isASCIIUpper reports whether r is an ASCII uppercase letter.
func isASCIIUpper(r rune) bool {
	return r >= 'A' && r <= 'Z'
}

// isASCIIDigit reports whether r is an ASCII digit.
func isASCIIDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

// isASCII reports whether every byte of s is ASCII.
func isASCII(s string) bool {
	for i := range len(s) {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// trimStart removes leading White_Space.
func trimStart(s string) string {
	return strings.TrimLeftFunc(s, isWhitespace)
}

// trimEnd removes trailing White_Space.
func trimEnd(s string) string {
	return strings.TrimRightFunc(s, isWhitespace)
}

// firstRune returns the first character of s, and false when s is empty.
func firstRune(s string) (rune, bool) {
	if s == "" {
		return 0, false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return r, true
}

// lastRune returns the last character of s, and false when s is empty.
func lastRune(s string) (rune, bool) {
	if s == "" {
		return 0, false
	}
	r, _ := utf8.DecodeLastRuneInString(s)
	return r, true
}

// hasSecondRune reports whether s holds at least two characters.
func hasSecondRune(s string) bool {
	if s == "" {
		return false
	}
	_, size := utf8.DecodeRuneInString(s)
	return size < len(s)
}

// lastPieceAfter returns the text after the last character that sep accepts,
// or all of s when there is none. It is the first item of a reverse split.
func lastPieceAfter(s string, sep func(rune) bool) string {
	i := strings.LastIndexFunc(s, sep)
	if i < 0 {
		return s
	}
	_, size := utf8.DecodeRuneInString(s[i:])
	return s[i+size:]
}

// matchIndices returns the byte offsets of the non-overlapping occurrences of
// token in s, scanning from the left. token must not be empty.
func matchIndices(s, token string) []int {
	var out []int
	for from := 0; from <= len(s); {
		i := strings.Index(s[from:], token)
		if i < 0 {
			break
		}
		out = append(out, from+i)
		from += i + len(token)
	}
	return out
}

// lines splits s into lines. A line ends at "\n" or "\r\n", and the final line
// ending is optional, so a trailing newline yields no empty last line. A bare
// "\r" that does not precede "\n" is kept.
func lines(s string) []string {
	var out []string
	for s != "" {
		i := strings.IndexByte(s, '\n')
		if i < 0 {
			out = append(out, s)
			break
		}
		out = append(out, strings.TrimSuffix(s[:i], "\r"))
		s = s[i+1:]
	}
	return out
}

// partitionPoint returns the index of the first element for which pred is
// false, assuming the elements for which it holds all come first. It runs the
// exact probe sequence of a fixed-iteration binary search, so an input that is
// not partitioned still yields the same, deterministic, index.
func partitionPoint(n int, pred func(i int) bool) int {
	if n == 0 {
		return 0
	}
	base, size := 0, n
	for size > 1 {
		half := size / 2
		mid := base + half
		if pred(mid) {
			base = mid
		}
		size -= half
	}
	if pred(base) {
		return base + 1
	}
	return base
}

// toLowercase returns s with every character mapped to its full lowercase
// form, including the one multi-character mapping (U+0130) and the
// word-final form of the Greek capital sigma.
func toLowercase(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i, r := range s {
		switch r {
		case '\u03A3':
			b.WriteRune(lowercaseSigma(s, i))
		case '\u0130':
			b.WriteString("i\u0307")
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// lowercaseSigma lowers the capital sigma at byte offset i of s to the final
// form when it ends a word (Final_Sigma: a cased letter before it and none
// after it, skipping case-ignorable characters on both sides).
func lowercaseSigma(s string, i int) rune {
	before := false
	for j := i; j > 0; {
		r, size := utf8.DecodeLastRuneInString(s[:j])
		j -= size
		if !isCaseIgnorable(r) {
			before = isCased(r)
			break
		}
	}
	after := false
	for j := i + len("\u03A3"); j < len(s); {
		r, size := utf8.DecodeRuneInString(s[j:])
		j += size
		if !isCaseIgnorable(r) {
			after = isCased(r)
			break
		}
	}
	if before && !after {
		return '\u03C2'
	}
	return '\u03C3'
}

// isCased reports the Unicode Cased property.
func isCased(r rune) bool {
	return isLowercase(r) || isUppercase(r) || unicode.Is(unicode.Lt, r)
}

// isCaseIgnorable reports the Unicode Case_Ignorable property: Mn, Me, Cf, Lm
// and Sk, plus the characters whose Word_Break value is MidLetter, MidNumLet
// or Single_Quote.
func isCaseIgnorable(r rune) bool {
	switch r {
	case '\'', '.', ':', '\u00B7', '\u0387', '\u055F', '\u05F4', '\u2018', '\u2019',
		'\u2024', '\u2027', '\uFE13', '\uFE52', '\uFE55', '\uFF07', '\uFF0E', '\uFF1A':
		return true
	}
	return unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf, unicode.Lm, unicode.Sk)
}

// charToUppercase returns the full uppercase form of r, which is more than one
// character for the entries of specialUppercase.
func charToUppercase(r rune) string {
	if s, ok := specialUppercase[r]; ok {
		return s
	}
	return string(unicode.ToUpper(r))
}

// specialUppercase holds the unconditional multi-character uppercase mappings
// of SpecialCasing.txt.
//
//nolint:gochecknoglobals // fixed lookup table
var specialUppercase = map[rune]string{
	0x00DF: "\u0053\u0053",
	0x0149: "\u02BC\u004E",
	0x01F0: "\u004A\u030C",
	0x0390: "\u0399\u0308\u0301",
	0x03B0: "\u03A5\u0308\u0301",
	0x0587: "\u0535\u0552",
	0x1E96: "\u0048\u0331",
	0x1E97: "\u0054\u0308",
	0x1E98: "\u0057\u030A",
	0x1E99: "\u0059\u030A",
	0x1E9A: "\u0041\u02BE",
	0x1F50: "\u03A5\u0313",
	0x1F52: "\u03A5\u0313\u0300",
	0x1F54: "\u03A5\u0313\u0301",
	0x1F56: "\u03A5\u0313\u0342",
	0x1F80: "\u1F08\u0399",
	0x1F81: "\u1F09\u0399",
	0x1F82: "\u1F0A\u0399",
	0x1F83: "\u1F0B\u0399",
	0x1F84: "\u1F0C\u0399",
	0x1F85: "\u1F0D\u0399",
	0x1F86: "\u1F0E\u0399",
	0x1F87: "\u1F0F\u0399",
	0x1F88: "\u1F08\u0399",
	0x1F89: "\u1F09\u0399",
	0x1F8A: "\u1F0A\u0399",
	0x1F8B: "\u1F0B\u0399",
	0x1F8C: "\u1F0C\u0399",
	0x1F8D: "\u1F0D\u0399",
	0x1F8E: "\u1F0E\u0399",
	0x1F8F: "\u1F0F\u0399",
	0x1F90: "\u1F28\u0399",
	0x1F91: "\u1F29\u0399",
	0x1F92: "\u1F2A\u0399",
	0x1F93: "\u1F2B\u0399",
	0x1F94: "\u1F2C\u0399",
	0x1F95: "\u1F2D\u0399",
	0x1F96: "\u1F2E\u0399",
	0x1F97: "\u1F2F\u0399",
	0x1F98: "\u1F28\u0399",
	0x1F99: "\u1F29\u0399",
	0x1F9A: "\u1F2A\u0399",
	0x1F9B: "\u1F2B\u0399",
	0x1F9C: "\u1F2C\u0399",
	0x1F9D: "\u1F2D\u0399",
	0x1F9E: "\u1F2E\u0399",
	0x1F9F: "\u1F2F\u0399",
	0x1FA0: "\u1F68\u0399",
	0x1FA1: "\u1F69\u0399",
	0x1FA2: "\u1F6A\u0399",
	0x1FA3: "\u1F6B\u0399",
	0x1FA4: "\u1F6C\u0399",
	0x1FA5: "\u1F6D\u0399",
	0x1FA6: "\u1F6E\u0399",
	0x1FA7: "\u1F6F\u0399",
	0x1FA8: "\u1F68\u0399",
	0x1FA9: "\u1F69\u0399",
	0x1FAA: "\u1F6A\u0399",
	0x1FAB: "\u1F6B\u0399",
	0x1FAC: "\u1F6C\u0399",
	0x1FAD: "\u1F6D\u0399",
	0x1FAE: "\u1F6E\u0399",
	0x1FAF: "\u1F6F\u0399",
	0x1FB2: "\u1FBA\u0399",
	0x1FB3: "\u0391\u0399",
	0x1FB4: "\u0386\u0399",
	0x1FB6: "\u0391\u0342",
	0x1FB7: "\u0391\u0342\u0399",
	0x1FBC: "\u0391\u0399",
	0x1FC2: "\u1FCA\u0399",
	0x1FC3: "\u0397\u0399",
	0x1FC4: "\u0389\u0399",
	0x1FC6: "\u0397\u0342",
	0x1FC7: "\u0397\u0342\u0399",
	0x1FCC: "\u0397\u0399",
	0x1FD2: "\u0399\u0308\u0300",
	0x1FD3: "\u0399\u0308\u0301",
	0x1FD6: "\u0399\u0342",
	0x1FD7: "\u0399\u0308\u0342",
	0x1FE2: "\u03A5\u0308\u0300",
	0x1FE3: "\u03A5\u0308\u0301",
	0x1FE4: "\u03A1\u0313",
	0x1FE6: "\u03A5\u0342",
	0x1FE7: "\u03A5\u0308\u0342",
	0x1FF2: "\u1FFA\u0399",
	0x1FF3: "\u03A9\u0399",
	0x1FF4: "\u038F\u0399",
	0x1FF6: "\u03A9\u0342",
	0x1FF7: "\u03A9\u0342\u0399",
	0x1FFC: "\u03A9\u0399",
	0xFB00: "\u0046\u0046",
	0xFB01: "\u0046\u0049",
	0xFB02: "\u0046\u004C",
	0xFB03: "\u0046\u0046\u0049",
	0xFB04: "\u0046\u0046\u004C",
	0xFB05: "\u0053\u0054",
	0xFB06: "\u0053\u0054",
	0xFB13: "\u0544\u0546",
	0xFB14: "\u0544\u0535",
	0xFB15: "\u0544\u053B",
	0xFB16: "\u054E\u0546",
	0xFB17: "\u0544\u053D",
}

// perlWordClass is the body of a regular expression character class holding
// exactly the characters isWordChar accepts: the Unicode meaning of \w, which
// Go's \w (ASCII only) does not have.
//
//nolint:gochecknoglobals // built once
var perlWordClass = `\p{L}\p{M}\p{Nd}\p{Pc}\p{Nl}` +
	rangeTableClass(unicode.Join_Control, unicode.Other_Alphabetic, unicode.Other_Lowercase, unicode.Other_Uppercase)

// rangeTableClass writes the characters of the tables as character class
// ranges.
func rangeTableClass(tables ...*unicode.RangeTable) string {
	var b strings.Builder
	add := func(lo, hi, stride uint32) {
		if stride == 1 {
			fmt.Fprintf(&b, `\x{%X}-\x{%X}`, lo, hi)
			return
		}
		for r := lo; r <= hi; r += stride {
			fmt.Fprintf(&b, `\x{%X}`, r)
		}
	}
	for _, t := range tables {
		for _, r := range t.R16 {
			add(uint32(r.Lo), uint32(r.Hi), uint32(r.Stride))
		}
		for _, r := range t.R32 {
			add(r.Lo, r.Hi, r.Stride)
		}
	}
	return b.String()
}
