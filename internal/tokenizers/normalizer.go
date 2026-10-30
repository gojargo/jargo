package tokenizers

import (
	"unicode"
)

// normalizer rewrites a piece of text in place, keeping its alignment.
type normalizer interface {
	normalize(n *normalizedString)
}

// bertNormalizer is the BertNormalizer: control characters removed, every
// whitespace turned into a space, spaces around CJK ideographs, accents
// stripped and the text lowercased, each step optional.
type bertNormalizer struct {
	cleanText          bool
	handleChineseChars bool
	stripAccents       bool
	lowercase          bool
}

func (b bertNormalizer) normalize(n *normalizedString) {
	if b.cleanText {
		n.filter(func(c rune) bool { return c != 0 && c != 0xFFFD && !isBertControl(c) })
		n.mapRunes(func(c rune) rune {
			if isBertWhitespace(c) {
				return ' '
			}
			return c
		})
	}
	if b.handleChineseChars {
		dest := make([]change, 0, len(n.normalized))
		for _, c := range n.normalized {
			if isChineseChar(c) {
				dest = append(dest, change{' ', 0}, change{c, 1}, change{' ', 1})
			} else {
				dest = append(dest, change{c, 0})
			}
		}
		n.transform(dest, 0)
	}
	if b.stripAccents && !isASCII(n.normalized) {
		n.transform(nfd(n.normalized), 0)
		n.filter(func(c rune) bool { return !unicode.Is(categoryMarkNonspacing, c) })
	}
	if b.lowercase {
		lowercase(n)
	}
}

// nfcNormalizer is the NFC normalizer.
type nfcNormalizer struct{}

func (nfcNormalizer) normalize(n *normalizedString) {
	if isASCII(n.normalized) {
		return
	}
	n.transform(nfc(n.normalized), 0)
}

// isWhitespace is Rust's char::is_whitespace: the White_Space property.
func isWhitespace(c rune) bool {
	return unicode.Is(onigSpace, c)
}

func isBertWhitespace(c rune) bool {
	return c == '\t' || c == '\n' || c == '\r' || isWhitespace(c)
}

func isBertControl(c rune) bool {
	if c == '\t' || c == '\n' || c == '\r' {
		return false
	}
	return unicode.Is(categoryOther, c)
}

func isChineseChar(c rune) bool {
	switch {
	case c >= 0x4E00 && c <= 0x9FFF,
		c >= 0x3400 && c <= 0x4DBF,
		c >= 0x20000 && c <= 0x2A6DF,
		c >= 0x2A700 && c <= 0x2B73F,
		c >= 0x2B740 && c <= 0x2B81F,
		c >= 0x2B920 && c <= 0x2CEAF,
		c >= 0xF900 && c <= 0xFAFF,
		c >= 0x2F800 && c <= 0x2FA1F:
		return true
	}
	return false
}

// lowercase applies Rust's char::to_lowercase to every character. Its only
// mapping to more than one character is U+0130, which becomes "i" and U+0307;
// the inserted character is aligned with the one it comes from.
func lowercase(n *normalizedString) {
	dest := make([]change, 0, len(n.normalized))
	for _, c := range n.normalized {
		if c == 0x130 {
			dest = append(dest, change{'i', 0}, change{0x307, 1})
			continue
		}
		dest = append(dest, change{toLower(c), 0})
	}
	n.transform(dest, 0)
}

// toLower is the simple lowercase mapping Rust's char::to_lowercase uses for
// every character but U+0130. Go's unicode tables (Unicode 17.0.0) give the
// same mapping for every code point as the Rust release Hugging Face
// tokenizers 0.23.3 was built with; TestUnicode checks it.
func toLower(c rune) rune {
	if c >= 'A' && c <= 'Z' {
		return c + 'a' - 'A'
	}
	if c < 0x80 {
		return c
	}
	return unicode.ToLower(c)
}
