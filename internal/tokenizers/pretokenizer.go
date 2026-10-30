package tokenizers

import (
	"unicode"
	"unicode/utf8"
)

// preTokenizer cuts the normalized pieces of a sequence into words.
type preTokenizer interface {
	preTokenize(p *preTokenized)
}

// bertPreTokenizer is the BertPreTokenizer: it drops whitespace and isolates
// every punctuation character.
type bertPreTokenizer struct{}

func (bertPreTokenizer) preTokenize(p *preTokenized) {
	p.split(func(n *normalizedString) []split {
		return asSplits(n.split(matchRunes(n.normalized, isWhitespace), splitRemoved))
	})
	p.split(func(n *normalizedString) []split {
		return asSplits(n.split(matchRunes(n.normalized, isBertPunctuation), splitIsolated))
	})
}

func isBertPunctuation(c rune) bool {
	if c < 0x80 {
		return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~')
	}
	return unicode.Is(categoryPunctuation, c)
}

// byteLevelPreTokenizer is the ByteLevel pre-tokenizer without a prefix space
// and with its regular expression: it splits the text as GPT-2 does, then
// rewrites every byte of each word as the printable character standing for
// it.
type byteLevelPreTokenizer struct {
	// byteChar maps each byte to the character standing for it.
	byteChar [256]rune
}

func newByteLevelPreTokenizer() *byteLevelPreTokenizer {
	b := &byteLevelPreTokenizer{}
	printable := func(c int) bool {
		return (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF)
	}
	n := 0
	for c := range 256 {
		if printable(c) {
			b.byteChar[c] = rune(c)
			continue
		}
		b.byteChar[c] = rune(256 + n)
		n++
	}
	return b
}

func (b *byteLevelPreTokenizer) preTokenize(p *preTokenized) {
	p.split(func(n *normalizedString) []split {
		return asSplits(n.split(gpt2Matches(n.normalized), splitIsolated))
	})
	p.normalize(func(n *normalizedString) {
		s := n.normalized
		dest := make([]change, 0, len(s))
		for i := range len(s) {
			c := 0
			if !utf8.RuneStart(s[i]) {
				c = 1
			}
			dest = append(dest, change{b.byteChar[s[i]], c})
		}
		n.transform(dest, 0)
	})
}

// gpt2Matches splits s as Oniguruma does with the GPT-2 pattern
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Every character belongs to one of the classes, so the matches cover s.
func gpt2Matches(s string) []match {
	if s == "" {
		return []match{{span{0, 0}, false}}
	}
	var out []match
	for i := 0; i < len(s); {
		e := gpt2MatchAt(s, i)
		out = append(out, match{span{i, e}, true})
		i = e
	}
	return out
}

type gpt2Class int

const (
	gpt2Other gpt2Class = iota
	gpt2Letter
	gpt2Number
	gpt2Space
)

func gpt2ClassOf(c rune) gpt2Class {
	switch {
	case unicode.Is(onigLetter, c):
		return gpt2Letter
	case unicode.Is(onigNumber, c):
		return gpt2Number
	case unicode.Is(onigSpace, c):
		return gpt2Space
	}
	return gpt2Other
}

// classAt returns the class of the character at byte i of s and its end.
func classAt(s string, i int) (gpt2Class, int) {
	c, n := utf8.DecodeRuneInString(s[i:])
	return gpt2ClassOf(c), i + n
}

// runOf returns the end of the run of characters of class k starting at i.
func runOf(s string, i int, k gpt2Class) int {
	for i < len(s) {
		c, e := classAt(s, i)
		if c != k {
			break
		}
		i = e
	}
	return i
}

// gpt2MatchAt returns the end of the match starting at byte i of s, trying
// the alternatives in order as a backtracking engine does.
func gpt2MatchAt(s string, i int) int {
	if s[i] == '\'' {
		for _, c := range [...]string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if len(s)-i-1 >= len(c) && s[i+1:i+1+len(c)] == c {
				return i + 1 + len(c)
			}
		}
	}
	first, firstEnd := classAt(s, i)
	// " ?\p{L}+", " ?\p{N}+" and " ?[^\s\p{L}\p{N}]+": with the optional
	// space first, then without it.
	for _, k := range [...]gpt2Class{gpt2Letter, gpt2Number, gpt2Other} {
		if s[i] == ' ' && firstEnd < len(s) {
			if c, e := classAt(s, firstEnd); c == k {
				return runOf(s, e, k)
			}
		}
		if first == k {
			return runOf(s, firstEnd, k)
		}
	}
	// The character is whitespace. "\s+(?!\S)" takes the run up to the end of
	// the text, or the run but its last character when that character comes
	// before a non-space; "\s+" takes a lone space before a non-space.
	e := runOf(s, firstEnd, gpt2Space)
	if e == len(s) {
		return e
	}
	_, last := utf8.DecodeLastRuneInString(s[:e])
	if e-last > i {
		return e - last
	}
	return e
}
