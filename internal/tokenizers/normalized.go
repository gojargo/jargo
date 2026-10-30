package tokenizers

import (
	"strings"
	"unicode/utf8"
)

// span is a half-open byte range [start, end).
type span struct{ start, end int }

// change is one character of a transformation and how it relates to the
// characters it replaces, as Hugging Face tokenizers describes it: 0 replaces
// the next character, a positive value inserts a new character, and -n
// replaces the next character and removes the n after it.
type change struct {
	r rune
	n int
}

// normalizedString is a piece of text being normalized, with the alignment
// that maps every byte of the normalized text back to a byte range of the
// original. Offsets reported for tokens are computed through it, so every
// transformation updates the alignment exactly as Hugging Face tokenizers does.
type normalizedString struct {
	original   string
	normalized string
	// alignments holds, for each byte of normalized, the original byte range
	// of the character it came from.
	alignments []span
	// shift is the offset of original in the text it was sliced from.
	shift int
}

func newNormalizedString(s string) *normalizedString {
	al := make([]span, 0, len(s))
	for b, r := range s {
		n := utf8.RuneLen(r)
		for range n {
			al = append(al, span{b, b + n})
		}
	}
	return &normalizedString{original: s, normalized: s, alignments: al}
}

// offsetsOriginal is the range this piece covers in the text it was sliced
// from.
func (n *normalizedString) offsetsOriginal() span {
	return span{n.shift, n.shift + len(n.original)}
}

// originalToNormalized converts a range of original bytes to the range of
// normalized bytes it produced.
func (n *normalizedString) originalToNormalized(t span) (span, bool) {
	if t.start == t.end {
		return t, true
	}
	if t.start > t.end {
		return span{}, false
	}
	start, end := -1, -1
	for i, a := range n.alignments {
		if t.end < a.end {
			break
		}
		if start < 0 && t.start <= a.start && a.start != a.end {
			start = i
		}
		end = i + 1
	}
	switch {
	case start >= 0 && end >= 0:
		return span{start, end}, true
	case start >= 0:
		return span{start, start}, true
	case end >= 0:
		return span{end, end}, true
	default:
		return span{}, false
	}
}

// normalizedToOriginal converts a range of normalized bytes to the range of
// original bytes it came from.
func (n *normalizedString) normalizedToOriginal(t span) (span, bool) {
	if t.start == t.end {
		return t, true
	}
	if t.start > t.end || t.end > len(n.alignments) {
		return span{}, false
	}
	return span{n.alignments[t.start].start, n.alignments[t.end-1].end}, true
}

// transform replaces the whole normalized text by dest. initialOffset is the
// number of characters removed before the first one of dest.
func (n *normalizedString) transform(dest []change, initialOffset int) {
	r, ok := n.originalToNormalized(span{0, len(n.original)})
	if !ok {
		return
	}
	n.transformRange(r, dest, initialOffset)
}

// transformRange replaces the normalized bytes in r by dest, aligning each new
// character with the one it replaces, or with the one before it when it is
// inserted.
func (n *normalizedString) transformRange(r span, dest []change, initialOffset int) {
	if initialOffset == 0 && isIdentity(n.normalized[r.start:r.end], dest) {
		// Every character replaces itself, and all the bytes of a character
		// share one alignment, so nothing changes.
		return
	}
	replaced := []rune(n.normalized[r.start:r.end])
	next := 0
	initialRemoved := 0
	for ; next < len(replaced) && next < initialOffset; next++ {
		initialRemoved += utf8.RuneLen(replaced[next])
	}
	offset := initialRemoved + r.start

	var sb strings.Builder
	sb.Grow(len(dest))
	alignments := make([]span, 0, r.end-r.start)
	for _, d := range dest {
		var align span
		switch {
		case d.n <= 0:
			align = n.alignments[offset]
		case offset >= 1:
			align = n.alignments[offset-1]
		}
		if d.n <= 0 && next < len(replaced) {
			offset += utf8.RuneLen(replaced[next])
			next++
		}
		for k := 0; k < -d.n && next < len(replaced); k++ {
			offset += utf8.RuneLen(replaced[next])
			next++
		}
		for range utf8.RuneLen(d.r) {
			alignments = append(alignments, align)
		}
		sb.WriteRune(d.r)
	}

	al := make([]span, 0, len(n.alignments)-(r.end-r.start)+len(alignments))
	al = append(al, n.alignments[:r.start]...)
	al = append(al, alignments...)
	al = append(al, n.alignments[r.end:]...)
	n.alignments = al
	n.normalized = n.normalized[:r.start] + sb.String() + n.normalized[r.end:]
}

// isIdentity reports whether dest replaces each character of s by itself.
func isIdentity(s string, dest []change) bool {
	i := 0
	for _, c := range s {
		if i >= len(dest) || dest[i].n != 0 || dest[i].r != c {
			return false
		}
		i++
	}
	return i == len(dest)
}

// filter removes the characters keep rejects.
func (n *normalizedString) filter(keep func(rune) bool) {
	removed := 0
	removedStart := 0
	dest := make([]change, 0, len(n.normalized))
	var last rune
	haveLast := false
	for _, c := range n.normalized {
		if !keep(c) {
			removed++
			continue
		}
		if haveLast {
			dest = append(dest, change{last, -removed})
		} else {
			removedStart = removed
		}
		last, haveLast = c, true
		removed = 0
	}
	if haveLast {
		dest = append(dest, change{last, -removed})
	}
	n.transform(dest, removedStart)
}

// mapRunes replaces every character by f of it.
func (n *normalizedString) mapRunes(f func(rune) rune) {
	dest := make([]change, 0, len(n.normalized))
	for _, c := range n.normalized {
		dest = append(dest, change{f(c), 0})
	}
	n.transform(dest, 0)
}

// slice returns the piece covering the normalized bytes in r, or false when r
// does not fall on character boundaries.
func (n *normalizedString) slice(r span) (*normalizedString, bool) {
	if r.start < 0 || r.start > r.end || r.end > len(n.normalized) ||
		!isBoundary(n.normalized, r.start) || !isBoundary(n.normalized, r.end) {
		return nil, false
	}
	o, ok := n.normalizedToOriginal(r)
	if !ok {
		return nil, false
	}
	original := ""
	if o.start <= o.end && o.end <= len(n.original) && isBoundary(n.original, o.start) && isBoundary(n.original, o.end) {
		original = n.original[o.start:o.end]
	}
	al := make([]span, r.end-r.start)
	for i, a := range n.alignments[r.start:r.end] {
		al[i] = span{a.start - o.start, a.end - o.start}
	}
	return &normalizedString{
		original:   original,
		normalized: n.normalized[r.start:r.end],
		alignments: al,
		shift:      n.shift + o.start,
	}, true
}

// isBoundary reports whether byte i of s starts a character or ends s.
func isBoundary(s string, i int) bool {
	return i == len(s) || (i < len(s) && utf8.RuneStart(s[i]))
}

// match is one piece of a pattern search: a match of the pattern, or the text
// between two matches.
type match struct {
	span

	isMatch bool
}

// matchRunes splits s around every character pred accepts, each such
// character being a match of its own.
func matchRunes(s string, pred func(rune) bool) []match {
	if s == "" {
		return []match{{span{0, 0}, false}}
	}
	var out []match
	last := 0
	for b, c := range s {
		if !pred(c) {
			continue
		}
		if last < b {
			out = append(out, match{span{last, b}, false})
		}
		e := b + utf8.RuneLen(c)
		out = append(out, match{span{b, e}, true})
		last = e
	}
	if last < len(s) {
		out = append(out, match{span{last, len(s)}, false})
	}
	return out
}

// splitBehavior says what a split does with the matches of its pattern.
type splitBehavior int

const (
	// splitRemoved drops the matches.
	splitRemoved splitBehavior = iota
	// splitIsolated keeps every match as a piece of its own.
	splitIsolated
)

// split cuts the normalized text along matches.
func (n *normalizedString) split(matches []match, behavior splitBehavior) []*normalizedString {
	out := make([]*normalizedString, 0, len(matches))
	for _, m := range matches {
		if behavior == splitRemoved && m.isMatch {
			continue
		}
		s, ok := n.slice(m.span)
		if !ok {
			panicf("bad split %v of %q", m.span, n.normalized)
		}
		out = append(out, s)
	}
	return out
}
