package tokenizers

import (
	"slices"
	"sort"
)

// Canonical decomposition and composition (NFD and NFC), ported from the
// iterators of the unicode-normalization-alignments crate that Hugging Face
// tokenizers uses. Besides each output character they yield the change that
// aligns it with the input, and the alignment of every token offset depends on
// those values, so this follows the crate step by step (including how it
// reorders combining marks and how it decides that a mark is blocked) over its
// own Unicode 9.0.0 tables, rather than using a general-purpose normalizer.

const (
	hangulSBase  = 0xAC00
	hangulLBase  = 0x1100
	hangulVBase  = 0x1161
	hangulTBase  = 0x11A7
	hangulLCount = 19
	hangulVCount = 21
	hangulTCount = 28
	hangulNCount = hangulVCount * hangulTCount
	hangulSCount = hangulLCount * hangulNCount
)

// combiningClassOf is the canonical combining class of r.
func combiningClassOf(r rune) uint8 {
	key := uint32(r)
	i := sort.Search(len(combiningClass), func(i int) bool { return combiningClass[i]>>8 >= key })
	if i < len(combiningClass) && combiningClass[i]>>8 == key {
		return uint8(combiningClass[i] & 0xff)
	}
	return 0
}

// decomposeCanonical calls emit with each character of the full canonical
// decomposition of r.
func decomposeCanonical(r rune, emit func(rune)) {
	if r <= 0x7f {
		emit(r)
		return
	}
	if r >= hangulSBase && r < hangulSBase+hangulSCount {
		s := r - hangulSBase
		emit(hangulLBase + s/hangulNCount)
		emit(hangulVBase + (s%hangulNCount)/hangulTCount)
		if t := s % hangulTCount; t > 0 {
			emit(hangulTBase + t)
		}
		return
	}
	if d, ok := canonicalDecomposition[r]; ok {
		for _, c := range decompositionData[d>>4 : d>>4+d&0xf] {
			emit(c)
		}
		return
	}
	emit(r)
}

// composeCanonical returns the primary composite of a and b.
func composeCanonical(a, b rune) (rune, bool) {
	switch {
	case a >= hangulLBase && a < hangulLBase+hangulLCount && b >= hangulVBase && b < hangulVBase+hangulVCount:
		return hangulSBase + (a-hangulLBase)*hangulNCount + (b-hangulVBase)*hangulTCount, true
	case a >= hangulSBase && a < hangulSBase+hangulSCount && b > hangulTBase && b < hangulTBase+hangulTCount &&
		(a-hangulSBase)%hangulTCount == 0:
		return a + (b - hangulTBase), true
	}
	r, ok := canonicalComposition[uint64(a)<<21|uint64(b)]
	return r, ok
}

type classified struct {
	class uint8
	change
}

// isASCII reports whether s is all ASCII, which both normalization forms
// leave unchanged: no ASCII character decomposes, has a non-zero combining
// class, or composes with another ASCII character.
func isASCII(s string) bool {
	for i := range len(s) {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// nfd returns the canonical decomposition of s with its changes.
func nfd(s string) []change {
	out := make([]change, 0, len(s))
	var buffer []classified
	ready := 0
	sortPending := func() {
		slices.SortStableFunc(buffer[ready:], func(a, b classified) int { return int(a.class) - int(b.class) })
		ready = len(buffer)
	}
	for _, r := range s {
		first := true
		decomposeCanonical(r, func(d rune) {
			class := combiningClassOf(d)
			if class == 0 {
				sortPending()
			}
			n := 1
			if first {
				n = 0
			}
			first = false
			buffer = append(buffer, classified{class, change{d, n}})
		})
		// The crate hands out the characters before ready as soon as there are
		// any, keeping the rest pending.
		for _, c := range buffer[:ready] {
			out = append(out, c.change)
		}
		buffer = append(buffer[:0], buffer[ready:]...)
		ready = 0
	}
	sortPending()
	for _, c := range buffer {
		out = append(out, c.change)
	}
	return out
}

// nfc returns the canonical composition of s with its changes.
func nfc(s string) []change {
	dec := nfd(s)
	c := composer{out: make([]change, 0, len(dec)), lastClass: -1}
	for _, ch := range dec {
		c.push(ch)
	}
	if c.haveComposee {
		c.out = append(c.out, c.composee)
	}
	return append(c.out, c.buffer...)
}

// composer is the crate's recomposition state: the starter characters may
// compose with (composee), and the marks after it that did not compose
// (buffer), the last of which has class lastClass (-1 when there is none).
type composer struct {
	out          []change
	buffer       []change
	composee     change
	haveComposee bool
	lastClass    int
}

func (c *composer) push(ch change) {
	class := int(combiningClassOf(ch.r))
	if !c.haveComposee {
		if class != 0 {
			c.out = append(c.out, ch)
			return
		}
		c.composee, c.haveComposee = ch, true
		return
	}
	k := c.composee
	// A mark is blocked from the starter by an earlier mark of the same or a
	// higher class; a starter after a mark is never composed.
	if c.lastClass >= 0 && c.lastClass >= class {
		if class == 0 {
			c.out = append(c.out, k)
			c.out = append(c.out, c.buffer...)
			c.buffer = c.buffer[:0]
			c.composee = ch
			c.lastClass = -1
			return
		}
		c.buffer = append(c.buffer, ch)
		c.lastClass = class
		return
	}
	if r, ok := composeCanonical(k.r, ch.r); ok {
		c.composee = change{r, k.n + ch.n - 1}
		return
	}
	if c.lastClass < 0 && class == 0 {
		c.composee = ch
		c.out = append(c.out, k)
		return
	}
	c.buffer = append(c.buffer, ch)
	c.lastClass = class
}
