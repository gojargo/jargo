package text

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Part is one piece of text waiting to be joined into a whole by Concatenate.
type Part struct {
	// Text is the piece itself.
	Text string
	// IncludesInterPartSpaces reports whether the spacing that separates this
	// piece from the ones around it is already part of Text. A model streaming a
	// sentence sends " world" with the space; a stream of spoken words sends
	// "world" without one, and the join has to supply it.
	IncludesInterPartSpaces bool
}

// Concatenate joins parts into one string, supplying the spacing the parts do
// not carry themselves.
//
// Text reaching a conversation arrives as pieces of two kinds, and they mix
// within a single turn: what a model streams carries its own spacing, and what a
// synthesizer reports as spoken words does not. Joining them all with a space
// doubles the spaces of the first kind; joining them all without one runs the
// words of the second kind together. So each piece says which it is, a run of
// one kind is joined the way that kind wants, and the boundary between two runs
// gets a space only where neither side already has one.
//
// The logic assumes a run of parts that include their own spacing is
// well-formed: not several separate runs with a space-less boundary between
// them, as in the sequence "hello ", "world.", "goodnight ", "moon" would be.
func Concatenate(parts []Part) string {
	var b strings.Builder
	lastIncludesInterPartSpaces := false

	appendPart := func(p Part) {
		b.WriteString(p.Text)
		lastIncludesInterPartSpaces = p.IncludesInterPartSpaces
	}

	for _, p := range parts {
		// The part is empty. Skip it.
		if p.Text == "" {
			continue
		}
		// Nothing has been written yet. Just append.
		if b.Len() == 0 {
			appendPart(p)
			continue
		}
		switch {
		case p.IncludesInterPartSpaces && lastIncludesInterPartSpaces:
			// Part of an ongoing run that carries its own spacing. Just append.
			appendPart(p)
		case !p.IncludesInterPartSpaces && !lastIncludesInterPartSpaces:
			// Part of an ongoing run that carries none. Space it.
			b.WriteString(" ")
			appendPart(p)
		default:
			// A transition between the two kinds of run. Space it only where
			// neither side brought a space of its own.
			written := b.String()
			if !endsWithSpace(written) && !startsWithSpace(p.Text) {
				b.WriteString(" ")
			}
			appendPart(p)
		}
	}

	return strings.TrimSpace(b.String())
}

// endsWithSpace reports whether s ends in a whitespace character.
func endsWithSpace(s string) bool {
	r, size := utf8.DecodeLastRuneInString(s)
	return size > 0 && unicode.IsSpace(r)
}

// startsWithSpace reports whether s starts with a whitespace character.
func startsWithSpace(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size > 0 && unicode.IsSpace(r)
}
