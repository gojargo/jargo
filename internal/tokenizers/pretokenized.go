package tokenizers

import "unicode/utf8"

// token is a token of a piece, with its byte offsets in that piece's
// normalized text.
type token struct {
	id    int
	value string
	span
}

// split is one piece of a sequence, and its tokens once it has some.
type split struct {
	n         *normalizedString
	tokens    []token
	tokenized bool
}

func asSplits(ns []*normalizedString) []split {
	out := make([]split, len(ns))
	for i, n := range ns {
		out[i] = split{n: n}
	}
	return out
}

// preTokenized is a sequence cut into pieces, each normalized on its own and
// aligned with the sequence it came from.
type preTokenized struct {
	original string
	splits   []split
}

func newPreTokenized(s string) *preTokenized {
	return &preTokenized{original: s, splits: []split{{n: newNormalizedString(s)}}}
}

// split replaces every piece that has no tokens yet by the pieces f cuts it
// into, dropping the empty ones.
func (p *preTokenized) split(f func(n *normalizedString) []split) {
	out := make([]split, 0, len(p.splits))
	for _, s := range p.splits {
		if s.tokenized {
			out = append(out, s)
			continue
		}
		for _, ns := range f(s.n) {
			if ns.n.normalized != "" {
				out = append(out, ns)
			}
		}
	}
	p.splits = out
}

// normalize applies f to every piece that has no tokens yet.
func (p *preTokenized) normalize(f func(n *normalizedString)) {
	for _, s := range p.splits {
		if !s.tokenized {
			f(s.n)
		}
	}
}

// tokenize gives every piece that has no tokens yet the tokens of f.
func (p *preTokenized) tokenize(f func(s string) ([]token, error)) error {
	for i := range p.splits {
		if p.splits[i].tokenized {
			continue
		}
		toks, err := f(p.splits[i].n.normalized)
		if err != nil {
			return err
		}
		p.splits[i].tokens = toks
		p.splits[i].tokenized = true
	}
	return nil
}

// encoding gathers the tokens of every piece, with offsets in characters of
// the original sequence.
func (p *preTokenized) encoding(typeID int64) *encoding {
	e := &encoding{}
	if len(p.splits) == 0 {
		return e
	}
	n := 0
	for _, s := range p.splits {
		n += len(s.tokens)
	}
	e.ids = make([]int64, 0, n)
	e.typeIDs = make([]int64, 0, n)
	e.tokens = make([]string, 0, n)
	e.offsets = make([]Offset, 0, n)
	e.special = make([]int, 0, n)
	e.attention = make([]int64, 0, n)
	conv := newCharOffsets(p.original)
	for _, s := range p.splits {
		base := s.n.offsetsOriginal().start
		for _, t := range s.tokens {
			off := t.span
			if o, ok := s.n.normalizedToOriginal(t.span); ok {
				off = span{base + o.start, base + o.end}
			}
			off = conv.convert(off)
			e.ids = append(e.ids, int64(t.id))
			e.typeIDs = append(e.typeIDs, typeID)
			e.tokens = append(e.tokens, t.value)
			e.offsets = append(e.offsets, Offset{Start: off.start, End: off.end})
			e.special = append(e.special, 0)
			e.attention = append(e.attention, 1)
		}
	}
	return e
}

// charOffsets converts byte offsets of a text to character offsets, as the
// Python binding of Hugging Face tokenizers does: each byte maps to the index
// of the character it belongs to, and an end past the text maps to one past
// the character of its last byte.
type charOffsets struct {
	index []int
}

func newCharOffsets(s string) charOffsets {
	index := make([]int, len(s))
	i := 0
	for b, r := range s {
		for k := range utf8.RuneLen(r) {
			index[b+k] = i
		}
		i++
	}
	return charOffsets{index: index}
}

func (c charOffsets) convert(o span) span {
	n := len(c.index)
	if o.start < 0 || o.start >= n {
		return o
	}
	start := c.index[o.start]
	if o.end >= 0 && o.end < n {
		return span{start, c.index[o.end]}
	}
	last := start + 1
	if o.end-1 >= 0 && o.end-1 < n {
		last = c.index[o.end-1]
	}
	return span{start, last + 1}
}
