package tokenizers

// encoding is an encoding being built: one sequence, or the merge of the
// template's pieces.
type encoding struct {
	ids       []int64
	typeIDs   []int64
	tokens    []string
	offsets   []Offset
	special   []int
	attention []int64
	// sequences maps a sequence id to the token range it covers.
	sequences map[int]span
}

func (e *encoding) len() int { return len(e.ids) }

func (e *encoding) setSequenceID(id int) {
	if e.sequences == nil {
		e.sequences = map[int]span{}
	}
	e.sequences[id] = span{0, e.len()}
}

func (e *encoding) setTypeIDs(id int64) {
	e.typeIDs = make([]int64, e.len())
	for i := range e.typeIDs {
		e.typeIDs[i] = id
	}
}

// truncate keeps maxLen tokens, from the start (Right) or the end (Left).
// The overflowing tokens are dropped and so is the sequence information, as
// Hugging Face tokenizers drops it.
func (e *encoding) truncate(maxLen int, d Direction) {
	if maxLen >= e.len() {
		return
	}
	if maxLen == 0 {
		*e = encoding{}
		return
	}
	start, stop := 0, maxLen
	if d == Left {
		start, stop = e.len()-maxLen, e.len()
	}
	*e = encoding{
		ids:       e.ids[start:stop],
		typeIDs:   e.typeIDs[start:stop],
		tokens:    e.tokens[start:stop],
		offsets:   e.offsets[start:stop],
		special:   e.special[start:stop],
		attention: e.attention[start:stop],
	}
}

// mergeWith appends o, shifting its sequence ranges.
func (e *encoding) mergeWith(o *encoding) {
	n := e.len()
	for id, r := range o.sequences {
		if e.sequences == nil {
			e.sequences = map[int]span{}
		}
		e.sequences[id] = span{n + r.start, n + r.end}
	}
	e.ids = append(e.ids, o.ids...)
	e.typeIDs = append(e.typeIDs, o.typeIDs...)
	e.tokens = append(e.tokens, o.tokens...)
	e.offsets = append(e.offsets, o.offsets...)
	e.special = append(e.special, o.special...)
	e.attention = append(e.attention, o.attention...)
}

// sequenceIDs returns the sequence of each token, -1 for the tokens outside
// every sequence.
func (e *encoding) sequenceIDs() []int {
	out := make([]int, e.len())
	for i := range out {
		out[i] = -1
	}
	n := len(e.sequences)
	if n == 0 {
		n = 1
	}
	for id := range n {
		r, ok := e.sequences[id]
		if !ok {
			r = span{0, e.len()}
		}
		for i := r.start; i < r.end; i++ {
			out[i] = id
		}
	}
	return out
}

func (e *encoding) export() Encoding {
	return Encoding{
		IDs:               e.ids,
		TypeIDs:           e.typeIDs,
		AttentionMask:     e.attention,
		Tokens:            e.tokens,
		Offsets:           e.offsets,
		SpecialTokensMask: e.special,
		SequenceIDs:       e.sequenceIDs(),
	}
}

// truncateEncodings fits the two encodings (pair may be nil) in maxLength
// tokens with the given strategy.
func truncateEncodings(enc, pair *encoding, maxLength int, s Strategy, d Direction) error {
	if maxLength == 0 {
		enc.truncate(0, d)
		if pair != nil {
			pair.truncate(0, d)
		}
		return nil
	}
	total := enc.len()
	if pair != nil {
		total += pair.len()
	}
	if total <= maxLength {
		return nil
	}
	toRemove := total - maxLength

	switch s {
	case OnlyFirst, OnlySecond:
		target := enc
		if s == OnlySecond {
			if pair == nil {
				return &truncationError{ErrSecondSequenceNotProvided, msgSecondSequenceNotProvided}
			}
			target = pair
		}
		if target.len() <= toRemove {
			return &truncationError{ErrSequenceTooShort, msgSequenceTooShort}
		}
		target.truncate(target.len()-toRemove, d)
	default:
		if pair == nil {
			enc.truncate(total-toRemove, d)
			return nil
		}
		// n1 is the shorter length, n2 the longer one's share.
		swap := enc.len() > pair.len()
		n1 := min(enc.len(), pair.len())
		n2 := n1
		if n1 <= maxLength {
			n2 = max(n1, maxLength-n1)
		}
		if n1+n2 > maxLength {
			n1 = maxLength / 2
			n2 = n1 + maxLength%2
		}
		if swap {
			n1, n2 = n2, n1
		}
		enc.truncate(n1, d)
		pair.truncate(n2, d)
	}
	return nil
}

// The messages Hugging Face tokenizers gives the truncation failures.
const (
	msgSecondSequenceNotProvided = "Truncation error: Second sequence not provided"
	msgSequenceTooShort          = "Truncation error: Sequence to truncate too short to respect the provided max_length"
)

// truncationError carries the message Hugging Face tokenizers gives a
// truncation failure, and unwraps to the matching sentinel.
type truncationError struct {
	sentinel error
	msg      string
}

func (e *truncationError) Error() string { return e.msg }
func (e *truncationError) Unwrap() error { return e.sentinel }

// piece is one element of a template: a sequence (seq is 0 for A, 1 for B) or
// a special token.
type piece struct {
	seq     int
	special *specialToken
	typeID  int64
}

type specialToken struct {
	ids    []int64
	tokens []string
}

// template is the TemplateProcessing post-processor.
type template struct {
	single, pair []piece
}

func addedCount(pieces []piece) int {
	n := 0
	for _, p := range pieces {
		if p.special != nil {
			n += len(p.special.ids)
		}
	}
	return n
}

func (t *template) added(isPair bool) int {
	if isPair {
		return addedCount(t.pair)
	}
	return addedCount(t.single)
}

// process lays the encodings out along the template.
func (t *template) process(encs []*encoding) *encoding {
	for i, e := range encs {
		e.setSequenceID(i)
		e.setTypeIDs(int64(i))
	}
	pieces := t.single
	if len(encs) == 2 {
		pieces = t.pair
	}
	n := addedCount(pieces)
	for _, e := range encs {
		n += e.len()
	}
	out := &encoding{
		ids:       make([]int64, 0, n),
		typeIDs:   make([]int64, 0, n),
		tokens:    make([]string, 0, n),
		offsets:   make([]Offset, 0, n),
		special:   make([]int, 0, n),
		attention: make([]int64, 0, n),
	}
	for _, p := range pieces {
		if p.special == nil {
			e := encs[p.seq]
			e.setTypeIDs(p.typeID)
			e.setSequenceID(p.seq)
			out.mergeWith(e)
			continue
		}
		n := len(p.special.ids)
		s := &encoding{
			ids:       p.special.ids,
			typeIDs:   make([]int64, n),
			tokens:    p.special.tokens,
			offsets:   make([]Offset, n),
			special:   make([]int, n),
			attention: make([]int64, n),
		}
		for i := range n {
			s.typeIDs[i] = p.typeID
			s.special[i] = 1
			s.attention[i] = 1
		}
		out.mergeWith(s)
	}
	return out
}
