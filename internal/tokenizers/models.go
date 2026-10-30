package tokenizers

import (
	"container/heap"
	"sync"
	"unicode/utf8"
)

// model turns one pre-tokenized word into tokens.
type model interface {
	tokenize(s string) ([]token, error)
	tokenID(s string) (int, bool)
	vocabSize() int
}

// wordPiece is the WordPiece model: the longest vocabulary entry from the
// start of the word, then the longest continuation, the whole word becoming
// the unknown token when some part of it matches nothing.
type wordPiece struct {
	vocab         map[string]int
	unk           string
	unkID         int
	prefix        string
	maxInputChars int
}

func (w *wordPiece) tokenID(s string) (int, bool) {
	id, ok := w.vocab[s]
	return id, ok
}

func (w *wordPiece) vocabSize() int { return len(w.vocab) }

func (w *wordPiece) tokenize(s string) ([]token, error) {
	if utf8.RuneCountInString(s) > w.maxInputChars {
		return []token{{id: w.unkID, value: w.unk, span: span{0, len(s)}}}, nil
	}
	var out []token
	for start := 0; start < len(s); {
		end := len(s)
		found := false
		for start < end {
			sub := s[start:end]
			if start > 0 {
				sub = w.prefix + sub
			}
			if id, ok := w.vocab[sub]; ok {
				out = append(out, token{id: id, value: sub, span: span{start, end}})
				found = true
				break
			}
			_, n := utf8.DecodeLastRuneInString(s[start:end])
			end -= n
		}
		if !found {
			return []token{{id: w.unkID, value: w.unk, span: span{0, len(s)}}}, nil
		}
		start = end
	}
	return out, nil
}

// bpeCacheCapacity and bpeCacheMaxLength bound the word cache as Hugging Face
// tokenizers bounds its own.
const (
	bpeCacheCapacity  = 10000
	bpeCacheMaxLength = 256
)

// bpe is the BPE model without an unknown token, dropout, subword prefix or
// suffix: every character of the word is a symbol, and the pair with the
// lowest merge rank is merged until none is left.
type bpe struct {
	vocab  map[string]int
	vocabR []string
	merges map[[2]int]bpeMerge

	mu    sync.RWMutex
	cache map[string][]token
}

type bpeMerge struct{ rank, id int }

func (b *bpe) tokenID(s string) (int, bool) {
	id, ok := b.vocab[s]
	return id, ok
}

func (b *bpe) vocabSize() int { return len(b.vocab) }

// tokenize returns the tokens of word s. The slice may be shared with the
// cache: callers only read it.
func (b *bpe) tokenize(s string) ([]token, error) {
	if s == "" {
		return nil, nil
	}
	b.mu.RLock()
	hit, ok := b.cache[s]
	b.mu.RUnlock()
	if ok {
		return hit, nil
	}
	out := b.mergeWord(s)
	if len(s) < bpeCacheMaxLength {
		b.mu.Lock()
		if len(b.cache) < bpeCacheCapacity {
			b.cache[s] = out
		}
		b.mu.Unlock()
	}
	return out, nil
}

// symbol is a token of a word being merged, linked to its neighbors. A
// symbol merged into the one before it has a zero length.
type symbol struct {
	c          int
	prev, next int
	len        int
}

// mergeWord splits w into characters and merges them.
func (b *bpe) mergeWord(w string) []token {
	syms := make([]symbol, 0, len(w))
	for i, r := range w {
		n := utf8.RuneLen(r)
		id, ok := b.vocab[w[i:i+n]]
		if !ok {
			// Without an unknown token, a character outside the vocabulary
			// produces nothing.
			continue
		}
		prev := len(syms) - 1
		if prev >= 0 {
			syms[prev].next = len(syms)
		}
		syms = append(syms, symbol{c: id, prev: prev, next: -1, len: n})
	}

	b.mergeAll(syms)

	out := make([]token, 0, len(syms))
	pos := 0
	for _, s := range syms {
		if s.len == 0 {
			continue
		}
		out = append(out, token{id: s.c, value: b.vocabR[s.c], span: span{pos, pos + s.len}})
		pos += s.len
	}
	return out
}

// mergeAll applies the merges to syms, lowest rank first and leftmost first
// among equal ranks, marking the absorbed symbols with a zero length.
func (b *bpe) mergeAll(syms []symbol) {
	q := &mergeQueue{}
	for i := 0; i+1 < len(syms); i++ {
		if m, ok := b.merges[[2]int{syms[i].c, syms[i+1].c}]; ok {
			*q = append(*q, queued{pos: i, rank: m.rank, id: m.id})
		}
	}
	heap.Init(q)
	for q.Len() > 0 {
		top, _ := heap.Pop(q).(queued)
		cur := &syms[top.pos]
		if cur.len == 0 || cur.next == -1 {
			continue
		}
		nextPos := cur.next
		right := syms[nextPos]
		if m, ok := b.merges[[2]int{cur.c, right.c}]; !ok || m.id != top.id {
			continue
		}
		cur.c = top.id
		cur.len += right.len
		cur.next = right.next
		syms[nextPos].len = 0
		if right.next > -1 && right.next < len(syms) {
			syms[right.next].prev = top.pos
		}
		if cur.prev >= 0 {
			if m, ok := b.merges[[2]int{syms[cur.prev].c, cur.c}]; ok {
				heap.Push(q, queued{pos: cur.prev, rank: m.rank, id: m.id})
			}
		}
		if cur.next >= 0 && cur.next < len(syms) {
			if m, ok := b.merges[[2]int{cur.c, syms[cur.next].c}]; ok {
				heap.Push(q, queued{pos: top.pos, rank: m.rank, id: m.id})
			}
		}
	}
}

// queued is a candidate merge, ordered by rank, then by position.
type queued struct{ pos, rank, id int }

type mergeQueue []queued

func (q mergeQueue) Len() int { return len(q) }
func (q mergeQueue) Less(i, j int) bool {
	if q[i].rank != q[j].rank {
		return q[i].rank < q[j].rank
	}
	return q[i].pos < q[j].pos
}
func (q mergeQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *mergeQueue) Push(x any) {
	v, _ := x.(queued)
	*q = append(*q, v)
}

func (q *mergeQueue) Pop() any {
	old := *q
	v := old[len(old)-1]
	*q = old[:len(old)-1]
	return v
}
