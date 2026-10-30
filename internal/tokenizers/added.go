package tokenizers

import (
	"fmt"
	"unicode/utf8"
)

// addedToken is a token of the added vocabulary: matched in the text before
// the model sees it, and given its id directly.
type addedToken struct {
	content string
	lstrip  bool
	rstrip  bool
	// normalized tokens are matched in the normalized text, the others in
	// the raw text.
	normalized bool
	special    bool
}

// trie matches a set of byte strings, leftmost first and longest at that
// position.
type trie struct {
	nodes []trieNode
	// root maps a first byte to its node, 0 for none.
	root [256]int32
}

type trieNode struct {
	next map[byte]int
	id   int // -1 when no pattern ends here
}

func newTrie() *trie {
	return &trie{nodes: []trieNode{{next: map[byte]int{}, id: -1}}}
}

func (t *trie) add(pattern string, id int) error {
	if pattern == "" {
		return fmt.Errorf("%w: empty added token pattern", ErrUnsupported)
	}
	n := 0
	for i := range len(pattern) {
		c := pattern[i]
		m, ok := t.nodes[n].next[c]
		if !ok {
			m = len(t.nodes)
			t.nodes = append(t.nodes, trieNode{next: map[byte]int{}, id: -1})
			t.nodes[n].next[c] = m
			if n == 0 {
				t.root[c] = int32(m)
			}
		}
		n = m
	}
	if t.nodes[n].id >= 0 {
		return fmt.Errorf("%w: two added tokens match %q", ErrUnsupported, pattern)
	}
	t.nodes[n].id = id
	return nil
}

// longestAt returns the id and end of the longest pattern starting at byte i
// of s.
func (t *trie) longestAt(s string, i int) (int, int, bool) {
	n := int(t.root[s[i]])
	if n == 0 {
		return -1, 0, false
	}
	id, end := t.nodes[n].id, i+1
	for j := i + 1; j < len(s); j++ {
		m, ok := t.nodes[n].next[s[j]]
		if !ok {
			break
		}
		n = m
		if t.nodes[n].id >= 0 {
			id, end = t.nodes[n].id, j+1
		}
	}
	return id, end, id >= 0
}

// addedVocabulary is the added vocabulary of a tokenizer.
type addedVocabulary struct {
	byID map[int]addedToken
	// raw matches the tokens that are not normalized, in the raw text;
	// normalized matches the others, in the normalized text. Either may be
	// nil.
	raw        *trie
	normalized *trie
}

// newAddedVocabulary gives the tokens their ids as Hugging Face tokenizers
// does when it loads them: the id the model or an earlier added token has for
// the same content, otherwise the next one after the model's vocabulary.
// Normalized tokens are matched by their normalized form.
func newAddedVocabulary(
	tokens []addedToken, modelID func(string) (int, bool), vocabSize int, norm normalizer,
) (*addedVocabulary, error) {
	v := &addedVocabulary{byID: map[int]addedToken{}}
	byContent := map[string]int{}
	normalizedForm := map[int]string{}
	nextID := vocabSize
	for _, t := range tokens {
		if t.content == "" {
			continue
		}
		if id, ok := byContent[t.content]; ok && v.byID[id] == t {
			continue
		}
		id, ok := byContent[t.content]
		if !ok {
			id, ok = modelID(t.content)
		}
		if !ok {
			id = nextID
			nextID++
		}
		if t.normalized && norm != nil {
			n := newNormalizedString(t.content)
			norm.normalize(n)
			if n.normalized != t.content {
				normalizedForm[id] = n.normalized
			}
		}
		byContent[t.content] = id
		v.byID[id] = t
	}
	for id, t := range v.byID {
		pattern, tr := t.content, &v.raw
		if t.normalized {
			tr = &v.normalized
			if f, ok := normalizedForm[id]; ok {
				pattern = f
			}
		}
		if *tr == nil {
			*tr = newTrie()
		}
		if err := (*tr).add(pattern, id); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// addedMatch is a piece of a sentence: an added token when id >= 0, or the
// text between two of them.
type addedMatch struct {
	id int
	span
}

// findMatches cuts sentence around the added tokens tr matches, extending
// each match over the whitespace its token strips.
func (v *addedVocabulary) findMatches(sentence string, tr *trie) []addedMatch {
	if sentence == "" {
		return []addedMatch{{-1, span{0, 0}}}
	}
	if tr == nil {
		return []addedMatch{{-1, span{0, len(sentence)}}}
	}
	var out []addedMatch
	startOffset := 0
	for i := 0; i < len(sentence); {
		id, stop, ok := tr.longestAt(sentence, i)
		if !ok {
			i++
			continue
		}
		start := i
		i = stop
		t := v.byID[id]
		if t.lstrip {
			start = max(trailingSpaceStart(sentence[:start]), startOffset)
		}
		if t.rstrip {
			stop += leadingSpaceEnd(sentence[stop:])
		}
		if startOffset < start {
			out = append(out, addedMatch{-1, span{startOffset, start}})
		}
		out = append(out, addedMatch{id, span{start, stop}})
		startOffset = stop
	}
	if startOffset != len(sentence) {
		out = append(out, addedMatch{-1, span{startOffset, len(sentence)}})
	}
	return out
}

// trailingSpaceStart returns where the whitespace that ends s starts.
func trailingSpaceStart(s string) int {
	for len(s) > 0 {
		c, n := utf8.DecodeLastRuneInString(s)
		if !isWhitespace(c) {
			break
		}
		s = s[:len(s)-n]
	}
	return len(s)
}

// leadingSpaceEnd returns where the whitespace that starts s ends.
func leadingSpaceEnd(s string) int {
	i := 0
	for i < len(s) {
		c, n := utf8.DecodeRuneInString(s[i:])
		if !isWhitespace(c) {
			break
		}
		i += n
	}
	return i
}

// splitWithIndices cuts n around the added tokens tr matches; the pieces
// holding one get its token.
func (v *addedVocabulary) splitWithIndices(n *normalizedString, tr *trie) []split {
	matches := v.findMatches(n.normalized, tr)
	out := make([]split, 0, len(matches))
	for _, m := range matches {
		s, ok := n.slice(m.span)
		if !ok {
			panicf("bad added-token split %v of %q", m.span, n.normalized)
		}
		if m.id < 0 {
			out = append(out, split{n: s})
			continue
		}
		tok := token{id: m.id, value: s.normalized, span: span{0, len(s.normalized)}}
		out = append(out, split{n: s, tokenized: true, tokens: []token{tok}})
	}
	return out
}

// extractAndNormalize cuts the raw-text added tokens out of sequence, then
// normalizes the rest and cuts out the normalized added tokens.
func (v *addedVocabulary) extractAndNormalize(norm normalizer, sequence string) *preTokenized {
	p := newPreTokenized(sequence)
	p.split(func(n *normalizedString) []split { return v.splitWithIndices(n, v.raw) })
	p.split(func(n *normalizedString) []split {
		if norm != nil {
			norm.normalize(n)
		}
		return v.splitWithIndices(n, v.normalized)
	})
	return p
}
