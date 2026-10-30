package tokenizers

import (
	"encoding/json"
	"runtime"
	"sync"
	"testing"
)

// probe is what Hugging Face tokenizers 0.23.3 does to every code point (but
// the surrogates) in each step that depends on Unicode data. Runs list
// [first, last, value] for the code points that do not map to themselves (or
// are not of the default class).
type probe struct {
	Version          string      `json:"tokenizers_version"`
	BertNormalizer   []probeRun  `json:"bert_normalizer"`
	NFC              []probeRun  `json:"nfc"`
	BertPreTokenizer []probeRun  `json:"bert_pretokenizer"`
	ByteLevel        []probeRun  `json:"byte_level"`
	NFCOfNFD         [][2]string `json:"nfc_of_nfd"`
}

type probeRun struct {
	First, Last rune
	Value       string
}

func (r *probeRun) UnmarshalJSON(b []byte) error {
	var v [3]any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	first, _ := v[0].(float64)
	last, _ := v[1].(float64)
	value, _ := v[2].(string)
	*r = probeRun{rune(first), rune(last), value}
	return nil
}

// expand returns the value of every code point, def for those no run lists.
func expand(runs []probeRun, def func(rune) string) func(rune) string {
	m := map[rune]string{}
	for _, r := range runs {
		for c := r.First; c <= r.Last; c++ {
			m[c] = r.Value
		}
	}
	return func(c rune) string {
		if v, ok := m[c]; ok {
			return v
		}
		return def(c)
	}
}

// forEachCodePoint calls f with every code point but the surrogates, from
// as many goroutines as there are CPUs.
func forEachCodePoint(f func(rune)) {
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			for c := rune(w); c <= 0x10FFFF; c += rune(workers) {
				if c < 0xD800 || c > 0xDFFF {
					f(c)
				}
			}
		})
	}
	wg.Wait()
}

func normalizeString(n normalizer, s string) string {
	ns := newNormalizedString(s)
	n.normalize(ns)
	return ns.normalized
}

func pieces(p preTokenizer, s string) []string {
	pt := newPreTokenized(s)
	p.preTokenize(pt)
	out := make([]string, len(pt.splits))
	for i, sp := range pt.splits {
		out[i] = sp.n.original
	}
	return out
}

// TestUnicode checks every code point against what Hugging Face tokenizers
// does with it, so that every table this package carries matches the Unicode
// version of the library it reproduces.
func TestUnicode(t *testing.T) {
	var p probe
	if err := json.Unmarshal(readGzip(t, "unicode.json.gz"), &p); err != nil {
		t.Fatal(err)
	}
	if p.Version != "0.23.3" {
		t.Fatalf("probe comes from tokenizers %s", p.Version)
	}
	self := func(c rune) string { return string(c) }
	other := func(rune) string { return "o" }
	bert := bertNormalizer{cleanText: true, handleChineseChars: true, stripAccents: true, lowercase: true}
	wantBert := expand(p.BertNormalizer, self)
	wantNFC := expand(p.NFC, self)
	wantBertClass := expand(p.BertPreTokenizer, other)
	wantByteClass := expand(p.ByteLevel, other)
	byteLevel := newByteLevelPreTokenizer()

	var mu sync.Mutex
	failures := 0
	fail := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		failures++
		if failures <= 30 {
			t.Errorf(format, args...)
		}
	}
	forEachCodePoint(func(c rune) {
		s := string(c)
		if got := normalizeString(bert, s); got != wantBert(c) {
			fail("BertNormalizer U+%04X: %+q, want %+q", c, got, wantBert(c))
		}
		if got := normalizeString(nfcNormalizer{}, s); got != wantNFC(c) {
			fail("NFC U+%04X: %+q, want %+q", c, got, wantNFC(c))
		}
		class := "o"
		switch p := pieces(bertPreTokenizer{}, "a"+s+"a"); {
		case len(p) == 2:
			class = "w"
		case len(p) == 3:
			class = "p"
		}
		if class != wantBertClass(c) {
			fail("BertPreTokenizer U+%04X: class %s, want %s", c, class, wantBertClass(c))
		}
		class = "s"
		switch {
		case len(pieces(byteLevel, "a"+s)) == 1:
			class = "L"
		case len(pieces(byteLevel, "1"+s)) == 1:
			class = "N"
		case len(pieces(byteLevel, "!"+s)) == 1:
			class = "o"
		}
		if class != wantByteClass(c) {
			fail("ByteLevel U+%04X: class %s, want %s", c, class, wantByteClass(c))
		}
	})
	for _, pair := range p.NFCOfNFD {
		if got := normalizeString(nfcNormalizer{}, pair[0]); got != pair[1] {
			fail("NFC %+q: %+q, want %+q", pair[0], got, pair[1])
		}
	}
	if failures > 0 {
		t.Errorf("%d code points differ", failures)
	}
}
