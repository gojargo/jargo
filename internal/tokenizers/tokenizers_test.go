package tokenizers

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func readGzip(tb testing.TB, name string) []byte {
	tb.Helper()
	f, err := os.Open(filepath.Join("testdata", name)) //nolint:gosec // fixed test data
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		tb.Fatal(err)
	}
	b, err := io.ReadAll(r)
	if err != nil {
		tb.Fatal(err)
	}
	return b
}

// loaded holds the test tokenizers, parsed once: a Tokenizer is immutable, so
// the tests can share them.
var loaded sync.Map //nolint:gochecknoglobals // test cache

func load(tb testing.TB, name string) *Tokenizer {
	tb.Helper()
	if tok, ok := loaded.Load(name); ok {
		t, _ := tok.(*Tokenizer)
		return t
	}
	tok, err := New(readGzip(tb, name+".tokenizer.json.gz"))
	if err != nil {
		tb.Fatal(err)
	}
	loaded.Store(name, tok)
	return tok
}

type fixtureTruncation struct {
	MaxLength int    `json:"max_length"`
	Strategy  string `json:"strategy"`
	Direction string `json:"direction"`
}

type fixtureEncoding struct {
	IDs               []int64  `json:"ids"`
	TypeIDs           []int64  `json:"type_ids"`
	Tokens            []string `json:"tokens"`
	Offsets           [][2]int `json:"offsets"`
	SpecialTokensMask []int    `json:"special_tokens_mask"`
	AttentionMask     []int64  `json:"attention_mask"`
	SequenceIDs       []int    `json:"sequence_ids"`
}

type fixture struct {
	First      string             `json:"first"`
	Second     *string            `json:"second"`
	Truncation *fixtureTruncation `json:"truncation"`
	Encoding   *fixtureEncoding   `json:"encoding"`
	Error      string             `json:"error"`
}

type fixtures struct {
	Version string    `json:"tokenizers_version"`
	Reader  []fixture `json:"reader"`
	Tagger  []fixture `json:"tagger"`
}

func loadFixtures(tb testing.TB) fixtures {
	tb.Helper()
	var f fixtures
	if err := json.Unmarshal(readGzip(tb, "fixtures.json.gz"), &f); err != nil {
		tb.Fatal(err)
	}
	if f.Version != "0.23.3" {
		tb.Fatalf("fixtures come from tokenizers %s", f.Version)
	}
	return f
}

func (f *fixtureTruncation) convert(t *testing.T) *Truncation {
	t.Helper()
	if f == nil {
		return nil
	}
	tr := &Truncation{MaxLength: f.MaxLength}
	switch f.Strategy {
	case "longest_first":
		tr.Strategy = LongestFirst
	case "only_first":
		tr.Strategy = OnlyFirst
	case "only_second":
		tr.Strategy = OnlySecond
	default:
		t.Fatalf("strategy %q", f.Strategy)
	}
	switch f.Direction {
	case "right":
		tr.Direction = Right
	case "left":
		tr.Direction = Left
	default:
		t.Fatalf("direction %q", f.Direction)
	}
	return tr
}

func (f fixture) name() string {
	s := fmt.Sprintf("%q", f.First)
	if f.Second != nil {
		s += fmt.Sprintf(" + %q", *f.Second)
	}
	if f.Truncation != nil {
		s += fmt.Sprintf(" [%s %s %d]", f.Truncation.Strategy, f.Truncation.Direction, f.Truncation.MaxLength)
	}
	return s
}

// diff describes how got differs from want, token by token, or returns "".
func diff(got Encoding, want *fixtureEncoding) string {
	offsets := make([][2]int, len(got.Offsets))
	for i, o := range got.Offsets {
		offsets[i] = [2]int{o.Start, o.End}
	}
	type column struct {
		name      string
		got, want []string
	}
	cols := []column{
		{"id", strs(got.IDs), strs(want.IDs)},
		{"type", strs(got.TypeIDs), strs(want.TypeIDs)},
		{"token", quoted(got.Tokens), quoted(want.Tokens)},
		{"offset", strs(offsets), strs(want.Offsets)},
		{"special", strs(got.SpecialTokensMask), strs(want.SpecialTokensMask)},
		{"attention", strs(got.AttentionMask), strs(want.AttentionMask)},
		{"sequence", strs(got.SequenceIDs), strs(want.SequenceIDs)},
	}
	var b strings.Builder
	n := 0
	for _, c := range cols {
		n = max(n, len(c.got), len(c.want))
	}
	bad := false
	for i := range n {
		line := fmt.Sprintf("%4d", i)
		mismatch := false
		for _, c := range cols {
			g, w := at(c.got, i), at(c.want, i)
			if g == w {
				line += fmt.Sprintf("  %s=%s", c.name, g)
				continue
			}
			mismatch = true
			line += fmt.Sprintf("  %s=%s (want %s)", c.name, g, w)
		}
		if mismatch {
			bad = true
			line = "!" + line[1:]
		}
		b.WriteString(line + "\n")
	}
	if !bad {
		return ""
	}
	return b.String()
}

func at(s []string, i int) string {
	if i < len(s) {
		return s[i]
	}
	return "<none>"
}

func strs[T any](s []T) []string {
	out := make([]string, len(s))
	for i, v := range s {
		out[i] = fmt.Sprint(v)
	}
	return out
}

func quoted(s []string) []string {
	out := make([]string, len(s))
	for i, v := range s {
		out[i] = fmt.Sprintf("%q", v)
	}
	return out
}

func run(tok *Tokenizer, f fixture, tr *Truncation) (Encoding, error) {
	if f.Second != nil {
		return tok.EncodePair(f.First, *f.Second, tr)
	}
	return tok.Encode(f.First, tr)
}

// TestFixtures compares every field of every encoding with what Hugging Face
// tokenizers 0.23.3 produced for it.
func TestFixtures(t *testing.T) {
	all := loadFixtures(t)
	for name, cases := range map[string][]fixture{"reader": all.Reader, "tagger": all.Tagger} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tok := load(t, name)
			failed := 0
			for _, f := range cases {
				got, err := run(tok, f, f.Truncation.convert(t))
				var msg string
				switch {
				case f.Error != "":
					if err == nil || err.Error() != f.Error {
						msg = fmt.Sprintf("error %v, want %q", err, f.Error)
					}
				case err != nil:
					msg = fmt.Sprintf("error %v", err)
				default:
					msg = diff(got, f.Encoding)
				}
				if msg != "" {
					failed++
					if failed <= 20 {
						t.Errorf("%s:\n%s", f.name(), msg)
					}
				}
			}
			t.Logf("%d/%d fixtures match", len(cases)-failed, len(cases))
		})
	}
}

func TestTruncationErrors(t *testing.T) {
	tok := load(t, "tagger")
	_, err := tok.Encode("hello world foo", &Truncation{MaxLength: 3, Strategy: OnlySecond})
	if !errors.Is(err, ErrSecondSequenceNotProvided) {
		t.Fatalf("got %v", err)
	}
	_, err = tok.EncodePair("hello world foo", "bar baz", &Truncation{MaxLength: 4, Strategy: OnlyFirst})
	if !errors.Is(err, ErrSequenceTooShort) {
		t.Fatalf("got %v", err)
	}
	if _, err = tok.Encode("a", &Truncation{MaxLength: -1}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("got %v", err)
	}
	if _, err = tok.Encode("\xff", nil); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("got %v", err)
	}
}

func TestFileTruncation(t *testing.T) {
	if got := load(t, "reader").Truncation(); got == nil || *got != (Truncation{MaxLength: 160}) {
		t.Fatalf("reader truncation %+v", got)
	}
	if got := load(t, "tagger").Truncation(); got != nil {
		t.Fatalf("tagger truncation %+v", got)
	}
}

func TestUnsupported(t *testing.T) {
	var file map[string]any
	if err := json.Unmarshal(readGzip(t, "tagger.tokenizer.json.gz"), &file); err != nil {
		t.Fatal(err)
	}
	// Every option is refused before the vocabulary is read, so a file without
	// one keeps the test fast.
	model, _ := file["model"].(map[string]any)
	model["vocab"] = map[string]any{}
	model["merges"] = []any{}
	base, _ := json.Marshal(file)
	if _, err := New(base); err != nil {
		t.Fatal(err)
	}
	obj := func(v any) map[string]any {
		m, ok := v.(map[string]any)
		if !ok {
			t.Fatalf("%v is not an object", v)
		}
		return m
	}
	for name, edit := range map[string]func(map[string]any){
		"normalizer":  func(f map[string]any) { f["normalizer"] = map[string]any{"type": "NFKC"} },
		"prefixspace": func(f map[string]any) { obj(f["pre_tokenizer"])["add_prefix_space"] = true },
		"dropout":     func(f map[string]any) { obj(f["model"])["dropout"] = 0.1 },
		"stride": func(f map[string]any) {
			f["truncation"] = map[string]any{"max_length": 8, "strategy": "LongestFirst", "stride": 2, "direction": "Right"}
		},
		"singleword": func(f map[string]any) {
			tokens, _ := f["added_tokens"].([]any)
			obj(tokens[0])["single_word"] = true
		},
		"version": func(f map[string]any) { f["version"] = "2.0" },
	} {
		t.Run(name, func(t *testing.T) {
			var f map[string]any
			_ = json.Unmarshal(base, &f)
			edit(f)
			b, _ := json.Marshal(f)
			if _, err := New(b); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("got %v", err)
			}
		})
	}
}

func TestConcurrent(t *testing.T) {
	tok := load(t, "tagger")
	all := loadFixtures(t)
	var wg sync.WaitGroup
	for w := range 8 {
		wg.Go(func() {
			for i := w; i < len(all.Tagger); i += 8 {
				f := all.Tagger[i]
				if f.Error != "" {
					continue
				}
				got, err := run(tok, f, f.Truncation.convert(t))
				if err != nil || diff(got, f.Encoding) != "" {
					t.Errorf("%s differs under concurrency", f.name())
				}
			}
		})
	}
	wg.Wait()
}
