package tagger

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"
	"unicode"
	"unicode/utf8"
)

// TestThePatternsMatchAsPythonsReDoes matches the patterns against what
// Python 3.14's re found in the same strings: thousands of codes, digits with
// dashes, spaces, dots, commas and dollars, spelled-out letters, emails, the
// phrases with their case variants, and Unicode digits and letters. The
// strings leave out the code points Go's Unicode tables (15.0) and Python's
// (16.0) class differently.
func TestThePatternsMatchAsPythonsReDoes(t *testing.T) {
	f, err := os.Open("testdata/patterns.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		S         string   `json:"s"`
		Details   [][2]int `json:"d"`
		Apology   bool     `json:"a"`
		Repeat    bool     `json:"r"`
		Normalize string   `json:"n"`
	}
	if err := json.NewDecoder(r).Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 5000 {
		t.Fatalf("%d cases", len(cases))
	}
	failed := 0
	for _, c := range cases {
		got := details([]rune(c.S))
		ok := (len(got) == 0 && len(c.Details) == 0) || reflect.DeepEqual(got, c.Details)
		ok = ok && apologizes(c.S) == c.Apology && asksToSlowOrRepeat(c.S) == c.Repeat && normalize(c.S) == c.Normalize
		if !ok {
			failed++
			if failed <= 20 {
				t.Errorf("%q: details %v (want %v), apology %v (want %v), repeat %v (want %v), normalized %q (want %q)",
					c.S, got, c.Details, apologizes(c.S), c.Apology, asksToSlowOrRepeat(c.S), c.Repeat,
					normalize(c.S), c.Normalize)
			}
		}
	}
	if failed > 0 {
		t.Errorf("%d of %d strings disagree", failed, len(cases))
	}
}

// Python, over every code point: re.fullmatch(letter, c, re.IGNORECASE) holds
// for the letter in either case and, beyond them, only for İ and ı with i, the
// Kelvin sign with k and the long s with s.
func TestCaseInsensitiveLettersMatchAsPythonsReDoes(t *testing.T) {
	beyond := map[rune][]rune{'i': {'İ', 'ı'}, 'k': {'K'}, 's': {'ſ'}}
	for c := 'a'; c <= 'z'; c++ {
		want := append([]rune{c - 'a' + 'A', c}, beyond[c]...)
		var got []rune
		for r := rune(0); r <= unicode.MaxRune; r++ {
			if utf8.ValidRune(r) && foldEqual(c, r) {
				got = append(got, r)
			}
		}
		slices.Sort(want)
		if !slices.Equal(got, want) {
			t.Errorf("%c matches %q, want %q", c, got, want)
		}
	}
}

// Python, over every code point: c.lower() holds an ASCII letter or digit only
// for the ASCII letters and digits, U+0130 (lowered to "i" and U+0307) and the
// Kelvin sign (lowered to "k").
func TestLowerYieldsTheASCIIPythonsLowerDoes(t *testing.T) {
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		var want string
		switch {
		case r >= 'A' && r <= 'Z':
			want = string(r - 'A' + 'a')
		case isLowerAlnumASCII(r):
			want = string(r)
		case r == 'İ':
			want = "i̇"
		case r == 'K':
			want = "k"
		}
		got := lower(string(r))
		ascii := slices.ContainsFunc(got, isLowerAlnumASCII)
		if (want == "") == ascii || (want != "" && string(got) != want) {
			t.Errorf("lower(%U) = %q, want %q", r, string(got), want)
		}
	}
}
