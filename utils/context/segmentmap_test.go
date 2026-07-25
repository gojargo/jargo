package context

import "testing"

func TestGetOpcodesReplaceSpan(t *testing.T) {
	orig := tokenizeWS("Your balance is $42.50")
	tts := tokenizeWS("Your balance is forty two dollars and fifty cents")
	codes := getOpcodes(orig, tts)
	// Expect a leading equal span then a replace covering the currency.
	if len(codes) != 2 {
		t.Fatalf("opcodes = %d (%v), want 2", len(codes), codes)
	}
	if codes[0].tag != "equal" || codes[1].tag != "replace" {
		t.Fatalf("tags = %q,%q, want equal,replace", codes[0].tag, codes[1].tag)
	}
}

func TestSegmentTransformedClassification(t *testing.T) {
	cases := []struct {
		name     string
		tts      string
		orig     string
		wantXfrm bool
	}{
		{"identical", "hello world", "hello world", false},
		{"currency", "forty two dollars", "$42", true},
		{"acronym-spacing", "A P I", "API", true},
		{"markup-same-word", "<spell>SQL</spell>", "SQL", true},
		{"case-only-not-transformed", "hello", "hello", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			segs := buildSegments(c.tts, c.orig)
			if len(segs) != 1 {
				t.Fatalf("segments = %d, want 1 for a single replace/equal", len(segs))
			}
			if got := segs[0].isTransformed(); got != c.wantXfrm {
				t.Fatalf("isTransformed = %v, want %v", got, c.wantXfrm)
			}
		})
	}
}

func TestSegmentMapUnchangedCursorTracksOriginal(t *testing.T) {
	m := NewTextSegmentMap("Hello world", "Hello world", "")
	m.AdvanceWord("Hello")
	if got := string(m.origRunes[:m.UserFacingPos()]); got != "Hello" {
		t.Fatalf("after 'Hello' user-facing prefix = %q, want %q", got, "Hello")
	}
	m.AdvanceWord("world")
	if !m.IsComplete() {
		t.Fatal("map not complete after both words")
	}
}

func TestSegmentMapOverflowCarriesSuffix(t *testing.T) {
	// A single token that runs past the end of the TTS text reports the suffix as
	// overflow.
	m := NewTextSegmentMap("Hello", "Hello", "")
	m.AdvanceWord("HelloThere")
	if got := m.LastOverflow(); got != "There" {
		t.Fatalf("overflow = %q, want %q", got, "There")
	}
}
