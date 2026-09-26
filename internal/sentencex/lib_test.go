package sentencex

import (
	"slices"
	"strings"
	"testing"
)

func TestChunkTextBasic(t *testing.T) {
	text := "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
	chunks := chunkText(text, 20)
	want := []textChunk{
		{offset: 0, text: "First paragraph."},
		{offset: 18, text: "Second paragraph."},
		{offset: 37, text: "Third paragraph."},
	}
	if !slices.Equal(chunks, want) {
		t.Fatalf("chunks = %+v, want %+v", chunks, want)
	}
}

func TestChunkTextNoParagraphBreaks(t *testing.T) {
	// A text with no \n\n but longer than the chunk size comes back as a
	// single chunk, never split mid-word or mid-sentence.
	text := "This is a long text without paragraph breaks that should be returned as one chunk."
	chunks := chunkText(text, 20)
	want := []textChunk{{offset: 0, text: text}}
	if !slices.Equal(chunks, want) {
		t.Fatalf("chunks = %+v, want %+v", chunks, want)
	}
}

func TestSegmentNoWordSplitAtChunkBoundary(t *testing.T) {
	// A text over 10 KiB with no \n\n, where "Christopher" straddles the
	// 10240-byte boundary ("Chri" before it, "stopher." after).
	text := strings.Repeat("a", chunkSize-4) + "Christopher."
	if len(text) <= chunkSize {
		t.Fatal("test input must exceed the chunk size")
	}
	if text[chunkSize-4:chunkSize] != "Chri" {
		t.Fatal("the split must fall inside Christopher")
	}

	sentences := Segment("en", text)
	for _, s := range sentences {
		if strings.HasSuffix(trimEnd(s), "Chri") {
			t.Errorf("Christopher was split: a sentence ends with Chri: %q", s)
		}
		if strings.HasPrefix(trimStart(s), "stopher") {
			t.Errorf("Christopher was split: a sentence starts with stopher: %q", s)
		}
	}
	if !slices.ContainsFunc(sentences, func(s string) bool { return strings.Contains(s, "Christopher") }) {
		t.Errorf("expected Christopher whole in a sentence, got %q", sentences)
	}
}

func TestSegmentAutomaticChunking(t *testing.T) {
	small := "First sentence. Second sentence.\n\nThird sentence. Fourth sentence."
	large := strings.Repeat(small, 10000)

	result := Segment("en", large)
	perRepetition := Segment("en", small)

	if len(result) < len(perRepetition)*9000 {
		t.Errorf("got %d sentences, want at least %d", len(result), len(perRepetition)*9000)
	}

	if smallResult := Segment("en", small); !slices.Equal(smallResult, perRepetition) {
		t.Errorf("small text = %q, want %q", smallResult, perRepetition)
	}
}

func TestSegmentWithMultibyteCharacters(t *testing.T) {
	text := "日本語です。中文文章。"
	sentences := Segment("en", text)

	if len(sentences) == 0 {
		t.Fatal("expected at least one sentence")
	}
	if got := strings.Join(sentences, ""); got != text {
		t.Errorf("reconstruction = %q, want %q", got, text)
	}
}

func TestSegmentQuoteFollowedBySpacedDotsDoesNotPanic(t *testing.T) {
	// A quote followed by spaced dots once made the quote extension push a
	// later boundary backwards.
	text := "\"x.\" . ."
	if got := Segment("en", text); !slices.Equal(got, []string{text}) {
		t.Errorf("Segment = %q, want %q", got, []string{text})
	}
}

func TestSegmentExample(t *testing.T) {
	got := Segment("en", "Hello world. This is a test.")
	want := []string{"Hello world. ", "This is a test."}
	if !slices.Equal(got, want) {
		t.Errorf("Segment = %q, want %q", got, want)
	}
}

func TestSegmentInitialBeforeNonASCIISpace(t *testing.T) {
	// A single capital before a non-ASCII space and a period reads as a name
	// initial. (The upstream implementation panics here, slicing the text
	// inside the multi-byte space.)
	for _, text := range []string{"R\u3000.", "R\u00A0. Next one.", "A B\u2028. c"} {
		if got := strings.Join(Segment("en", text), ""); got != text {
			t.Errorf("reconstruction of %q = %q", text, got)
		}
	}
}

func TestLanguageFactoryFallbacks(t *testing.T) {
	tests := []struct {
		code string
		want string
	}{
		{"en", "en"},
		{"de-at", "de"},
		{"als", "de"},
		{"mr", "mr"},
		{"yi", "en"},
		{"zh", "en"},
		{"xx", "en"},
		{"", "en"},
	}
	for _, tt := range tests {
		if got, want := languageFactory(tt.code), languages[tt.want](); got != want {
			t.Errorf("languageFactory(%q) did not resolve to %q", tt.code, tt.want)
		}
	}
}
