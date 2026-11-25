package align

import (
	"slices"
	"testing"
)

func TestMergePunctTokens(t *testing.T) {
	in := []WordTiming{
		{Word: "questions", Offset: 1.0}, {Word: ", ", Offset: 1.2}, {Word: "explain", Offset: 1.4},
	}
	got := MergePunctTokens(in)
	want := []WordTiming{{Word: "questions,", Offset: 1.0}, {Word: "explain", Offset: 1.4}}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Word != want[i].Word || got[i].Offset != want[i].Offset {
			t.Fatalf("token %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestMergePunctTokensMergedWordEndsWhereItsPunctuationEnds checks a word
// keeps its own start and takes the end of the punctuation merged into it.
func TestMergePunctTokensMergedWordEndsWhereItsPunctuationEnds(t *testing.T) {
	got := MergePunctTokens([]WordTiming{
		{Word: "hello", Offset: 0.0, End: 0.4},
		{Word: ", ", Offset: 0.4, End: 0.5},
		{Word: "world", Offset: 0.6, End: 1.0},
	})
	want := []WordTiming{{Word: "hello,", Offset: 0.0, End: 0.5}, {Word: "world", Offset: 0.6, End: 1.0}}
	if !slices.Equal(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestMergePunctTokensDropsLeadingPunct(t *testing.T) {
	got := MergePunctTokens([]WordTiming{{Word: ". ", Offset: 0.0}, {Word: "Hello", Offset: 0.1}})
	if len(got) != 1 || got[0].Word != "Hello" {
		t.Fatalf("got %v, want single 'Hello'", got)
	}
}

func TestMergePunctTokensTrimsWhitespace(t *testing.T) {
	got := MergePunctTokens([]WordTiming{{Word: " Hello ", Offset: 0.0}})
	if len(got) != 1 || got[0].Word != "Hello" {
		t.Fatalf("got %v, want trimmed 'Hello'", got)
	}
}

// TestMergePunctTokensKeepsAWordAfterALoneAngleBracket checks a lone '<' is
// not a tag: "<3" has a digit, so it stays a word of its own, while a complete
// tag is still stripped before the check.
func TestMergePunctTokensKeepsAWordAfterALoneAngleBracket(t *testing.T) {
	got := MergePunctTokens([]WordTiming{
		{Word: "I", Offset: 0}, {Word: "<3", Offset: 0.5}, {Word: "<break/>", Offset: 0.7},
	})
	if len(got) != 2 || got[0].Word != "I" || got[1].Word != "<3<break/>" {
		t.Errorf("got %+v, want I and <3, the tag merged into the word before it", got)
	}
}
