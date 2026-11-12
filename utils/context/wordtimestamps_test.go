package context

import "testing"

func TestMergePunctTokens(t *testing.T) {
	in := []WordTiming{
		{"questions", 1.0}, {", ", 1.2}, {"explain", 1.4},
	}
	got := MergePunctTokens(in)
	want := []WordTiming{{"questions,", 1.0}, {"explain", 1.4}}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Word != want[i].Word || got[i].Offset != want[i].Offset {
			t.Fatalf("token %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestMergePunctTokensDropsLeadingPunct(t *testing.T) {
	got := MergePunctTokens([]WordTiming{{". ", 0.0}, {"Hello", 0.1}})
	if len(got) != 1 || got[0].Word != "Hello" {
		t.Fatalf("got %v, want single 'Hello'", got)
	}
}

func TestMergePunctTokensTrimsWhitespace(t *testing.T) {
	got := MergePunctTokens([]WordTiming{{" Hello ", 0.0}})
	if len(got) != 1 || got[0].Word != "Hello" {
		t.Fatalf("got %v, want trimmed 'Hello'", got)
	}
}

// TestMergePunctTokensKeepsAWordAfterALoneAngleBracket checks a lone '<' is
// not a tag: "<3" has a digit, so it stays a word of its own, while a complete
// tag is still stripped before the check.
func TestMergePunctTokensKeepsAWordAfterALoneAngleBracket(t *testing.T) {
	got := MergePunctTokens([]WordTiming{{"I", 0}, {"<3", 0.5}, {"<break/>", 0.7}})
	if len(got) != 2 || got[0].Word != "I" || got[1].Word != "<3<break/>" {
		t.Errorf("got %+v, want I and <3, the tag merged into the word before it", got)
	}
}
