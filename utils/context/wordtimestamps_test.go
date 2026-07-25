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
