package inworld

import "testing"

func TestFormatPronunciation(t *testing.T) {
	if got, ok := FormatPronunciation("Crete", "[kriːt]"); !ok || got != "/kriːt/" {
		t.Fatalf("got %q, %v", got, ok)
	}
	a, _ := FormatPronunciation("Achoo", "/əˈʧu/")
	b, _ := FormatPronunciation("Achoo", "ə'tʃu")
	if a != b {
		t.Fatalf("notation variants format differently: %q and %q", a, b)
	}
	if _, ok := FormatPronunciation("Glyburide metformin", "ˈɡlaɪ mɛt"); ok {
		t.Fatal("a pronunciation spanning several words was used: one word per pair of slashes")
	}
	if _, ok := FormatPronunciation("Crete", ""); ok {
		t.Fatal("an empty pronunciation was used")
	}
	if got, _ := (&synthesizer{}).FormatPronunciation("Crete", "kriːt"); got != "/kriːt/" {
		t.Fatalf("the service formats %q", got)
	}
}
