package cartesia

import "testing"

func TestFormatPronunciation(t *testing.T) {
	if got, ok := FormatPronunciation("Metformin", "mɛtˈfɔɹmɪn"); !ok || got != "<<m|ɛ|t|f|ˈ|ɔ|ɹ|m|ɪ|n>>" {
		t.Fatalf("got %q, %v", got, ok)
	}
	a, _ := FormatPronunciation("Achoo", "/əˈʧu/")
	b, _ := FormatPronunciation("Achoo", "ə'tʃu")
	if a != b {
		t.Fatalf("notation variants format differently: %q and %q", a, b)
	}
	if got, _ := FormatPronunciation("Glyburide metformin", "ˈɡlaɪ mɛt"); got != "<<ɡ|l|ˈ|aɪ>> <<m|ɛ|t>>" {
		t.Fatalf("phrase = %q", got)
	}
	if _, ok := FormatPronunciation("Crete", ""); ok {
		t.Fatal("an empty pronunciation was used")
	}
}

func TestBothServicesFormatPronunciationsTheSame(t *testing.T) {
	ws, _ := (&synthesizer{}).FormatPronunciation("Metformin", "mɛtˈfɔɹmɪn")
	http, _ := (&httpSynthesizer{}).FormatPronunciation("Metformin", "mɛtˈfɔɹmɪn")
	if ws != http {
		t.Fatalf("the services differ: %q and %q", ws, http)
	}
}
