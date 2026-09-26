package elevenlabs

import (
	"net/url"
	"testing"
)

func TestFormatPronunciation(t *testing.T) {
	cases := map[string]struct{ word, ipa, want string }{
		"one word": {"Allegra", "/əˈlɛgrə/", `<phoneme alphabet="ipa" ph="əˈlɛɡrə">Allegra</phoneme>`},
		"a phrase gets a tag per word": {
			"Zovirax tablets", "ˈzoʊvəˌræks ˈtæbləts",
			`<phoneme alphabet="ipa" ph="ˈzoʊvəˌræks">Zovirax</phoneme> ` +
				`<phoneme alphabet="ipa" ph="ˈtæbləts">tablets</phoneme>`,
		},
		"markup is escaped": {"A&B", "eɪ", `<phoneme alphabet="ipa" ph="eɪ">A&amp;B</phoneme>`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got, ok := FormatPronunciation(tc.word, tc.ipa); !ok || got != tc.want {
				t.Fatalf("got %q, %v; want %q", got, ok, tc.want)
			}
		})
	}
	if _, ok := FormatPronunciation("Zovirax tablets", "ˈzoʊvəˌræks"); ok {
		t.Fatal("a word count mismatch was used")
	}
}

func TestBothServicesFormatPronunciationsTheSame(t *testing.T) {
	ws, _ := (&realtimeSynthesizer{}).FormatPronunciation("Metformin", "mɛtˈfɔɹmɪn")
	http, _ := (&synthesizer{}).FormatPronunciation("Metformin", "mɛtˈfɔɹmɪn")
	if ws != http {
		t.Fatalf("the services differ: %q and %q", ws, http)
	}
}

func TestTheWebSocketServiceNeedsAPhonemeModelAndSSMLParsing(t *testing.T) {
	on, off := true, false
	cases := []struct {
		model string
		ssml  *bool
		want  bool
	}{
		{"eleven_flash_v2", &on, true},
		{"eleven_turbo_v2", &on, true},
		{"eleven_flash_v2_5", &on, false},
		{"eleven_flash_v2", nil, false},
		{"eleven_flash_v2", &off, false},
	}
	for _, tc := range cases {
		s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{Model: tc.model, EnableSSMLParsing: tc.ssml}}
		if got := s.SupportsPronunciations(); got != tc.want {
			t.Errorf("model %s, ssml %v: supports = %v, want %v", tc.model, tc.ssml, got, tc.want)
		}
	}
}

func TestTheHTTPServiceNeedsAPhonemeModel(t *testing.T) {
	if !(&synthesizer{cfg: Config{Model: "eleven_turbo_v2"}}).SupportsPronunciations() {
		t.Error("eleven_turbo_v2 reads phoneme tags")
	}
	if (&synthesizer{cfg: Config{Model: "eleven_multilingual_v2"}}).SupportsPronunciations() {
		t.Error("eleven_multilingual_v2 does not read phoneme tags")
	}
}

func TestSSMLParsingIsSentWhenSet(t *testing.T) {
	on := true
	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{EnableSSMLParsing: &on}.withDefaults()}
	if q := parseEndpoint(t, s.endpoint()); q.Get("enable_ssml_parsing") != "true" {
		t.Fatalf("enable_ssml_parsing = %q, want true", q.Get("enable_ssml_parsing"))
	}
	s = &realtimeSynthesizer{cfg: RealtimeTTSConfig{}.withDefaults()}
	if q := parseEndpoint(t, s.endpoint()); q.Has("enable_ssml_parsing") {
		t.Fatal("enable_ssml_parsing was sent when it was not set")
	}
}

func parseEndpoint(t *testing.T, raw string) url.Values {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u.Query()
}
