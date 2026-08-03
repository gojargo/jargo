package polly

import (
	"strings"
	"testing"
)

// Arbitrary text cannot break the SSML document: Polly rejects a malformed one
// outright, so an ampersand in a reply would otherwise fail the whole sentence.
func TestSSMLEscapesReservedCharacters(t *testing.T) {
	cfg := Config{Language: "en-US"}

	got := cfg.ssml(`Fish & chips for <5 "quid"`)
	for _, want := range []string{"&amp;", "&lt;5"} {
		if !strings.Contains(got, want) {
			t.Errorf("ssml is missing %s:\n%s", want, got)
		}
	}
	if strings.Contains(got, "& chips") || strings.Contains(got, "<5") {
		t.Errorf("ssml carries an unescaped reserved character:\n%s", got)
	}
	// The markup around the text is untouched.
	if !strings.HasPrefix(got, "<speak><lang xml:lang='en-US'>") || !strings.HasSuffix(got, "</lang></speak>") {
		t.Errorf("ssml lost its own markup:\n%s", got)
	}
}

// The prosody controls still wrap the escaped text rather than being escaped
// themselves.
func TestSSMLKeepsProsodyAroundEscapedText(t *testing.T) {
	cfg := Config{Language: "en-US", Rate: "slow", Volume: "loud"}

	got := cfg.ssml("a & b")
	if !strings.Contains(got, "<prosody rate='slow' volume='loud'>a &amp; b</prosody>") {
		t.Errorf("prosody does not wrap the escaped text:\n%s", got)
	}
}
