package speech

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
)

// ForceLocale wraps the text in SSML's <lang> element, which is what pins a
// multilingual voice to the configured language instead of the one it reads out
// of the text.
func TestSSMLForceLocale(t *testing.T) {
	cfg := TTSConfig{
		APIKey: "k", Region: "eastus",
		Voice: "en-US-EmmaMultilingualNeural", Language: language.Language("fr-FR"),
		ForceLocale: true,
	}
	s := &ttsSynthesizer{cfg: cfg}

	doc, err := s.ssml("bonjour")
	if err != nil {
		t.Fatalf("ssml: %v", err)
	}
	if want := "<lang xml:lang='fr-FR'>bonjour</lang>"; !strings.Contains(string(doc), want) {
		t.Errorf("ssml is missing %s:\n%s", want, doc)
	}
}

// Without it the text is spoken as it is, leaving a multilingual voice to read
// the language out of the text.
func TestSSMLWithoutForceLocale(t *testing.T) {
	s := &ttsSynthesizer{cfg: TTSConfig{
		APIKey: "k", Region: "eastus",
		Voice: "en-US-EmmaMultilingualNeural", Language: language.Language("fr-FR"),
	}}

	doc, err := s.ssml("bonjour")
	if err != nil {
		t.Fatalf("ssml: %v", err)
	}
	if strings.Contains(string(doc), "<lang") {
		t.Errorf("ssml pins the locale without being asked to:\n%s", doc)
	}
	if !strings.Contains(string(doc), ">bonjour<") {
		t.Errorf("ssml lost the text:\n%s", doc)
	}
}
