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

// TestSSMLEffect checks the audio effect processor reaches the <voice> element,
// which is what compensates for the playback distortion of the device the audio
// is bound for.
func TestSSMLEffect(t *testing.T) {
	s := &ttsSynthesizer{cfg: TTSConfig{
		Voice:  "en-US-JennyNeural",
		Effect: "eq_telecomhp8k",
	}}

	doc, err := s.ssml("hello")
	if err != nil {
		t.Fatalf("ssml: %v", err)
	}
	if want := "<voice name='en-US-JennyNeural' effect='eq_telecomhp8k'>"; !strings.Contains(string(doc), want) {
		t.Errorf("ssml is missing %s:\n%s", want, doc)
	}
}

// TestSSMLWithoutEffect checks the attribute is left off rather than sent empty
// when no effect was asked for.
func TestSSMLWithoutEffect(t *testing.T) {
	s := &ttsSynthesizer{cfg: TTSConfig{Voice: "en-US-JennyNeural"}}

	doc, err := s.ssml("hello")
	if err != nil {
		t.Fatalf("ssml: %v", err)
	}
	if strings.Contains(string(doc), "effect=") {
		t.Errorf("ssml carries an effect nobody asked for:\n%s", doc)
	}
}
