package assemblyai

import (
	"errors"
	"testing"

	"github.com/gojargo/jargo/language"
)

// Declared languages resolve to the codes AssemblyAI names them by, collapsing
// the regional variants that share one, and keeping the order the steering
// follows.
func TestPrepareLanguageCodes(t *testing.T) {
	got := prepareLanguageCodes([]language.Language{
		language.Language("es-MX"), language.Language("en-US"),
		language.Language("es-ES"), language.Language("xx"),
	})
	want := []string{"es", "en"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("codes = %v, want %v", got, want)
	}
}

// Steering is prompt-based, so the declared languages are sent to a U3 Pro model
// and to no other.
func TestQuerySendsLanguageCodesToU3ProOnly(t *testing.T) {
	langs := []language.Language{language.Language("en"), language.Language("es")}

	pro := Config{APIKey: "k", Model: "u3-rt-pro", Encoding: defaultEncoding, LanguageCodes: langs}
	if got := pro.query(16000).Get("language_codes"); got != `["en","es"]` {
		t.Errorf("language_codes = %q, want the declared list", got)
	}

	other := Config{APIKey: "k", Model: "universal-streaming", Encoding: defaultEncoding, LanguageCodes: langs}
	if got := other.query(16000).Get("language_codes"); got != "" {
		t.Errorf("language_codes = %q, sent to a model that cannot be steered", got)
	}
}

// universal-3-6-pro is universal-3-5-pro upgraded, so the U3 Pro settings reach
// it as they reach any model of the family.
func TestQuerySendsU3ProSettingsToUniversal36Pro(t *testing.T) {
	cfg := Config{
		APIKey:        "k",
		Model:         "universal-3-6-pro",
		Encoding:      defaultEncoding,
		LanguageCodes: []language.Language{language.Language("en"), language.Language("es")},
	}
	q := cfg.query(16000)
	if got := q.Get("speech_model"); got != "universal-3-6-pro" {
		t.Errorf("speech_model = %q, want universal-3-6-pro", got)
	}
	if got := q.Get("language_codes"); got != `["en","es"]` {
		t.Errorf("language_codes = %q, want the declared list", got)
	}
}

// universal-3-6-pro is the default model sent to AssemblyAI.
func TestDefaultModelIsUniversal36Pro(t *testing.T) {
	cfg := Config{APIKey: "k"}.withDefaults()
	if got := cfg.query(16000).Get("speech_model"); got != "universal-3-6-pro" {
		t.Errorf("speech_model = %q, want universal-3-6-pro", got)
	}
}

// The default model is a U3 Pro model, so the declared languages steer it
// without a model being named.
func TestDefaultModelSendsLanguageCodes(t *testing.T) {
	cfg := Config{APIKey: "k", LanguageCodes: []language.Language{language.Language("en")}}.withDefaults()
	if got := cfg.query(16000).Get("language_codes"); got != `["en"]` {
		t.Errorf("language_codes = %q, want the declared list", got)
	}
}

// universal-3-6-pro inherits universal-3-5-pro's full language set, so each of
// these languages is a verified entry sent as its code, not one dropped as
// unsupported.
func TestLanguageCodesCoverUniversal36ProAdditions(t *testing.T) {
	cases := []struct {
		lang language.Language
		want string
	}{
		{language.Urdu, "ur"},
		{language.Russian, "ru"},
		{language.Korean, "ko"},
		{language.Catalan, "ca"},
		{language.Galician, "gl"},
		{language.Romanian, "ro"},
		{language.Estonian, "et"},
		{language.Persian, "fa"},
		{language.YueChineseCantonese, "yue"},
		{language.Afrikaans, "af"},
		{language.Marathi, "mr"},
		{language.Zulu, "zu"},
		{language.Xhosa, "xh"},
		{language.NorwegianNynorsk, "nn"},
	}
	for _, c := range cases {
		cfg := Config{APIKey: "k", LanguageCodes: []language.Language{c.lang}}.withDefaults()
		want := `["` + c.want + `"]`
		if got := cfg.query(16000).Get("language_codes"); got != want {
			t.Errorf("language_codes for %s = %q, want %q", c.lang, got, want)
		}
	}
}

// U3 Pro models accept a prompt together with key terms.
func TestValidateAcceptsPromptAndKeytermsForU3RtPro(t *testing.T) {
	cfg := Config{
		APIKey:         "k",
		Model:          "u3-rt-pro",
		Prompt:         "Transcribe the order.",
		KeytermsPrompt: []string{"espresso"},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	cfg = cfg.withDefaults()
	q := cfg.query(16000)
	if q.Get("prompt") != "Transcribe the order." || q.Get("keyterms_prompt") != `["espresso"]` {
		t.Errorf("prompt = %q, keyterms_prompt = %q, want both sent",
			q.Get("prompt"), q.Get("keyterms_prompt"))
	}
}

// A model that is not U3 Pro rejects a prompt, with or without key terms, so
// it is refused before anything connects.
func TestValidateRejectsPromptForUniversalStreaming(t *testing.T) {
	withKeyterms := Config{
		APIKey:         "k",
		Model:          "universal-streaming-english",
		Prompt:         "Transcribe the order.",
		KeytermsPrompt: []string{"espresso"},
	}
	if err := withKeyterms.Validate(); !errors.Is(err, errPromptNotSupported) {
		t.Errorf("Validate with prompt and key terms = %v, want errPromptNotSupported", err)
	}

	alone := Config{
		APIKey: "k",
		Model:  "universal-streaming-english",
		Prompt: "Some context for the session.",
	}
	if err := alone.Validate(); !errors.Is(err, errPromptNotSupported) {
		t.Errorf("Validate with prompt alone = %v, want errPromptNotSupported", err)
	}
}

// The default model is U3 Pro, so a prompt is accepted without a model being
// named.
func TestValidateAcceptsPromptForDefaultModel(t *testing.T) {
	if err := (Config{APIKey: "k", Prompt: "Transcribe the order."}).Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// More languages than AssemblyAI accepts is rejected before anything connects,
// since the service closes the session over it rather than ignoring it.
func TestValidateRejectsTooManyLanguages(t *testing.T) {
	// Eleven distinct languages, one past the limit.
	langs := []language.Language{"en", "es", "fr", "de", "it", "pt", "tr", "nl", "sv", "no", "da"}
	if err := (Config{APIKey: "k", LanguageCodes: langs}).Validate(); err == nil {
		t.Error("Validate accepted more declared languages than the service takes")
	}
	if err := (Config{APIKey: "k", LanguageCodes: langs[:maxLanguageCodes]}).Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
	// Regional variants collapse before counting, so these are two languages.
	dupes := []language.Language{"en-US", "en-GB", "es-MX", "es-ES"}
	if err := (Config{APIKey: "k", LanguageCodes: dupes}).Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// A U3 Pro model is recognized by the prefixes the service names its streaming
// variants with.
func TestIsU3ProModel(t *testing.T) {
	cases := map[string]bool{
		"u3-rt-pro":           true,
		"u3-rt-pro-2026":      true,
		"universal-3-5-pro":   true,
		"universal-3-6-pro":   true,
		"universal-streaming": false,
		"universal-3-5":       false,
		"":                    false,
	}
	for model, want := range cases {
		if got := isU3ProModel(model); got != want {
			t.Errorf("isU3ProModel(%q) = %v, want %v", model, got, want)
		}
	}
}
