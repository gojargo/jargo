package assemblyai

import (
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
