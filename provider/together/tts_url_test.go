package together

import (
	"net/url"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for the TTS endpoint. Together reads the model, the voice and the
// language from the query string of the WebSocket URL, so each has to arrive
// there escaped and in the form the API accepts.

// endpointQuery builds the endpoint for cfg, with the defaults NewTTS fills in,
// and returns its query.
func endpointQuery(t *testing.T, cfg TTSConfig) url.Values {
	t.Helper()
	cfg.APIKey = "test-key"
	_, s := newTTS(cfg)
	u, err := url.Parse(s.buildWebsocketURL())
	if err != nil {
		t.Fatalf("parsing the endpoint: %v", err)
	}
	return u.Query()
}

// TestEndpointSendsDefaultLanguage checks a config naming nothing reaches the
// endpoint with the default model and voice, English, and no partial cap.
func TestEndpointSendsDefaultLanguage(t *testing.T) {
	q := endpointQuery(t, TTSConfig{})

	if got := q.Get("model"); got != "hexgrad/Kokoro-82M" {
		t.Errorf("model = %q, want hexgrad/Kokoro-82M", got)
	}
	if got := q.Get("voice"); got != "af_heart" {
		t.Errorf("voice = %q, want af_heart", got)
	}
	if got := q.Get("language"); got != "en" {
		t.Errorf("language = %q, want en", got)
	}
	if q.Has("max_partial_length") {
		t.Error("max_partial_length was sent though none was configured")
	}
}

// TestEndpointEscapesBlendedVoice checks a Kokoro blended voice keeps its `+`:
// unescaped, the server reads it as a space and rejects the voice. The model's
// `/` stays as it is.
func TestEndpointEscapesBlendedVoice(t *testing.T) {
	endpoint := newSynthesizer(TTSConfig{APIKey: "test-key", Voice: "af_bella(2)+af_heart(1)"}).buildWebsocketURL()

	if !strings.Contains(endpoint, "voice=af_bella%282%29%2Baf_heart%281%29") {
		t.Errorf("endpoint = %q, want the blended voice escaped", endpoint)
	}
	if !strings.Contains(endpoint, "model=hexgrad/Kokoro-82M") {
		t.Errorf("endpoint = %q, want the model with its slash intact", endpoint)
	}
}

// TestEndpointResolvesLanguage checks a language is sent as the ISO 639-1 or
// lowercase locale code Together accepts, a regional variant without a
// verified locale falling back to its base code.
func TestEndpointResolvesLanguage(t *testing.T) {
	cases := []struct {
		lang language.Language
		want string
	}{
		{language.French, "fr"},
		{language.EnglishUS, "en"},
		{language.ChineseHK, "zh-hk"},
		{"pt-BR", "pt"},
	}
	for _, c := range cases {
		t.Run(string(c.lang), func(t *testing.T) {
			if got := endpointQuery(t, TTSConfig{Language: c.lang}).Get("language"); got != c.want {
				t.Errorf("language = %q, want %q", got, c.want)
			}
		})
	}
}

// TestEndpointOmitsAnEmptyLanguage checks settings holding no language send
// none, leaving Together on its own default.
func TestEndpointOmitsAnEmptyLanguage(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "test-key"})
	cleared := &TTSSettings{}
	cleared.Language = settings.Cleared[string]()
	if _, err := settings.Apply(s.live, cleared); err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse(s.buildWebsocketURL())
	if err != nil {
		t.Fatalf("parsing the endpoint: %v", err)
	}
	if u.Query().Has("language") {
		t.Errorf("endpoint = %q, want no language", u)
	}
}

// TestEndpointSendsMaxPartialLength checks the optional cap reaches the
// endpoint as a query parameter when it is configured.
func TestEndpointSendsMaxPartialLength(t *testing.T) {
	partial := 120
	if got := endpointQuery(t, TTSConfig{MaxPartialLength: &partial}).Get("max_partial_length"); got != "120" {
		t.Errorf("max_partial_length = %q, want 120", got)
	}
}

// TestServiceLanguage checks a neutral language given in a settings update is
// named the way Together names it.
func TestServiceLanguage(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "test-key"})
	for in, want := range map[language.Language]string{
		language.French:    "fr",
		language.EnglishUS: "en",
		language.ChineseHK: "zh-hk",
	} {
		if got := s.ServiceLanguage(in); got != want {
			t.Errorf("ServiceLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}
