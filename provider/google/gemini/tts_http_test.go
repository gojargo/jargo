package gemini

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/gojargo/jargo/service/settings"
	"golang.org/x/oauth2"
)

// Ported from upstream (test_text_is_escaped_inside_ssml). The text spoken by a
// voice that takes SSML is escaped, so characters that are markup in SSML are
// read as text rather than breaking the document.
func TestTextIsEscapedInsideSSML(t *testing.T) {
	syn := newTTSSynthesizer(TTSConfig{APIKey: "k", VoiceName: "en-US-Neural2-A"}, &http.Client{})

	ssml := syn.constructSSML("Q&A: is 3 < 5 > 2?")

	if !strings.Contains(ssml, "Q&amp;A: is 3 &lt; 5 &gt; 2?") {
		t.Errorf("SSML = %q, want the text escaped inside it", ssml)
	}
}

// TestSSMLCarriesTheVoiceControls checks a voice that takes SSML is sent the
// voice, prosody, emphasis and style controls, nested in that order.
func TestSSMLCarriesTheVoiceControls(t *testing.T) {
	rt := &roundTripper{body: wavReply([]byte{1, 2})}
	syn := newTTSSynthesizer(TTSConfig{
		APIKey:      "k",
		VoiceName:   "en-US-Neural2-A",
		Pitch:       "+2st",
		Rate:        "slow",
		Volume:      "loud",
		Emphasis:    "strong",
		Gender:      "female",
		GoogleStyle: "calm",
	}, &http.Client{Transport: rt})

	synthesize(t, syn, "hi")

	input, _ := rt.sent["input"].(map[string]any)
	want := "<speak><voice name='en-US-Neural2-A' language='en-US' gender='female'>" +
		"<prosody pitch='+2st' rate='slow' volume='loud'><emphasis level='strong'>" +
		"<google:style name='calm'>hi</google:style></emphasis></prosody></voice></speak>"
	if input["ssml"] != want {
		t.Errorf("ssml = %v, want %s", input["ssml"], want)
	}
	if _, ok := input["text"]; ok {
		t.Error("a voice that takes SSML was also sent plain text")
	}
}

// TestPlainTextVoicesTakeTheSpeakingRate checks Chirp and Journey voices, which
// take no SSML, are sent plain text with the speaking rate in the audio config,
// and other voices are not sent the speaking rate.
func TestPlainTextVoicesTakeTheSpeakingRate(t *testing.T) {
	rate := 1.5
	for voice, plain := range map[string]bool{
		"en-US-Chirp3-HD-Charon": true,
		"en-US-Journey-D":        true,
		"en-US-Neural2-A":        false,
	} {
		rt := &roundTripper{body: wavReply([]byte{1, 2})}
		syn := newTTSSynthesizer(TTSConfig{APIKey: "k", VoiceName: voice, SpeakingRate: &rate},
			&http.Client{Transport: rt})
		synthesize(t, syn, "hi")

		input, _ := rt.sent["input"].(map[string]any)
		audio, _ := rt.sent["audioConfig"].(map[string]any)
		if plain {
			if input["text"] != "hi" {
				t.Errorf("%s: input = %v, want plain text", voice, input)
			}
			if audio["speakingRate"] != rate {
				t.Errorf("%s: speakingRate = %v, want %v", voice, audio["speakingRate"], rate)
			}
			continue
		}
		if _, ok := input["ssml"]; !ok {
			t.Errorf("%s: input = %v, want SSML", voice, input)
		}
		if _, ok := audio["speakingRate"]; ok {
			t.Errorf("%s: a voice taking SSML was sent the speaking rate", voice)
		}
	}
}

// TestSpeakingRateOutsideTheRangeIsRefused checks an update giving a speaking
// rate Google would reject leaves the rate as it was, while the rest of the
// update applies.
func TestSpeakingRateOutsideTheRangeIsRefused(t *testing.T) {
	syn := newTTSSynthesizer(TTSConfig{APIKey: "k"}, &http.Client{})
	store := syn.live

	changed, err := settings.Apply(store, &TTSSettings{
		SpeakingRate: settings.Set(3.0),
		Pitch:        settings.Set("+1st"),
	})
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if changed.Has("speaking_rate") {
		t.Error("an out-of-range speaking rate was reported as changed")
	}
	if _, ok := store.SpeakingRate.Value(); ok {
		t.Error("an out-of-range speaking rate was stored")
	}
	if pitch, _ := store.Pitch.Value(); pitch != "+1st" {
		t.Errorf("pitch = %q, want the rest of the update applied", pitch)
	}

	if _, err := settings.Apply(store, &TTSSettings{SpeakingRate: settings.Set(1.25)}); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if rate, _ := store.SpeakingRate.Value(); rate != 1.25 {
		t.Errorf("speaking rate = %v, want an in-range rate stored", rate)
	}
}

// TestCredentialsAuthorizeWithABearerToken checks a service given credentials
// sends an access token rather than a key, and that a location addresses the
// regional endpoint.
func TestCredentialsAuthorizeWithABearerToken(t *testing.T) {
	rt := &roundTripper{body: wavReply([]byte{1, 2})}
	syn := newTTSSynthesizer(TTSConfig{
		APIKey:      "unused",
		Credentials: Credentials{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok-123"})},
		Location:    "us-central1",
	}, &http.Client{Transport: rt})

	synthesize(t, syn, "hi")

	if got := rt.header.Get("Authorization"); got != "Bearer tok-123" {
		t.Errorf("Authorization = %q, want the access token", got)
	}
	if rt.url != "https://us-central1-texttospeech.googleapis.com/v1/text:synthesize" {
		t.Errorf("URL = %q, want the regional endpoint with no key", rt.url)
	}
}

// TestAPIKeyStillAuthorizes checks a service given only a key sends it, and no
// access token.
func TestAPIKeyStillAuthorizes(t *testing.T) {
	rt := &roundTripper{body: wavReply([]byte{1, 2})}
	syn := newTTSSynthesizer(TTSConfig{APIKey: "test-key"}, &http.Client{Transport: rt})

	synthesize(t, syn, "hi")

	if rt.url != ttsEndpoint+"?key=test-key" {
		t.Errorf("URL = %q, want the key in the query", rt.url)
	}
	if got := rt.header.Get("Authorization"); got != "" {
		t.Errorf("Authorization = %q, want none", got)
	}
}

// TestTTSConfigValidateNeedsAuth checks a configuration with neither a key nor
// credentials is refused, and either one alone is accepted.
func TestTTSConfigValidateNeedsAuth(t *testing.T) {
	if err := (TTSConfig{}).Validate(); !errors.Is(err, errNoTTSAuth) {
		t.Errorf("Validate() = %v, want errNoTTSAuth", err)
	}
	if err := (TTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() with a key = %v, want nil", err)
	}
	if err := (TTSConfig{Credentials: Credentials{Path: "/key.json"}}).Validate(); err != nil {
		t.Errorf("Validate() with credentials = %v, want nil", err)
	}
	if err := (TTSConfig{APIKey: "k", Emphasis: "loudest"}).Validate(); err == nil {
		t.Error("Validate() accepted an emphasis Google does not offer")
	}
}
