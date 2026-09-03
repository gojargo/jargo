package cartesia

import (
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TestTTSSettingsSeed checks the store opens on what the service was built with,
// so an update is compared against the real configuration rather than nothing.
func TestTTSSettingsSeed(t *testing.T) {
	speed := 1.2
	s := newSynthesizer(ttsDefaults(Config{
		APIKey:              "k",
		VoiceID:             testVoiceID,
		Language:            language.French,
		GenerationConfig:    &GenerationConfig{Speed: &speed},
		PronunciationDictID: "dict-1",
	}))

	live, ok := any(s).(tts.SettingsHolder).Settings().(*TTSSettings)
	if !ok {
		t.Fatal("Settings() did not return the provider's own store")
	}
	if got := live.Model.Or(""); got != defaultModel {
		t.Errorf("model = %q, want %q", got, defaultModel)
	}
	if got := live.Voice.Or(""); got != testVoiceID {
		t.Errorf("voice = %q, want the configured one", got)
	}
	if got := live.Language.Or(""); got != "fr" {
		t.Errorf("language = %q, want Cartesia's code for it", got)
	}
	if got := live.PronunciationDictID.Or(""); got != "dict-1" {
		t.Errorf("pronunciation dictionary = %q, want the configured one", got)
	}
}

// TestTTSSettingsReachTheRequest checks a changed setting goes out on the next
// synthesis rather than the value the service was built with.
func TestTTSSettingsReachTheRequest(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{{"type": "done"}})
	s := newSynthesizer(ttsConfig(endpoint))
	s.live.Voice = settings.Set("voice-2")
	s.live.Model = settings.Set("sonic-3.5")
	s.live.Language = settings.Set("de")
	s.live.PronunciationDictID = settings.Set("dict-2")

	speak(t, s, "hallo")

	got := seen.first()
	voice, _ := got["voice"].(map[string]any)
	if voice["id"] != "voice-2" {
		t.Errorf("voice = %v, want the updated one", voice)
	}
	if got["model_id"] != "sonic-3.5" {
		t.Errorf("model_id = %v, want the updated one", got["model_id"])
	}
	if got["language"] != "de" {
		t.Errorf("language = %v, want the updated one", got["language"])
	}
	if got["pronunciation_dict_id"] != "dict-2" {
		t.Errorf("pronunciation_dict_id = %v, want the updated one", got["pronunciation_dict_id"])
	}
	// The model labels the metrics and is what the characters are priced
	// against, so it follows the settings too.
	if m := s.Metadata(); m.Model != "sonic-3.5" || m.VoiceID != "voice-2" {
		t.Errorf("Metadata() = %+v, want the updated model and voice", m)
	}
}

// Cartesia fixes the voice, the model and the language for the length of a
// context, so a change to any of them opens a new one: the rest of the turn is
// spoken with the new settings, and what was already sent finishes as it was.
func TestTTSSettingsRotateTheContext(t *testing.T) {
	s := newSynthesizer(ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID}))
	host := &ttsHost{}
	s.SetAudioContextHost(host)

	for _, field := range []string{"voice", "model", "language"} {
		before := host.rotated()
		if err := s.UpdateSettings(t.Context(), settings.Changed{field: nil}); err != nil {
			t.Fatalf("UpdateSettings(%s): %v", field, err)
		}
		if host.rotated() != before+1 {
			t.Errorf("a changed %s did not open a new context, so it never takes effect", field)
		}
	}
}

// A setting that goes out on every request needs no new context: the next
// sentence carries it.
func TestTTSSettingsThatNeedNoNewContext(t *testing.T) {
	s := newSynthesizer(ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID}))
	host := &ttsHost{}
	s.SetAudioContextHost(host)

	changed := settings.Changed{"generation_config": nil, "pronunciation_dict_id": nil}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if got := host.rotated(); got != 0 {
		t.Errorf("the turn's context was rotated %d times for settings the next request carries", got)
	}
}

// TestTTSServiceLanguage checks a language given neutrally is stored under
// Cartesia's own code, so the same language does not read as a change.
func TestTTSServiceLanguage(t *testing.T) {
	s := newSynthesizer(ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID}))
	namer, ok := any(s).(tts.LanguageNamer)
	if !ok {
		t.Fatal("the synthesizer does not name languages Cartesia's way")
	}
	if got := namer.ServiceLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("ServiceLanguage(fr-CA) = %q, want the base code", got)
	}
}
