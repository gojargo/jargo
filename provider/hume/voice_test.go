package hume

import "testing"

// TestVoiceSelectorPrefersAnID covers how a voice is named. An id names one
// exactly, so it wins over a name whenever both are configured.
func TestVoiceSelectorPrefersAnID(t *testing.T) {
	s := &synthesizer{cfg: Config{
		VoiceID:       "abc123",
		VoiceName:     "Aria",
		VoiceProvider: defaultVoiceProvider,
	}}

	got := s.voice()
	if got == nil {
		t.Fatal("no voice was selected with one configured")
	}
	if got["id"] != "abc123" {
		t.Errorf("id = %v, want the configured id", got["id"])
	}
	if _, named := got["name"]; named {
		t.Errorf("the selector carried a name as well as an id: %v", got)
	}
	if got["provider"] != defaultVoiceProvider {
		t.Errorf("provider = %v, want %q", got["provider"], defaultVoiceProvider)
	}
}

// TestVoiceSelectorFallsBackToAName covers a caller who knows the voice by name
// rather than by id.
func TestVoiceSelectorFallsBackToAName(t *testing.T) {
	s := &synthesizer{cfg: Config{VoiceName: "Aria", VoiceProvider: defaultVoiceProvider}}

	got := s.voice()
	if got == nil {
		t.Fatal("no voice was selected with a name configured")
	}
	if got["name"] != "Aria" {
		t.Errorf("name = %v, want the configured name", got["name"])
	}
	if _, byID := got["id"]; byID {
		t.Errorf("the selector carried an id with none configured: %v", got)
	}
}

// TestVoiceSelectorCarriesTheProvider covers a custom voice, which is named the
// same way as a stock one but has to say where it comes from or the wrong
// library is searched.
func TestVoiceSelectorCarriesTheProvider(t *testing.T) {
	s := &synthesizer{cfg: Config{VoiceID: "mine", VoiceProvider: "CUSTOM_VOICE"}}

	if got := s.voice()["provider"]; got != "CUSTOM_VOICE" {
		t.Errorf("provider = %v, want the configured CUSTOM_VOICE", got)
	}
}

// TestNoVoiceSelectsNone covers leaving the voice unset: nothing is sent, so the
// service is left to its own default rather than being handed an empty
// selector it would have to reject.
func TestNoVoiceSelectsNone(t *testing.T) {
	s := &synthesizer{cfg: Config{VoiceProvider: defaultVoiceProvider}}

	if got := s.voice(); got != nil {
		t.Errorf("voice = %v, want none selected", got)
	}
}
