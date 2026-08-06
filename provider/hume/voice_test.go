package hume

import (
	"encoding/json"
	"testing"
)

// requestOf builds the synthesis body for text and decodes it back from JSON, so
// the wire field names are covered alongside the values.
func requestOf(t *testing.T, cfg Config, text string) map[string]any {
	t.Helper()
	s := &synthesizer{cfg: cfg}
	raw, err := json.Marshal(s.request(text))
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return out
}

// utteranceOf returns the single utterance the request carries.
func utteranceOf(t *testing.T, req map[string]any) map[string]any {
	t.Helper()
	list, ok := req["utterances"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("utterances = %v, want exactly one", req["utterances"])
	}
	u, ok := list[0].(map[string]any)
	if !ok {
		t.Fatalf("utterance = %v, want an object", list[0])
	}
	return u
}

// TestRequestNamesTheVoiceByID covers how the voice is chosen. Hume names a
// voice by id and by nothing else, so that is all that is sent: a name or a
// provider alongside it would be fields the request does not have.
func TestRequestNamesTheVoiceByID(t *testing.T) {
	req := requestOf(t, Config{APIKey: "k", VoiceID: "abc123"}, "hello")

	u := utteranceOf(t, req)
	if u["text"] != "hello" {
		t.Errorf("text = %v, want %q", u["text"], "hello")
	}
	voice, ok := u["voice"].(map[string]any)
	if !ok {
		t.Fatalf("voice = %v, want an object", u["voice"])
	}
	if voice["id"] != "abc123" {
		t.Errorf("voice id = %v, want the configured id", voice["id"])
	}
	if len(voice) != 1 {
		t.Errorf("voice = %v, want the id alone", voice)
	}
}

// TestRequestAlwaysUsesInstantMode covers the mode the service runs in. It is
// not a choice, and it needs a named voice, which is what requiring one buys.
func TestRequestAlwaysUsesInstantMode(t *testing.T) {
	req := requestOf(t, Config{APIKey: "k", VoiceID: "abc123"}, "hello")

	if req["instant_mode"] != true {
		t.Errorf("instant_mode = %v, want it always on", req["instant_mode"])
	}
	if req["strip_headers"] != true {
		t.Errorf("strip_headers = %v, want the container stripped", req["strip_headers"])
	}
	format, ok := req["format"].(map[string]any)
	if !ok || format["type"] != "pcm" {
		t.Errorf("format = %v, want raw samples", req["format"])
	}
}

// TestRequestOmitsWhatWasNotConfigured covers the optional half: a field left
// unset is left out entirely, so the service applies its own default rather than
// being handed a zero value.
func TestRequestOmitsWhatWasNotConfigured(t *testing.T) {
	req := requestOf(t, Config{APIKey: "k", VoiceID: "abc123"}, "hello")
	u := utteranceOf(t, req)

	for _, k := range []string{"description", "speed"} {
		if _, present := u[k]; present {
			t.Errorf("%s was sent without being configured: %v", k, u[k])
		}
	}
	if _, present := req["version"]; present {
		t.Errorf("version was sent without being configured: %v", req["version"])
	}
}

// TestRequestIncludesWhatWasConfigured is the other half: what was set reaches
// the service under the name it expects, the delivery prompt and the rate on the
// utterance and the model version on the request itself.
func TestRequestIncludesWhatWasConfigured(t *testing.T) {
	speed := 1.25
	req := requestOf(t, Config{
		APIKey:      "k",
		VoiceID:     "abc123",
		Description: "warm and unhurried",
		Speed:       &speed,
		Version:     "2",
	}, "hello")

	u := utteranceOf(t, req)
	if u["description"] != "warm and unhurried" {
		t.Errorf("description = %v, want the configured prompt", u["description"])
	}
	if u["speed"] != 1.25 {
		t.Errorf("speed = %v, want 1.25", u["speed"])
	}
	if req["version"] != "2" {
		t.Errorf("version = %v, want %q", req["version"], "2")
	}
}
