package gladia

import (
	"encoding/json"
	"testing"
)

// settingsOf builds the session payload a config would open a session with, at
// the given rate, decoded back from JSON so the wire field names are covered
// alongside the values.
func settingsOf(t *testing.T, cfg Config, sampleRate int) map[string]any {
	t.Helper()
	// Through the same defaults the service is built with, so what is asserted
	// is what a session would really be opened with.
	filled := withDefaults(cfg)

	raw, err := json.Marshal(filled.settings(sampleRate))
	if err != nil {
		t.Fatalf("marshal settings: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode settings: %v", err)
	}
	return out
}

// TestSettingsCarryTheAudioShape covers what the session must always be opened
// with. The service is told how to decode the audio it is about to be sent, so a
// missing or wrong field here means every session transcribes noise.
func TestSettingsCarryTheAudioShape(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k"}, 16000)

	want := map[string]any{
		"encoding":    "wav/pcm",
		"bit_depth":   float64(16),
		"channels":    float64(1),
		"model":       "solaria-1",
		"sample_rate": float64(16000),
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("%s = %v, want %v", k, got[k], w)
		}
	}
}

// TestSettingsUseTheRateTheSessionRunsAt covers the rate being decided when the
// session opens rather than when the service is built: the transport's rate is
// what the audio actually arrives at.
func TestSettingsUseTheRateTheSessionRunsAt(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k", SampleRate: 8000}, 24000)

	if got["sample_rate"] != float64(24000) {
		t.Errorf("sample_rate = %v, want the session's 24000", got["sample_rate"])
	}
}

// TestSettingsOmitWhatWasNotConfigured covers the optional half. A field left
// unset is left out entirely, so the service applies its own default instead of
// being handed a zero value that means something different.
func TestSettingsOmitWhatWasNotConfigured(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k"}, 16000)

	for _, k := range []string{
		"endpointing",
		"maximum_duration_without_endpointing",
		"language_config",
		"pre_processing",
		"realtime_processing",
		"custom_metadata",
	} {
		if _, present := got[k]; present {
			t.Errorf("%s was sent without being configured: %v", k, got[k])
		}
	}
}

// TestSettingsIncludeWhatWasConfigured is the other half: a field that was set
// reaches the service under the name it expects.
func TestSettingsIncludeWhatWasConfigured(t *testing.T) {
	endpointing := 0.3
	maxDuration := 12
	got := settingsOf(t, Config{
		APIKey:                            "k",
		Endpointing:                       &endpointing,
		MaximumDurationWithoutEndpointing: &maxDuration,
		CustomMetadata:                    map[string]any{"call": "42"},
	}, 16000)

	if got["endpointing"] != 0.3 {
		t.Errorf("endpointing = %v, want 0.3", got["endpointing"])
	}
	if got["maximum_duration_without_endpointing"] != float64(12) {
		t.Errorf("maximum_duration_without_endpointing = %v, want 12", got["maximum_duration_without_endpointing"])
	}
	meta, ok := got["custom_metadata"].(map[string]any)
	if !ok || meta["call"] != "42" {
		t.Errorf("custom_metadata = %v, want it to carry the configured entry", got["custom_metadata"])
	}
}

// TestSettingsAskForBothTranscriptKinds covers the default message filter. The
// pipeline needs the partials to show speech as it arrives and the finals to end
// a turn on, so both are requested unless the caller says otherwise.
func TestSettingsAskForBothTranscriptKinds(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k"}, 16000)

	mc, ok := got["messages_config"].(map[string]any)
	if !ok {
		t.Fatalf("messages_config = %v, want the default filter", got["messages_config"])
	}
	if mc["receive_partial_transcripts"] != true || mc["receive_final_transcripts"] != true {
		t.Errorf("messages_config = %v, want both transcript kinds requested", mc)
	}
}

// TestSettingsKeepACallersMessageFilter covers a caller narrowing the filter
// themselves: their choice replaces the default rather than being merged into
// it, so asking for finals only really does turn the partials off.
func TestSettingsKeepACallersMessageFilter(t *testing.T) {
	off, on := false, true
	got := settingsOf(t, Config{
		APIKey: "k",
		MessagesConfig: &MessagesConfig{
			ReceivePartialTranscripts: &off,
			ReceiveFinalTranscripts:   &on,
		},
	}, 16000)

	mc, ok := got["messages_config"].(map[string]any)
	if !ok {
		t.Fatalf("messages_config = %v, want the caller's filter", got["messages_config"])
	}
	if mc["receive_partial_transcripts"] != false {
		t.Errorf("receive_partial_transcripts = %v, want the caller's false", mc["receive_partial_transcripts"])
	}
}

// TestSettingsLetExtraSettingsWin covers the escape hatch for options this
// config does not model. It is last, so a caller can reach a new service option
// without waiting for it to be added here, including by overriding one that is.
func TestSettingsLetExtraSettingsWin(t *testing.T) {
	got := settingsOf(t, Config{
		APIKey:        "k",
		ExtraSettings: map[string]any{"model": "override", "brand_new_option": true},
	}, 16000)

	if got["model"] != "override" {
		t.Errorf("model = %v, want the override", got["model"])
	}
	if got["brand_new_option"] != true {
		t.Errorf("brand_new_option = %v, want it passed through", got["brand_new_option"])
	}
}
