package live

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// Ports of tests/test_gemini_live_language.py.

func TestConstructorConvertsLanguageToLocaleCode(t *testing.T) {
	s := New(Config{APIKey: "test-key", Language: language.Spanish})
	if got := s.live.Language.Or(""); got != "es-ES" {
		t.Errorf("language = %q, want es-ES", got)
	}
}

func TestConstructorKeepsLanguageStringAsGiven(t *testing.T) {
	s := New(Config{APIKey: "test-key", Language: "es-MX"})
	if got := s.live.Language.Or(""); got != "es-MX" {
		t.Errorf("language = %q, want es-MX", got)
	}
}

// updateSettings sends s a settings update the way the pipeline does.
func updateSettings(t *testing.T, s *Service, delta *Settings) {
	t.Helper()
	f := frames.NewLLMUpdateSettingsFrame(delta)
	if err := s.Base.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
}

func TestSettingsUpdateConvertsLanguageToLocaleCode(t *testing.T) {
	s, _ := newTestService(t)
	updateSettings(t, s, &Settings{Language: settings.Set(language.French)})
	if got := s.live.Language.Or(""); got != "fr-FR" {
		t.Errorf("language = %q, want fr-FR", got)
	}
}

func TestSettingsUpdateKeepsLanguageStringAsGiven(t *testing.T) {
	s, _ := newTestService(t)
	updateSettings(t, s, &Settings{Language: settings.Set(language.Language("fr-CA"))})
	if got := s.live.Language.Or(""); got != "fr-CA" {
		t.Errorf("language = %q, want fr-CA", got)
	}
}

// The tests below are not ports of upstream tests: they cover the setup message
// built from the settings, which upstream builds in _connect through the SDK.

// The defaults are upstream's: the Charon voice, US English, 4096 tokens, the
// audio modality and the model's own media resolution. The sampling settings
// left unset are not sent.
func TestTheSetupCarriesTheDefaults(t *testing.T) {
	gen, _ := setupOf(t, New(Config{APIKey: "k"}))["generationConfig"].(map[string]any)
	speech, _ := gen["speechConfig"].(map[string]any)
	voice, _ := speech["voiceConfig"].(map[string]any)["prebuiltVoiceConfig"].(map[string]any)
	if voice["voiceName"] != "Charon" {
		t.Errorf("voice = %v, want Charon", voice["voiceName"])
	}
	if speech["languageCode"] != "en-US" {
		t.Errorf("language = %v, want en-US", speech["languageCode"])
	}
	if gen["maxOutputTokens"] != 4096 {
		t.Errorf("maxOutputTokens = %v, want 4096", gen["maxOutputTokens"])
	}
	if m, _ := gen["responseModalities"].([]string); len(m) != 1 || m[0] != "AUDIO" {
		t.Errorf("responseModalities = %v, want AUDIO", gen["responseModalities"])
	}
	if gen["mediaResolution"] != "MEDIA_RESOLUTION_UNSPECIFIED" {
		t.Errorf("mediaResolution = %v, want unspecified", gen["mediaResolution"])
	}
	for _, key := range []string{
		"temperature", "topP", "topK", "seed", "frequencyPenalty", "presencePenalty", "enableAffectiveDialog",
	} {
		if v, ok := gen[key]; ok {
			t.Errorf("%s = %v, want it left out", key, v)
		}
	}
}

// What the Config sets reaches the setup where the API reads it.
func TestTheSetupCarriesTheConfiguredSettings(t *testing.T) {
	temp, topP, tokens, trigger, padding := 0.4, 0.9, 512, 1000, 200
	off, proactive := false, true
	setup := setupOf(t, New(Config{
		APIKey: "k", Temperature: &temp, TopP: &topP, MaxTokens: &tokens,
		MediaResolution:          MediaResolutionLow,
		EnableAffectiveDialog:    true,
		ContextWindowCompression: &ContextWindowCompression{Enabled: true, TriggerTokens: &trigger},
		Proactivity:              &Proactivity{ProactiveAudio: &proactive},
		VAD: &VADParams{
			Disabled: &off, StartSensitivity: StartSensitivityLow, PrefixPaddingMs: &padding,
		},
		TurnCoverage: TurnIncludesAllInput,
	}))
	gen, _ := setup["generationConfig"].(map[string]any)
	if gen["temperature"] != 0.4 || gen["topP"] != 0.9 || gen["maxOutputTokens"] != 512 {
		t.Errorf("generationConfig = %v, want the sampling settings", gen)
	}
	if gen["mediaResolution"] != "MEDIA_RESOLUTION_LOW" || gen["enableAffectiveDialog"] != true {
		t.Errorf("generationConfig = %v, want the resolution and affective dialog", gen)
	}
	cwc, _ := setup["contextWindowCompression"].(map[string]any)
	if _, ok := cwc["slidingWindow"]; !ok || cwc["triggerTokens"] != 1000 {
		t.Errorf("contextWindowCompression = %v, want a sliding window triggered at 1000", cwc)
	}
	if p, _ := setup["proactivity"].(map[string]any); p["proactiveAudio"] != true {
		t.Errorf("proactivity = %v, want proactive audio", p)
	}
	input, _ := setup["realtimeInputConfig"].(map[string]any)
	detection, _ := input["automaticActivityDetection"].(map[string]any)
	if detection["disabled"] != false || detection["startOfSpeechSensitivity"] != "START_SENSITIVITY_LOW" ||
		detection["prefixPaddingMs"] != 200 {
		t.Errorf("automaticActivityDetection = %v, want the VAD settings given", detection)
	}
	if _, ok := detection["endOfSpeechSensitivity"]; ok {
		t.Errorf("automaticActivityDetection = %v, want an unset sensitivity left out", detection)
	}
	if input["turnCoverage"] != "TURN_INCLUDES_ALL_INPUT" {
		t.Errorf("realtimeInputConfig = %v, want the turn coverage", input)
	}
}

// Compression that is not enabled, and VAD with nothing set, are left out.
func TestDisabledCompressionAndEmptyVADAreLeftOut(t *testing.T) {
	setup := setupOf(t, New(Config{
		APIKey: "k", ContextWindowCompression: &ContextWindowCompression{}, VAD: &VADParams{},
	}))
	for _, key := range []string{"contextWindowCompression", "realtimeInputConfig", "proactivity"} {
		if v, ok := setup[key]; ok {
			t.Errorf("%s = %v, want it left out", key, v)
		}
	}
}

// A settings update is stored, so the next session opens with it, and reported
// as not applied to the session already open.
func TestASettingsUpdateIsStoredAndReportedAsNotApplied(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	s, _ := newTestService(t)
	updateSettings(t, s, &Settings{Voice: settings.Set("Kore")})

	if got := s.live.Voice.Or(""); got != "Kore" {
		t.Errorf("voice = %q, want Kore", got)
	}
	gen, _ := setupOf(t, s)["generationConfig"].(map[string]any)
	speech, _ := gen["speechConfig"].(map[string]any)
	voice, _ := speech["voiceConfig"].(map[string]any)["prebuiltVoiceConfig"].(map[string]any)
	if voice["voiceName"] != "Kore" {
		t.Errorf("setup voice = %v, want the updated one", voice["voiceName"])
	}
	if out := buf.String(); !strings.Contains(out, "not currently supported") || !strings.Contains(out, "voice") {
		t.Errorf("log = %q, want the voice reported as not applied", out)
	}
}

// Asking for the text modality is reported, since recent models may refuse it.
func TestTheTextModalityIsReported(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	New(Config{APIKey: "k", Modalities: ModalityText})
	if !strings.Contains(buf.String(), "TEXT modality") {
		t.Errorf("log = %q, want the TEXT modality reported", buf.String())
	}
}
