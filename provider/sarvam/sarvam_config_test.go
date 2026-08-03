package sarvam_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/sarvam"
)

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: sarvam.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: sarvam.STTConfig{APIKey: "k"}, Valid: true},
		{Name: "known model", Cfg: sarvam.STTConfig{APIKey: "k", Model: "saaras:v3"}, Valid: true},
		{Name: "unknown model", Cfg: sarvam.STTConfig{APIKey: "k", Model: "saaras:v9"}, Valid: false},
		{Name: "known mode", Cfg: sarvam.STTConfig{APIKey: "k", Mode: "translate"}, Valid: true},
		{Name: "unknown mode", Cfg: sarvam.STTConfig{APIKey: "k", Mode: "paraphrase"}, Valid: false},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: sarvam.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: sarvam.TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "known model", Cfg: sarvam.TTSConfig{APIKey: "k", Model: "bulbul:v3"}, Valid: true},
		{Name: "unknown model", Cfg: sarvam.TTSConfig{APIKey: "k", Model: "bulbul:v9"}, Valid: false},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "SarvamSTT", sarvam.NewSTT(sarvam.STTConfig{APIKey: "k"}))
	providertest.Service(t, "SarvamTTS", sarvam.NewTTS(sarvam.TTSConfig{APIKey: "k"}))
}

// TestNewLLM checks the Sarvam OpenAI-compatible LLM shim wires the right
// service name and default model into the shared client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "SarvamLLM", "sarvam-105b", sarvam.NewLLM)
}
