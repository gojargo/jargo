package elevenlabs_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/elevenlabs"
)

// TestConfigValidateTTS pins which Config fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: elevenlabs.Config{}, Valid: false},
		{Name: "API key only", Cfg: elevenlabs.Config{APIKey: "k"}, Valid: true},
	})
}

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: elevenlabs.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: elevenlabs.STTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "ElevenLabsTTS", elevenlabs.NewTTS(elevenlabs.Config{APIKey: "k"}))
	providertest.Service(t, "ElevenLabsSTT", elevenlabs.NewSTT(elevenlabs.STTConfig{APIKey: "k"}))
}
