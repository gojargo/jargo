package gradium_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/gradium"
)

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: gradium.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: gradium.STTConfig{APIKey: "k"}, Valid: true},
		{Name: "supported frame delay", Cfg: gradium.STTConfig{APIKey: "k", DelayInFrames: new(24)}, Valid: true},
		{Name: "unsupported frame delay", Cfg: gradium.STTConfig{APIKey: "k", DelayInFrames: new(25)}, Valid: false},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: gradium.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: gradium.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "GradiumSTT", gradium.NewSTT(gradium.STTConfig{APIKey: "k"}))
	providertest.Service(t, "GradiumTTS", gradium.NewTTS(gradium.TTSConfig{APIKey: "k"}))
}
