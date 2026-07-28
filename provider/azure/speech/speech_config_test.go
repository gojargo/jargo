package speech_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/azure/speech"
)

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: speech.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: speech.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "AzureTTS", speech.NewTTS(speech.TTSConfig{APIKey: "k"}))
}
