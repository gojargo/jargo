package mistral_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/mistral"
)

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: mistral.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: mistral.STTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: mistral.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: mistral.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "MistralSTT", mistral.NewSTT(mistral.STTConfig{APIKey: "k"}))
	providertest.Service(t, "MistralTTS", mistral.NewTTS(mistral.TTSConfig{APIKey: "k"}))
}

// TestNewLLM checks the Mistral OpenAI-compatible LLM shim wires the right
// service name and default model into the shared client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "MistralLLM", "mistral-small-latest", mistral.NewLLM)
}
