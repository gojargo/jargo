package together_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/together"
)

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: together.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: together.STTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: together.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: together.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "TogetherSTT", together.NewSTT(together.STTConfig{APIKey: "k"}))
	providertest.Service(t, "TogetherTTS", together.NewTTS(together.TTSConfig{APIKey: "k"}))
}

// TestNewLLM checks the Together AI OpenAI-compatible LLM shim wires the right
// service name and default model into the shared client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "TogetherLLM", "zai-org/GLM-5.1", together.NewLLM)
}
