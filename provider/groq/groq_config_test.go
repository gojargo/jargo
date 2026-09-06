package groq_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/groq"
)

// TestConfigValidateSTT pins which STTConfig fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: groq.STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: groq.STTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: groq.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: groq.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "GroqSTT", groq.NewSTT(groq.STTConfig{APIKey: "k"}))
	providertest.Service(t, "GroqTTS", groq.NewTTS(groq.TTSConfig{APIKey: "k"}))
}

// TestNewLLM checks the Groq OpenAI-compatible LLM shim wires the right
// service name and default model into the shared client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "GroqLLM", "openai/gpt-oss-120b", groq.NewLLM)
}
