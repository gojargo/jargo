package polly_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/polly"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "empty config falls back to the AWS credential chain", Cfg: polly.Config{}, Valid: true},
		{Name: "known engine", Cfg: polly.Config{Engine: "generative"}, Valid: true},
		{Name: "unknown engine", Cfg: polly.Config{Engine: "experimental"}, Valid: false},
		{Name: "supported PCM rate", Cfg: polly.Config{SampleRate: 16000}, Valid: true},
		{Name: "unsupported PCM rate", Cfg: polly.Config{SampleRate: 24000}, Valid: false},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "PollyTTS", polly.NewTTS(polly.Config{}))
}
