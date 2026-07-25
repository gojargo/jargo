package inworld_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/inworld"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: inworld.Config{}, Valid: false},
		{Name: "API key only", Cfg: inworld.Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "InworldTTS", inworld.NewTTS(inworld.Config{APIKey: "k"}))
}
