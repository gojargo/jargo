package minimax_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/minimax"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "empty", Cfg: minimax.Config{}, Valid: false},
		{Name: "missing group id", Cfg: minimax.Config{APIKey: "k"}, Valid: false},
		{Name: "missing API key", Cfg: minimax.Config{GroupID: "g"}, Valid: false},
		{Name: "both credentials", Cfg: minimax.Config{APIKey: "k", GroupID: "g"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "MiniMaxTTS", minimax.NewTTS(minimax.Config{APIKey: "k", GroupID: "g"}))
}
