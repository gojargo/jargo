package hume_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/hume"
)

// TestConfigValidate pins which Config fields the provider requires. A voice is
// among them: Hume invents a new one on every call when none is named, so a
// config without one would not sound like the same bot twice.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "nothing set", Cfg: hume.Config{}, Valid: false},
		{Name: "an API key but no voice", Cfg: hume.Config{APIKey: "k"}, Valid: false},
		{Name: "a voice but no API key", Cfg: hume.Config{VoiceID: "v"}, Valid: false},
		{Name: "both", Cfg: hume.Config{APIKey: "k", VoiceID: "v"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "HumeTTS", hume.NewTTS(hume.Config{APIKey: "k", VoiceID: "v"}))
}
