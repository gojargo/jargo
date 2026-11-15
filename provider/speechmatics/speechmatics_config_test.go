package speechmatics_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/speechmatics"
)

// TestConfigValidateSTT pins which Config fields the provider requires.
func TestConfigValidateSTT(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: speechmatics.Config{}, Valid: false},
		{Name: "API key only", Cfg: speechmatics.Config{APIKey: "k"}, Valid: true},
		{
			Name:  "VAD turn detection",
			Cfg:   speechmatics.Config{APIKey: "k", TurnDetectionMode: speechmatics.TurnDetectionVAD},
			Valid: true,
		},
		{
			Name:  "unknown turn detection mode",
			Cfg:   speechmatics.Config{APIKey: "k", TurnDetectionMode: "smart"},
			Valid: false,
		},
		{
			Name:  "operating point naming the model",
			Cfg:   speechmatics.Config{APIKey: "k", Model: "linden-1", OperatingPoint: "linden-1"},
			Valid: true,
		},
		{
			Name:  "operating point naming another model",
			Cfg:   speechmatics.Config{APIKey: "k", Model: "linden-1", OperatingPoint: "other"},
			Valid: false,
		},
	})
}

// TestConfigValidateTTS pins which TTSConfig fields the provider requires.
func TestConfigValidateTTS(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: speechmatics.TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: speechmatics.TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "SpeechmaticsSTT", speechmatics.NewSTT(speechmatics.Config{APIKey: "k"}))
	providertest.Service(t, "SpeechmaticsTTS", speechmatics.NewTTS(speechmatics.TTSConfig{APIKey: "k"}))
}
