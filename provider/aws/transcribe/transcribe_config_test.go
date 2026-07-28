package transcribe_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/aws/transcribe"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "empty config falls back to the AWS credential chain", Cfg: transcribe.Config{}, Valid: true},
		{
			Name:  "static credentials",
			Cfg:   transcribe.Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"},
			Valid: true,
		},
		{Name: "known stability level", Cfg: transcribe.Config{PartialResultsStability: "medium"}, Valid: true},
		{Name: "unknown stability level", Cfg: transcribe.Config{PartialResultsStability: "highest"}, Valid: false},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "TranscribeSTT", transcribe.NewSTT(transcribe.Config{}))
}
