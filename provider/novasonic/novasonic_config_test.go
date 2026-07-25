package novasonic_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/novasonic"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "empty config falls back to the AWS credential chain", Cfg: novasonic.Config{}, Valid: true},
		{
			Name:  "static credentials",
			Cfg:   novasonic.Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"},
			Valid: true,
		},
	})
}

// TestNewServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "NovaSonic", novasonic.New(novasonic.Config{}))
}
