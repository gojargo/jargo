package bedrock

import (
	"context"
	"testing"
)

func TestConfigValidate(t *testing.T) {
	if err := (Config{}).Validate(); err == nil {
		t.Error("Validate() with empty Model: want error, got nil")
	}
	if err := (Config{Model: "us.anthropic.claude-3-5-haiku-20241022-v1:0"}).Validate(); err != nil {
		t.Errorf("Validate() with Model set: %v", err)
	}
}

func TestLoadOptions(t *testing.T) {
	// Region and static credentials both produce load options.
	full := Config{
		Model:           "m",
		Region:          "us-east-1",
		AccessKeyID:     "AKIA",
		SecretAccessKey: "secret",
	}
	if got := len(full.loadOptions()); got != 2 {
		t.Errorf("loadOptions with region+creds = %d, want 2", got)
	}
	// No region and no credentials falls back entirely to the default chain.
	if got := len(Config{Model: "m"}.loadOptions()); got != 0 {
		t.Errorf("loadOptions with defaults = %d, want 0", got)
	}
	// An access key without a secret is ignored (incomplete static credentials).
	partial := Config{Model: "m", AccessKeyID: "AKIA"}
	if got := len(partial.loadOptions()); got != 0 {
		t.Errorf("loadOptions with partial creds = %d, want 0", got)
	}
}

func TestNewLLM(t *testing.T) {
	// Static credentials avoid touching the environment or instance metadata.
	svc, err := NewLLM(context.Background(), Config{
		Model:           "us.anthropic.claude-3-5-haiku-20241022-v1:0",
		Region:          "us-east-1",
		AccessKeyID:     "AKIATEST",
		SecretAccessKey: "secret",
	})
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	if svc == nil {
		t.Fatal("NewLLM returned nil service")
	}
}
