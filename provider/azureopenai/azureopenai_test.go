package azureopenai

import (
	"context"
	"net/http"
	"testing"

	"github.com/gojargo/jargo/provider/openai"
)

func TestShaperEndpointAndAuth(t *testing.T) {
	s := shaper{apiVersion: "2024-10-21"}

	base := "https://r.openai.azure.com/openai/deployments/gpt4o"
	want := base + "/chat/completions?api-version=2024-10-21"
	if got := s.Endpoint(base); got != want {
		t.Errorf("Endpoint = %q, want %q", got, want)
	}

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, want, nil)
	s.Authorize(req, "secret")
	if got := req.Header.Get("api-key"); got != "secret" {
		t.Errorf("api-key header = %q, want %q", got, "secret")
	}
	if req.Header.Get("Authorization") != "" {
		t.Error("Authorization header set, want none for Azure")
	}
}

func TestConfigValidate(t *testing.T) {
	valid := Config{
		Endpoint:   "https://r.openai.azure.com",
		Deployment: "gpt4o",
		LLMConfig:  openai.LLMConfig{APIKey: "k"},
	}
	if err := valid.Validate(); err != nil {
		t.Errorf("Validate() valid: %v", err)
	}
	if err := (Config{Deployment: "gpt4o"}).Validate(); err == nil {
		t.Error("Validate() missing Endpoint: want error, got nil")
	}
	if err := (Config{Endpoint: "https://r.openai.azure.com"}).Validate(); err == nil {
		t.Error("Validate() missing Deployment: want error, got nil")
	}
}

func TestNewLLM(t *testing.T) {
	svc := NewLLM(Config{
		Endpoint:   "https://r.openai.azure.com",
		Deployment: "gpt4o",
		LLMConfig:  openai.LLMConfig{APIKey: "k"},
	})
	if svc == nil {
		t.Fatal("NewLLM returned nil")
	}
}
