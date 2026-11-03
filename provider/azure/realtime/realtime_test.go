package realtime

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

const baseURL = "wss://my-resource.openai.azure.com/openai/v1/realtime"

// Ports of tests/test_azure_realtime.py.

func TestAuthHeadersUseAPIKey(t *testing.T) {
	c := &connector{baseURL: baseURL, apiKey: "test-key"}
	_, header, err := c.Endpoint(context.Background())
	if err != nil {
		t.Fatalf("endpoint: %v", err)
	}
	if header.Get("api-key") != "test-key" || header.Get("Authorization") != "" {
		t.Errorf("header = %v, want the api-key alone", header)
	}
}

func TestAuthHeadersPreferTokenProvider(t *testing.T) {
	c := &connector{baseURL: baseURL, apiKey: "test-key", tokenProvider: func(context.Context) (string, error) {
		return "entra-token", nil
	}}
	_, header, err := c.Endpoint(context.Background())
	if err != nil {
		t.Fatalf("endpoint: %v", err)
	}
	if header.Get("Authorization") != "Bearer entra-token" || header.Get("api-key") != "" {
		t.Errorf("header = %v, want the bearer token alone", header)
	}
}

// The tests below are not ports of upstream tests.

// Either an API key or a token provider is required, and the URL always.
func TestConfigValidate(t *testing.T) {
	token := func(context.Context) (string, error) { return "t", nil }
	for _, tc := range []struct {
		cfg   Config
		valid bool
	}{
		{Config{BaseURL: baseURL, APIKey: "k"}, true},
		{Config{BaseURL: baseURL, TokenProvider: token}, true},
		{Config{BaseURL: baseURL}, false},
		{Config{APIKey: "k"}, false},
	} {
		if err := tc.cfg.Validate(); (err == nil) != tc.valid {
			t.Errorf("Validate(%+v) = %v, want valid=%v", tc.cfg, err, tc.valid)
		}
	}
}

// The deployment is appended as the model, unless the URL already carries a
// query string, which is used as it stands.
func TestTheDeploymentIsNamedByTheModel(t *testing.T) {
	svc := New(Config{BaseURL: baseURL, APIKey: "k", Model: "my-deployment"})
	if got := svc.Model(); got != "my-deployment" {
		t.Errorf("model = %q, want the deployment", got)
	}
	c := &connector{baseURL: baseURL, model: "my-deployment", apiKey: "k"}
	if got, _, _ := c.Endpoint(context.Background()); got != baseURL+"?model=my-deployment" {
		t.Errorf("endpoint = %q, want the deployment appended", got)
	}
	verbatim := baseURL + "?model=other"
	c = &connector{baseURL: verbatim, model: "my-deployment", apiKey: "k"}
	if got, _, _ := c.Endpoint(context.Background()); got != verbatim {
		t.Errorf("endpoint = %q, want the URL as given", got)
	}
}

// A URL on the preview surface is reported when the service is built.
func TestAPreviewURLIsReported(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	New(Config{BaseURL: "wss://r.openai.azure.com/openai/realtime?api-version=2025-04-01-preview", APIKey: "k"})
	if !strings.Contains(buf.String(), "preview realtime API") {
		t.Errorf("log = %q, want the preview surface reported", buf.String())
	}
}
