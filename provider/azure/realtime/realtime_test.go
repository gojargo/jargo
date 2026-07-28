package realtime

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
)

// TestConfigValidate pins which Config fields the provider requires: either the
// resource plus deployment, or a whole URL.
func TestConfigValidate(t *testing.T) {
	const resource = "https://my-resource.openai.azure.com"
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing everything", Cfg: Config{}, Valid: false},
		{
			Name:  "key, endpoint and deployment",
			Cfg:   Config{APIKey: "k", Endpoint: resource, Deployment: "rt"},
			Valid: true,
		},
		{Name: "key and URL", Cfg: Config{APIKey: "k", URL: "wss://custom/realtime?x=1"}, Valid: true},
		{Name: "endpoint without deployment", Cfg: Config{APIKey: "k", Endpoint: resource}, Valid: false},
		{Name: "deployment without endpoint", Cfg: Config{APIKey: "k", Deployment: "rt"}, Valid: false},
		{Name: "no key", Cfg: Config{Endpoint: resource, Deployment: "rt"}, Valid: false},
	})
}

// TestNew checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNew(t *testing.T) {
	svc := New(Config{APIKey: "k", Endpoint: "https://r.openai.azure.com", Deployment: "rt"})
	providertest.Service(t, "AzureRealtime", svc)
}

// TestEndpoint checks the session URL: Azure selects the model with a deployment
// query parameter, and the resource endpoint can be pasted from the portal in
// its https form.
func TestEndpoint(t *testing.T) {
	t.Run("derived from the resource", func(t *testing.T) {
		got := endpoint(Config{
			Endpoint:   "https://my-resource.openai.azure.com/",
			Deployment: "my-rt",
		})
		if !strings.HasPrefix(got, "wss://my-resource.openai.azure.com"+realtimePath+"?") {
			t.Fatalf("endpoint = %q, want the wss resource URL with no double slash", got)
		}
		q := parseQuery(t, got)
		if q.Get("deployment") != "my-rt" {
			t.Errorf("deployment = %q, want my-rt", q.Get("deployment"))
		}
		if q.Get("api-version") != defaultAPIVersion {
			t.Errorf("api-version = %q, want the default", q.Get("api-version"))
		}
		if q.Has("model") {
			t.Error("a model query parameter is present, but Azure selects it by deployment")
		}
	})

	t.Run("configured api version", func(t *testing.T) {
		got := endpoint(Config{
			Endpoint:   "https://r.openai.azure.com",
			Deployment: "d",
			APIVersion: "2099-01-01",
		})
		if q := parseQuery(t, got); q.Get("api-version") != "2099-01-01" {
			t.Errorf("api-version = %q, want the configured one", q.Get("api-version"))
		}
	})

	t.Run("explicit URL wins", func(t *testing.T) {
		const raw = "wss://custom.example/openai/realtime?api-version=x&deployment=y"
		if got := endpoint(Config{Endpoint: "https://ignored", Deployment: "ignored", URL: raw}); got != raw {
			t.Errorf("endpoint = %q, want the configured URL verbatim", got)
		}
	})
}

// TestWSScheme checks an endpoint copied from the portal is dialed over
// WebSocket, and that one already in ws form is left alone.
func TestWSScheme(t *testing.T) {
	cases := map[string]string{
		"https://r.openai.azure.com": "wss://r.openai.azure.com",
		"http://localhost:8080":      "ws://localhost:8080",
		"wss://r.openai.azure.com":   "wss://r.openai.azure.com",
		"ws://localhost:8080":        "ws://localhost:8080",
	}
	for in, want := range cases {
		if got := wsScheme(in); got != want {
			t.Errorf("wsScheme(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestConnectorAuthorizes checks Azure's api-key header is sent instead of a
// bearer token.
func TestConnectorAuthorizes(t *testing.T) {
	c := &connector{apiKey: "secret", url: "wss://r/realtime?deployment=d"}
	got, header, err := c.Endpoint(context.Background())
	if err != nil {
		t.Fatalf("Endpoint: %v", err)
	}
	if got != "wss://r/realtime?deployment=d" {
		t.Errorf("endpoint = %q, want the configured URL", got)
	}
	if h := header.Get("api-key"); h != "secret" {
		t.Errorf("api-key = %q, want the resource key", h)
	}
	if h := header.Get("Authorization"); h != "" {
		t.Errorf("Authorization = %q, want it absent: Azure uses the api-key header", h)
	}
	if h := header.Get("OpenAI-Beta"); h != "" {
		t.Errorf("OpenAI-Beta = %q, want it absent on Azure", h)
	}
}

// parseQuery pulls the query parameters off a built endpoint URL.
func parseQuery(t *testing.T, endpoint string) url.Values {
	t.Helper()
	u, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("endpoint %q is not a URL: %v", endpoint, err)
	}
	return u.Query()
}
