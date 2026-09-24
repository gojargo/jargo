package live

import (
	"context"
	"strings"
	"testing"
)

// The api key travels in a header, where a failed dial cannot put it in the
// error it reports, and never in the URL.
func TestTheAPIKeyTravelsInAHeader(t *testing.T) {
	c := apiKeyConnector{baseURL: "wss://example.test/ws", apiKey: "sk-secret"}
	endpoint, header, err := c.Endpoint(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(endpoint, "sk-secret") {
		t.Fatalf("endpoint %q carries the key", endpoint)
	}
	if got := header.Get("x-goog-api-key"); got != "sk-secret" {
		t.Fatalf("x-goog-api-key = %q, want the configured key", got)
	}
}
