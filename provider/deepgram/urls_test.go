package deepgram

import "testing"

// TestDeriveDeepgramURLs ports upstream's table. The streaming and REST
// endpoints have to agree on how secure the connection is, so the scheme the
// caller chose is carried into both, and the port and path come along with the
// host.
func TestDeriveDeepgramURLs(t *testing.T) {
	for _, tc := range []struct {
		base     string
		wantWS   string
		wantHTTP string
	}{
		// Secure schemes.
		{"wss://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"},
		{"https://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"},
		// Insecure schemes, for air-gapped deployments.
		{"ws://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"},
		{"http://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"},
		// A bare hostname defaults to secure.
		{"mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"},
		// With a port.
		{"ws://localhost:8080", "ws://localhost:8080", "http://localhost:8080"},
		{"wss://localhost:443", "wss://localhost:443", "https://localhost:443"},
		{"localhost:8080", "wss://localhost:8080", "https://localhost:8080"},
		// With a path.
		{"wss://host/v1/listen", "wss://host/v1/listen", "https://host/v1/listen"},
		{"http://host/v1/listen", "ws://host/v1/listen", "http://host/v1/listen"},
	} {
		ws, http := deriveDeepgramURLs(tc.base)
		if ws != tc.wantWS || http != tc.wantHTTP {
			t.Errorf("deriveDeepgramURLs(%q) = %q, %q, want %q, %q",
				tc.base, ws, http, tc.wantWS, tc.wantHTTP)
		}
	}
}

// TestDeriveDeepgramURLsUnknownScheme checks a scheme neither streaming nor REST
// can use falls back to the secure pair rather than being passed on to fail at
// dial time.
func TestDeriveDeepgramURLsUnknownScheme(t *testing.T) {
	ws, httpURL := deriveDeepgramURLs("ftp://mydeepgram.com")
	if ws != "wss://mydeepgram.com" || httpURL != "https://mydeepgram.com" {
		t.Errorf("deriveDeepgramURLs = %q, %q, want the secure pair", ws, httpURL)
	}
}

// TestSTTDefaultHost checks an unconfigured service addresses Deepgram's own
// host, securely.
func TestSTTDefaultHost(t *testing.T) {
	ws, _ := deriveDeepgramURLs(defaultHost)
	if want := "wss://" + defaultHost; ws != want {
		t.Errorf("default listen host = %q, want %q", ws, want)
	}
}
