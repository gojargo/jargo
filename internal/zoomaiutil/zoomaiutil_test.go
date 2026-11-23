package zoomaiutil

import (
	"bytes"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
)

// lockedWriter serializes writes to a buffer several goroutines log to.
type lockedWriter struct {
	mu *sync.Mutex
	w  *bytes.Buffer
}

func (l lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

// captureWarnings records what is logged at warning level and above from the
// call on, until the test ends.
func captureWarnings(t *testing.T) func() string {
	t.Helper()
	var mu sync.Mutex
	var buf bytes.Buffer
	prev := slog.Default()
	handler := slog.NewTextHandler(lockedWriter{mu: &mu, w: &buf}, &slog.HandlerOptions{Level: slog.LevelWarn})
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		return buf.String()
	}
}

// A plaintext endpoint on another host is warned about; one on this machine is
// not.
func TestCheckURLWarnsAboutAPlaintextRemoteEndpoint(t *testing.T) {
	for endpoint, wantWarning := range map[string]bool{
		"https://api.zoom.us/v2/aiservices/tts/synthesize": false,
		"http://localhost:8080/tts":                        false,
		"http://127.0.0.1:8080/tts":                        false,
		"http://tts.example.com/tts":                       true,
	} {
		logs := captureWarnings(t)
		if got := CheckURL(endpoint, "Zoom TTS", "text"); got != endpoint {
			t.Errorf("CheckURL(%q) = %q, want the URL back", endpoint, got)
		}
		if warned := strings.Contains(logs(), "not encrypted"); warned != wantWarning {
			t.Errorf("CheckURL(%q) warned = %t, want %t", endpoint, warned, wantWarning)
		}
	}
}

func TestSetRequestHeadersCarriesTheKeyTheClientAndTheTrackingID(t *testing.T) {
	trackingID := NewTrackingID()
	if !strings.HasPrefix(trackingID, "zoom_ai_community_jargo_") || trackingID == NewTrackingID() {
		t.Fatalf("tracking ID = %q, want a new one with the prefix", trackingID)
	}
	header := http.Header{}
	SetRequestHeaders(header, "key", trackingID)
	if header.Get("Authorization") != "Bearer key" || header.Get("User-Agent") != "jargo" ||
		header.Get(HeaderTrackingID) != trackingID {
		t.Fatalf("headers = %v", header)
	}
}
