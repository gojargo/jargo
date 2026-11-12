package deepgram

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	errs "github.com/gojargo/jargo/utils/errors"
)

// refusingServer refuses every WebSocket handshake with status, explaining it
// in the dg-error header when reason is set.
func refusingServer(t *testing.T, status int, reason string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if reason != "" {
			w.Header().Set("dg-error", reason)
		}
		w.WriteHeader(status)
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// TestFluxSTTRefusalReportsDeepgramsReason checks Deepgram's reason for
// refusing the handshake reaches the error, which keeps the status it is
// classified by.
func TestFluxSTTRefusalReportsDeepgramsReason(t *testing.T) {
	reason := "eot_threshold must be between 0.5 and 1, inclusive."
	c := newFluxConnector(FluxConfig{
		APIKey: "k", Model: defaultFluxModel,
		ListenURL: refusingServer(t, http.StatusBadRequest, reason),
	})

	_, err := c.Connect(t.Context(), 16000)

	if want := "Deepgram refused the connection (HTTP 400): " + reason; err == nil || err.Error() != want {
		t.Errorf("error = %v, want %q", err, want)
	}
	if got := errs.ClassifyError(err); !got.IsPermanent() {
		t.Errorf("category = %v, want a permanent one", got)
	}
}
