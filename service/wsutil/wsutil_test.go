package wsutil_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsutil"
)

// wsURL turns an httptest server URL into a WebSocket one.
func wsURL(srv *httptest.Server) string {
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// echoServer accepts a WebSocket and echoes each message back, recording the
// headers of the handshake request.
func echoServer(t *testing.T, got *http.Header) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got != nil {
			*got = r.Header.Clone()
		}
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if err := c.Write(r.Context(), typ, data); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestDialSendsHeaders checks the vendor headers a provider supplies — in
// practice its API key — reach the handshake.
func TestDialSendsHeaders(t *testing.T) {
	var seen http.Header
	srv := echoServer(t, &seen)

	h := http.Header{}
	h.Set("Authorization", "Token secret")
	conn, err := wsutil.Dial(t.Context(), wsURL(srv), h, 0)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if got := seen.Get("Authorization"); got != "Token secret" {
		t.Errorf("Authorization = %q, want the supplied header", got)
	}
}

// TestDialRoundTrip checks the returned connection is usable.
func TestDialRoundTrip(t *testing.T) {
	srv := echoServer(t, nil)

	conn, err := wsutil.Dial(t.Context(), wsURL(srv), nil, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := conn.Write(t.Context(), websocket.MessageText, []byte("ping")); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, data, err := conn.Read(t.Context())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(data) != "ping" {
		t.Errorf("echo = %q, want %q", data, "ping")
	}
}

// TestDialAppliesReadLimit checks a positive limit is enforced, so a vendor
// cannot exhaust memory with an oversized message.
func TestDialAppliesReadLimit(t *testing.T) {
	srv := echoServer(t, nil)

	conn, err := wsutil.Dial(t.Context(), wsURL(srv), nil, 16)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := conn.Write(t.Context(), websocket.MessageText, make([]byte, 64)); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, err := conn.Read(t.Context()); err == nil {
		t.Error("reading a message past the limit should fail")
	}
}

// TestDialZeroReadLimitLeavesDefault checks a non-positive limit is not applied,
// leaving the library default in place rather than pinning it to zero.
func TestDialZeroReadLimitLeavesDefault(t *testing.T) {
	srv := echoServer(t, nil)

	conn, err := wsutil.Dial(t.Context(), wsURL(srv), nil, 0)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := conn.Write(t.Context(), websocket.MessageText, make([]byte, 4096)); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, err := conn.Read(t.Context()); err != nil {
		t.Errorf("read: %v; a zero limit should not restrict message size", err)
	}
}

// TestDialFailure checks a failed handshake reports the error and no connection,
// rather than a half-open one.
func TestDialFailure(t *testing.T) {
	// A plain HTTP handler never upgrades, so the handshake fails.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()

	conn, err := wsutil.Dial(t.Context(), wsURL(srv), nil, 0)
	if err == nil {
		_ = conn.CloseNow()
		t.Fatal("Dial succeeded against a non-WebSocket endpoint, want an error")
	}
	if conn != nil {
		t.Error("Dial returned a connection alongside an error")
	}
}
