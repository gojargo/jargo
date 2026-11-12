package wsservice_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsservice"
)

// A connection opened through the options takes the close timeout they give:
// zero sends the close without waiting on a peer that never answers it.
func TestOptionsDialAppliesTheCloseTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		<-r.Context().Done() // deaf: never reads, so never answers a close
	}))
	defer srv.Close()

	zero := time.Duration(0)
	opts := wsservice.Options{CloseTimeout: &zero}
	conn, err := opts.Dial(t.Context(), "ws"+strings.TrimPrefix(srv.URL, "http"), nil, 0)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	start := time.Now()
	_ = conn.Close(websocket.StatusNormalClosure, "")
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond {
		t.Errorf("close took %v with a zero close timeout, want no wait", elapsed)
	}
}

// The reconnection part of the options is what a base is built with.
func TestOptionsConfigCarriesTheReconnection(t *testing.T) {
	off := false
	cfg := wsservice.Options{
		ReconnectOnError: &off, ReconnectBackoffMinWait: time.Second, ReconnectBackoffMaxWait: 3 * time.Second,
	}.Config()
	if cfg.ReconnectOnError == nil || *cfg.ReconnectOnError || cfg.ReconnectBackoffMinWait != time.Second ||
		cfg.ReconnectBackoffMaxWait != 3*time.Second {
		t.Errorf("config = %+v, want the options' reconnection", cfg)
	}
}
