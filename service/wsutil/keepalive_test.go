package wsutil

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// peer starts a WebSocket server that either reads (and so answers pings) or
// holds the connection without reading (and so never answers one), and keeps
// the connection open until the test ends.
func peer(t *testing.T, answers bool) string {
	t.Helper()
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		if answers {
			// A read answers each ping with a pong as it goes.
			go func() {
				for {
					if _, _, err := c.Read(context.Background()); err != nil {
						return
					}
				}
			}()
		}
		<-done
	}))
	t.Cleanup(func() {
		close(done)
		srv.Close()
	})
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// readResult runs one read on conn and returns its error once it ends.
func readResult(conn *Conn) <-chan error {
	errc := make(chan error, 1)
	go func() {
		_, _, err := conn.Read(context.Background())
		errc <- err
	}()
	return errc
}

// TestKeepaliveDropsAPeerThatStopsAnswering checks a connection whose peer no
// longer answers pings is dropped, failing the read in progress, rather than
// waiting on a dead connection until TCP gives up.
func TestKeepaliveDropsAPeerThatStopsAnswering(t *testing.T) {
	conn, err := dial(t.Context(), peer(t, false), nil, 0, 20*time.Millisecond, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.CloseNow() })

	select {
	case err := <-readResult(conn):
		if err == nil {
			t.Error("read ended without an error, want the dropped connection reported")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the read is still waiting on a peer that stopped answering")
	}
}

// TestKeepaliveKeepsAnAnsweringPeer checks a peer that answers its pings keeps
// the connection over several of them.
func TestKeepaliveKeepsAnAnsweringPeer(t *testing.T) {
	conn, err := dial(t.Context(), peer(t, true), nil, 0, 10*time.Millisecond, 200*time.Millisecond)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.CloseNow() })

	select {
	case err := <-readResult(conn):
		t.Fatalf("read ended with %v, want the connection kept", err)
	case <-time.After(300 * time.Millisecond):
	}
}

// TestKeepaliveLeavesAnUnreadConnection checks a connection nothing is reading
// is not dropped: the library takes in a pong only during a read, so a ping
// unanswered then says nothing about the peer.
func TestKeepaliveLeavesAnUnreadConnection(t *testing.T) {
	conn, err := dial(t.Context(), peer(t, true), nil, 0, 10*time.Millisecond, 20*time.Millisecond)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.CloseNow() })

	time.Sleep(150 * time.Millisecond)
	if err := conn.Write(t.Context(), websocket.MessageText, []byte("still here")); err != nil {
		t.Errorf("write = %v, want the unread connection still open", err)
	}
}

// TestCloseStopsTheKeepalive checks closing the connection ends its pings.
func TestCloseStopsTheKeepalive(t *testing.T) {
	conn, err := dial(t.Context(), peer(t, true), nil, 0, time.Hour, time.Hour)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	_ = conn.CloseNow()

	select {
	case <-conn.keepaliveDone:
	case <-time.After(time.Second):
		t.Error("the keepalive is still running after the connection closed")
	}
}
