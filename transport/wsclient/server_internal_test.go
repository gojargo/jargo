package wsclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// testServer is a WebSocket server the tests dial. It records how many
// connections were opened and what each one received, and can answer each
// message it reads.
type testServer struct {
	url string

	mu     sync.Mutex
	opened int
	conns  []*serverConn
}

// serverConn is one connection the test server accepted.
type serverConn struct {
	received chan []byte
	closed   chan struct{}
}

// newTestServer serves WebSocket connections until the test ends. reply, when
// set, is called for every message read, with the connection to answer on.
func newTestServer(t *testing.T, reply func(ctx context.Context, conn *websocket.Conn, message []byte)) *testServer {
	t.Helper()
	s := &testServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Counted before the upgrade, so a client that has finished dialing has
		// always been counted.
		sc := &serverConn{received: make(chan []byte, 64), closed: make(chan struct{})}
		s.mu.Lock()
		s.opened++
		s.conns = append(s.conns, sc)
		s.mu.Unlock()

		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			close(sc.closed)
			return
		}
		defer close(sc.closed)
		for {
			_, message, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			sc.received <- message
			if reply != nil {
				reply(r.Context(), conn, message)
			}
		}
	}))
	t.Cleanup(srv.Close)
	s.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return s
}

func (s *testServer) openedCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.opened
}

func (s *testServer) conn(t *testing.T, i int) *serverConn {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if i >= len(s.conns) {
		t.Fatalf("connection %d was never opened", i)
	}
	return s.conns[i]
}

// next waits for the next message the connection receives.
func (c *serverConn) next(t *testing.T) []byte {
	t.Helper()
	select {
	case m := <-c.received:
		return m
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for a message")
		return nil
	}
}

// isClosed reports whether the connection closed within wait.
func (c *serverConn) isClosed(wait time.Duration) bool {
	select {
	case <-c.closed:
		return true
	case <-time.After(wait):
		return false
	}
}
