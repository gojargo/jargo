package cartesia_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// fakeInk2 is an Ink-2 turns server whose failures a test scripts.
type fakeInk2 struct {
	// rejectedHandshakes is how many handshakes, from the first, are refused.
	rejectedHandshakes int
	// rejectAfter refuses every handshake after that many; 0 refuses none.
	rejectAfter int
	// dropFirstAfter drops the first connection, with no close frame, after
	// that many audio frames; 0 never drops it.
	dropFirstAfter int
	// dropMidTurn opens a turn just before that drop.
	dropMidTurn bool
	// resumeTurn has the second connection hear a whole turn of its own.
	resumeTurn bool
	// openTurnAfter opens a turn on the first connection after that many
	// audio frames, ended only when the client asks to close; 0 opens none.
	openTurnAfter int
	// unconfirmed is how many connections, from the first, are never
	// confirmed; 0 confirms every one.
	unconfirmed int
	// rejectSettings refuses the session's settings as the server does.
	rejectSettings bool

	mu         sync.Mutex
	handshakes int
	audio      []int
	srv        *httptest.Server
}

func (f *fakeInk2) start(t *testing.T) string {
	t.Helper()
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return wsURL(f.srv.URL)
}

func (f *fakeInk2) counts() (int, []int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.handshakes, append([]int(nil), f.audio...)
}

func (f *fakeInk2) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.handshakes++
	n := f.handshakes
	f.mu.Unlock()
	if n <= f.rejectedHandshakes || (f.rejectAfter > 0 && n > f.rejectAfter) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
		return
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	f.mu.Lock()
	index := len(f.audio)
	f.audio = append(f.audio, 0)
	f.mu.Unlock()
	send := func(m map[string]any) {
		b, _ := json.Marshal(m)
		_ = c.Write(ctx, websocket.MessageText, b)
	}
	sendTurn := func(typ, tx string) { send(map[string]any{"type": typ, "transcript": tx}) }
	if f.rejectSettings {
		send(map[string]any{
			"type": "error", "error_code": "model_not_found",
			"message": "Could not resolve model.", "status_code": 400,
		})
		_ = c.Close(websocket.StatusPolicyViolation, "Invalid model")
		return
	}
	if index >= f.unconfirmed {
		send(map[string]any{"type": "connected", "request_id": "test"})
	}
	turnOpen := false
	for {
		kind, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		if kind != websocket.MessageBinary {
			var m map[string]any
			if json.Unmarshal(data, &m) == nil && m["type"] == "close" {
				if turnOpen {
					sendTurn("turn.end", "hello there")
				}
				_ = c.Close(websocket.StatusNormalClosure, "")
				return
			}
			continue
		}
		f.mu.Lock()
		f.audio[index]++
		count := f.audio[index]
		f.mu.Unlock()
		if index == 0 && count == f.openTurnAfter {
			sendTurn("turn.start", "hello")
			sendTurn("turn.update", "hello there")
			turnOpen = true
		}
		if index == 1 && f.resumeTurn && count == 3 {
			sendTurn("turn.start", "how are you")
			sendTurn("turn.end", "how are you")
		}
		if index == 0 && count == f.dropFirstAfter {
			if f.dropMidTurn {
				sendTurn("turn.start", "hello")
				sendTurn("turn.update", "hello there")
				time.Sleep(50 * time.Millisecond)
			}
			// transport.abort(): drop the TCP connection with no close frame.
			_ = c.CloseNow()
			return
		}
	}
}
