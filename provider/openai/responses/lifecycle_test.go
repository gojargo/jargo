package responses

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsservice"
)

// refusingWS is a Responses WebSocket endpoint that refuses its first
// handshakes, then answers every request with a short text reply.
type refusingWS struct {
	*httptest.Server
	mu     sync.Mutex
	dials  int
	refuse int
}

func newRefusingWS(t *testing.T, refuse int) *refusingWS {
	t.Helper()
	f := &refusingWS{refuse: refuse}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.dials++
		refused := f.refuse > 0
		if refused {
			f.refuse--
		}
		f.mu.Unlock()
		if refused {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
			for _, ev := range textTurn("Hello") {
				if c.Write(r.Context(), websocket.MessageText, []byte(ev)) != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *refusingWS) url() string { return "ws" + strings.TrimPrefix(f.URL, "http") }

func (f *refusingWS) dialed() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.dials
}

// quickService is a WebSocket service on url whose reconnect backoff costs no
// real time.
func quickService(t *testing.T, url string) *Service {
	t.Helper()
	svc := NewLLM(Config{APIKey: "k", WSURL: url})
	svc.ws = llm.NewWebsocket(svc.Base, svc, wsservice.Config{
		ReconnectBackoffMinWait: time.Millisecond, ReconnectBackoffMaxWait: time.Millisecond,
	})
	t.Cleanup(svc.disconnect)
	return svc
}

// TestConnectsAtSetup checks the connection opens when the service is set up,
// so the first turn does not pay for the handshake.
func TestConnectsAtSetup(t *testing.T) {
	srv := newRefusingWS(t, 0)
	svc := quickService(t, srv.url())

	if err := svc.Setup(t.Context(), processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	if got := srv.dialed(); got != 1 {
		t.Errorf("connections opened at setup = %d, want 1", got)
	}
	if !svc.Connected() {
		t.Error("Connected = false after setup, want the connection open")
	}
}

// TestDisconnectsWhenThePipelineEnds checks an EndFrame closes the connection,
// and that it stays closed.
func TestDisconnectsWhenThePipelineEnds(t *testing.T) {
	for _, f := range []frames.Frame{frames.NewEndFrame(), frames.NewCancelFrame()} {
		srv := newRefusingWS(t, 0)
		svc := quickService(t, srv.url())
		if err := svc.Setup(t.Context(), processor.Setup{Clock: clock.NewSystem()}); err != nil {
			t.Fatalf("Setup: %v", err)
		}
		_ = svc.ProcessFrame(t.Context(), f, processor.Downstream)
		if svc.Connected() {
			t.Errorf("%T: Connected = true, want the connection closed", f)
		}
		if svc.ws.Reconnect(t.Context()) {
			t.Errorf("%T: a connection closed on purpose was restored", f)
		}
	}
}

// TestRefusedHandshakeIsRetried checks a refused handshake is retried with
// backoff before the turn gives up, so one refusal does not cost the reply.
func TestRefusedHandshakeIsRetried(t *testing.T) {
	srv := newRefusingWS(t, 1)
	svc := quickService(t, srv.url())

	var text strings.Builder
	err := svc.Generate(context.Background(), userConvo("hi"), func(s string) error {
		text.WriteString(s)
		return nil
	})
	if err != nil || text.String() != "Hello" {
		t.Errorf("turn = (%q, %v) after %d dials, want Hello after a retry", text.String(), err, srv.dialed())
	}
}

// TestUnreachableEndpointMakesTheServiceUnusable checks an endpoint that keeps
// refusing is given up on after the reconnect attempts run out, and that doing
// so reports a permanent error, which leaves the service unusable rather than
// failing every turn forever.
func TestUnreachableEndpointMakesTheServiceUnusable(t *testing.T) {
	srv := newRefusingWS(t, 1000)
	svc := quickService(t, srv.url())

	err := svc.Generate(context.Background(), userConvo("hi"), func(string) error { return nil })
	if !errors.Is(err, llm.ErrWebsocketNotConnected) {
		t.Errorf("Generate = %v, want ErrWebsocketNotConnected", err)
	}
	if got := srv.dialed(); got != wsservice.DefaultMaxRetries {
		t.Errorf("dialed %d times, want %d attempts", got, wsservice.DefaultMaxRetries)
	}
	if svc.Usable() {
		t.Error("Usable = true, want the service given up on")
	}
}
