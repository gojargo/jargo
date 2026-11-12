package tts

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/gojargo/jargo/service/wsservice"
)

// hookedWSSynth records the hooks the base calls around a redial and a
// disconnect, with whether the turn's context had been replaced yet.
type hookedWSSynth struct {
	*wsSynth
	b *Base

	hmu   sync.Mutex
	hooks []string
	turn  string
}

func (s *hookedWSSynth) PrepareReconnectWebsocket(context.Context) { s.hook("prepare") }

func (s *hookedWSSynth) CloseConnectionContexts() { s.hook("close contexts") }

func (s *hookedWSSynth) DisconnectWebsocket(ctx context.Context) error {
	s.hook("disconnect websocket")
	return s.wsSynth.DisconnectWebsocket(ctx)
}

func (s *hookedWSSynth) hook(name string) {
	s.hmu.Lock()
	defer s.hmu.Unlock()
	if s.b.turnContextID() != s.turn {
		name += " (contexts ended)"
	}
	s.hooks = append(s.hooks, name)
}

// TestARedialPreparesBeforeTheContextsEnd checks the service prepares a redial
// before the base ends the replaced connection's contexts, and is told about
// those contexts just before they end, on a redial and on a disconnect alike.
func TestARedialPreparesBeforeTheContextsEnd(t *testing.T) {
	syn := &hookedWSSynth{wsSynth: newWSSynth()}
	b := New("WebsocketTTS", syn)
	syn.b = b
	syn.ws = NewWebsocket(b, syn, wsservice.Options{})
	b.setTurnContextID("turn")
	syn.turn = "turn"

	if !syn.ws.TryReconnect(t.Context()) {
		t.Fatal("the redial failed")
	}
	syn.hmu.Lock()
	syn.turn = b.turnContextID()
	syn.hmu.Unlock()
	syn.ws.Disconnect()

	want := []string{
		"prepare", "close contexts", "disconnect websocket (contexts ended)",
		"close contexts",
	}
	syn.hmu.Lock()
	defer syn.hmu.Unlock()
	if !slices.Equal(syn.hooks, want) {
		t.Errorf("hooks = %q, want %q", syn.hooks, want)
	}
}
