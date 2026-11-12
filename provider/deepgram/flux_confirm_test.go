package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
)

// confirmingFlux is a fake Flux endpoint that leaves its first connections
// unconfirmed and counts the audio each one receives.
type confirmingFlux struct {
	*httptest.Server
	// unconfirmed is how many connections, from the first, never get
	// Connected; negative for all of them.
	unconfirmed int

	mu    sync.Mutex
	audio []int
}

func newConfirmingFlux(t *testing.T, unconfirmed int) *confirmingFlux {
	t.Helper()
	f := &confirmingFlux{unconfirmed: unconfirmed}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		f.mu.Lock()
		index := len(f.audio)
		f.audio = append(f.audio, 0)
		f.mu.Unlock()
		if f.unconfirmed >= 0 && index >= f.unconfirmed {
			connected, _ := json.Marshal(map[string]any{"type": fluxMsgConnected, "request_id": "test"})
			_ = c.Write(r.Context(), websocket.MessageText, connected)
		}
		for {
			kind, _, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if kind == websocket.MessageBinary {
				f.mu.Lock()
				f.audio[index]++
				f.mu.Unlock()
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// handshakes is how many connections were made, and audio how many audio
// messages each one received.
func (f *confirmingFlux) counts() (handshakes int, audio []int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.audio), append([]int(nil), f.audio...)
}

// runFlux streams 20 ms audio frames through a Flux service for d, against
// the endpoint, with the confirmation wait shortened to wait.
func runFlux(t *testing.T, endpoint string, wait, d time.Duration) *stt.StreamService {
	t.Helper()
	c := newFluxConnector(FluxConfig{
		APIKey: "k", ListenURL: endpoint, Model: defaultFluxModel,
	})
	c.connectionTimeout = wait
	svc := stt.NewStream("DeepgramFluxSTT", c, 16000)
	c.svc = svc

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for range int(d / (20 * time.Millisecond)) {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
		time.Sleep(20 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not shut down")
	}
	return svc
}

// TestFluxRetriesAnUnconfirmedConnection checks that a connection Flux opens
// but never confirms is retried like a dropped one, rather than treated as a
// rejected request on the first try.
func TestFluxRetriesAnUnconfirmedConnection(t *testing.T) {
	t.Parallel()
	server := newConfirmingFlux(t, 1)

	svc := runFlux(t, wsURL(server.URL), 100*time.Millisecond, 600*time.Millisecond)

	handshakes, audio := server.counts()
	if handshakes != 2 {
		t.Errorf("handshakes = %d, want 2", handshakes)
	}
	if len(audio) < 2 || audio[1] == 0 {
		t.Errorf("audio per connection = %v, want the confirmed one to receive audio", audio)
	}
	if !svc.Usable() {
		t.Error("the service is unusable after the retried connection was confirmed")
	}
}

// TestFluxGivesUpOnConnectionsThatNeverConfirm checks the retrying ends after
// three unconfirmed connections in a row, however long each wait took.
func TestFluxGivesUpOnConnectionsThatNeverConfirm(t *testing.T) {
	t.Parallel()
	server := newConfirmingFlux(t, -1)

	svc := runFlux(t, wsURL(server.URL), 100*time.Millisecond, time.Second)

	if handshakes, _ := server.counts(); handshakes != 3 {
		t.Errorf("handshakes = %d, want 3", handshakes)
	}
	if svc.Usable() {
		t.Error("the service is still usable after three unconfirmed connections")
	}
}
