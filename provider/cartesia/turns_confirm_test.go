package cartesia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
)

// fakeInk2 is a fake turn-detection endpoint. It leaves its first connections
// unconfirmed, or rejects every connection's settings, and counts the audio
// each connection receives.
type fakeInk2 struct {
	*httptest.Server
	// unconfirmed is how many connections, from the first, never get
	// connected; negative for all of them.
	unconfirmed int
	// rejectSettings answers every connection with a settings error.
	rejectSettings bool

	mu    sync.Mutex
	audio []int
}

func newFakeInk2(t *testing.T, unconfirmed int, rejectSettings bool) *fakeInk2 {
	t.Helper()
	f := &fakeInk2{unconfirmed: unconfirmed, rejectSettings: rejectSettings}
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
		send := func(m map[string]any) {
			b, _ := json.Marshal(m)
			_ = c.Write(r.Context(), websocket.MessageText, b)
		}
		if f.rejectSettings {
			send(map[string]any{
				"type": "error", "error_code": "model_not_found",
				"message": "Could not resolve model.", "status_code": 400,
			})
			_ = c.Close(websocket.StatusPolicyViolation, "Invalid model")
			return
		}
		if f.unconfirmed >= 0 && index >= f.unconfirmed {
			send(map[string]any{"type": "connected", "request_id": "test"})
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

// counts is how many connections were made, and how many audio messages each
// one received.
func (f *fakeInk2) counts() (handshakes int, audio []int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.audio), append([]int(nil), f.audio...)
}

// runTurns streams 20 ms audio frames through a turn-detection service for d,
// against the endpoint, with the confirmation wait shortened to wait. It
// returns the service and the errors that reached the start of the pipeline.
func runTurns(t *testing.T, endpoint string, wait, d time.Duration) (*stt.StreamService, []*frames.ErrorFrame) {
	t.Helper()
	c := newTurnsConnector(TurnsSTTConfig{
		APIKey: "k", URL: "ws" + strings.TrimPrefix(endpoint, "http"), Version: defaultVersion,
		Model: defaultTurnsModel,
	})
	c.connectionTimeout = wait
	svc := stt.NewStream("CartesiaTurnsSTT", c, 16000)
	c.svc = svc

	var mu sync.Mutex
	var errors []*frames.ErrorFrame
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		IdleTimeout: -1, ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errors = append(errors, ef)
			mu.Unlock()
		}
	})
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
	mu.Lock()
	defer mu.Unlock()
	return svc, append([]*frames.ErrorFrame(nil), errors...)
}

// TestTurnsRetriesAnUnconfirmedConnection checks that a connection the server
// opens but never confirms is retried like a dropped one.
func TestTurnsRetriesAnUnconfirmedConnection(t *testing.T) {
	t.Parallel()
	server := newFakeInk2(t, 1, false)

	svc, _ := runTurns(t, server.URL, 100*time.Millisecond, 600*time.Millisecond)

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

// TestTurnsGivesUpOnConnectionsThatNeverConfirm checks the retrying ends after
// three unconfirmed connections in a row.
func TestTurnsGivesUpOnConnectionsThatNeverConfirm(t *testing.T) {
	t.Parallel()
	server := newFakeInk2(t, -1, false)

	svc, _ := runTurns(t, server.URL, 100*time.Millisecond, time.Second)

	if handshakes, _ := server.counts(); handshakes != 3 {
		t.Errorf("handshakes = %d, want 3", handshakes)
	}
	if svc.Usable() {
		t.Error("the service is still usable after three unconfirmed connections")
	}
}

// TestTurnsRejectedSettingsAreNotRetried checks that settings the server
// rejects end the service at once, as the permanent invalid-request error they
// are, carrying the server's message.
func TestTurnsRejectedSettingsAreNotRetried(t *testing.T) {
	t.Parallel()
	server := newFakeInk2(t, 0, true)

	svc, errors := runTurns(t, server.URL, time.Second, 500*time.Millisecond)

	if handshakes, _ := server.counts(); handshakes != 1 {
		t.Errorf("handshakes = %d, want 1", handshakes)
	}
	if svc.Usable() {
		t.Error("the service is still usable after its settings were rejected")
	}
	if len(errors) != 1 || errors[0].Category != errclass.InvalidRequest ||
		!strings.Contains(errors[0].Error, "model_not_found") {
		t.Errorf("errors = %v, want one invalid-request error naming model_not_found", errors)
	}
}
