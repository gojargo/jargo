package tts_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/service/tts"
)

const keepaliveInterval = 10 * time.Millisecond

var errKeepaliveClosed = errors.New("connection closed")

// keepaliveRecorder is a websocket that records the keepalives sent on it.
type keepaliveRecorder struct {
	mu           sync.Mutex
	open         bool
	keepalives   int
	failuresLeft int
}

func (r *keepaliveRecorder) KeepaliveReady() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.open
}

func (r *keepaliveRecorder) SendKeepalive(context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.failuresLeft > 0 {
		r.failuresLeft--
		return errKeepaliveClosed
	}
	r.keepalives++
	return nil
}

func (r *keepaliveRecorder) setOpen(open bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.open = open
}

func (r *keepaliveRecorder) counts() (keepalives, failuresLeft int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.keepalives, r.failuresLeft
}

func waitForKeepalives(t *testing.T, r *keepaliveRecorder, count int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		if got, _ := r.counts(); got >= count {
			return
		}
		if time.Now().After(deadline) {
			got, _ := r.counts()
			t.Fatalf("%d keepalives sent, want %d", got, count)
		}
		time.Sleep(keepaliveInterval)
	}
}

// Keepalives are sent every interval while the websocket is open.
func TestKeepaliveSentPeriodically(t *testing.T) {
	r := &keepaliveRecorder{open: true}
	var k tts.Keepalive
	k.Start(t.Context(), keepaliveInterval, r)
	waitForKeepalives(t, r, 3)
	k.Stop()
	if k.Running() {
		t.Error("the keepalive is still running after Stop")
	}
}

// A failed send is logged and later keepalives still go out.
func TestKeepaliveSurvivesFailedSend(t *testing.T) {
	r := &keepaliveRecorder{open: true, failuresLeft: 2}
	var k tts.Keepalive
	k.Start(t.Context(), keepaliveInterval, r)
	defer k.Stop()
	waitForKeepalives(t, r, 2)
	if _, left := r.counts(); left != 0 {
		t.Errorf("%d failures left, want both used up", left)
	}
	if !k.Running() {
		t.Error("the keepalive stopped after a failed send")
	}
}

// No keepalive is sent while the websocket is closed, and sending resumes once
// it reopens.
func TestKeepaliveSkippedWhileWebsocketNotOpen(t *testing.T) {
	r := &keepaliveRecorder{}
	var k tts.Keepalive
	k.Start(t.Context(), keepaliveInterval, r)
	defer k.Stop()
	time.Sleep(5 * keepaliveInterval)
	if got, _ := r.counts(); got != 0 {
		t.Fatalf("%d keepalives sent on a closed websocket, want none", got)
	}
	r.setOpen(true)
	waitForKeepalives(t, r, 1)
}

// Starting the loop while one is running leaves the running one in place.
func TestKeepaliveStartKeepsRunningLoop(t *testing.T) {
	first := &keepaliveRecorder{open: true}
	second := &keepaliveRecorder{open: true}
	var k tts.Keepalive
	k.Start(t.Context(), keepaliveInterval, first)
	defer k.Stop()
	k.Start(t.Context(), keepaliveInterval, second)
	waitForKeepalives(t, first, 2)
	if got, _ := second.counts(); got != 0 {
		t.Errorf("the second Start replaced the running loop: %d keepalives went to it", got)
	}
}

// A loop that has ended is replaced by a new one.
func TestKeepaliveStartReplacesFinishedLoop(t *testing.T) {
	r := &keepaliveRecorder{open: true}
	var k tts.Keepalive
	ended, end := context.WithCancel(t.Context())
	k.Start(ended, keepaliveInterval, r)
	end()
	deadline := time.Now().Add(time.Second)
	for k.Running() {
		if time.Now().After(deadline) {
			t.Fatal("the loop did not end with its context")
		}
		time.Sleep(time.Millisecond)
	}
	k.Start(t.Context(), keepaliveInterval, r)
	defer k.Stop()
	if !k.Running() {
		t.Fatal("Start did not replace the ended loop")
	}
	waitForKeepalives(t, r, 1)
}
