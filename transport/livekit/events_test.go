package livekit

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// TestOnParticipantDisconnectedDeliversWhoLeftFirst checks a departure before
// the handler is registered is held, like an arrival.
func TestOnParticipantDisconnectedDeliversWhoLeftFirst(t *testing.T) {
	c := &Connection{}
	c.deliverParticipantLeft("first")
	var got []string
	c.OnParticipantDisconnected(func(identity string) { got = append(got, identity) })
	c.deliverParticipantLeft("second")
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Fatalf("departures = %v, want both in order", got)
	}
}

// TestClientEventsFollowParticipants covers the events a bot hangs its
// conversation on: a participant joining raises the participant and client
// events, and leaving raises their counterparts and the left event.
func TestClientEventsFollowParticipants(t *testing.T) {
	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())

	var mu sync.Mutex
	var got []string
	record := func(name string) {
		tr.Events().Add(name, func(_ context.Context, _ any, args ...any) {
			mu.Lock()
			defer mu.Unlock()
			switch v := args[0].(type) {
			case Client:
				got = append(got, name+":"+v.ID)
			case string:
				got = append(got, name+":"+v)
			}
		})
	}
	for _, name := range []string{
		EventParticipantConnected, EventParticipantDisconnected, EventParticipantLeft,
		transport.EventClientConnected, transport.EventClientDisconnected,
	} {
		record(name)
	}

	w := pipeline.NewWorker(pipeline.New(tr.Input()), pipeline.WorkerConfig{IdleTimeout: -1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	conn.deliverParticipant("caller")
	conn.deliverParticipantLeft("caller")

	want := map[string]bool{
		"on_participant_connected:caller": true, "on_client_connected:caller": true,
		"on_participant_disconnected:caller": true, "on_participant_left:caller": true,
		"on_client_disconnected:caller": true,
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		mu.Lock()
		n := 0
		for _, g := range got {
			if want[g] {
				n++
			}
		}
		snapshot := append([]string(nil), got...)
		mu.Unlock()
		if n == len(want) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("events = %v, want %d of them", snapshot, len(want))
		}
		time.Sleep(5 * time.Millisecond)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the worker never finished")
	}
}

// TestFirstParticipantAndDataEvents covers the first participant joining, which
// fires once until the room empties, and data packets: every one is raised as
// data received, and a JSON object also as an app message.
func TestFirstParticipantAndDataEvents(t *testing.T) {
	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())

	firsts := make(chan string, 4)
	data := make(chan string, 4)
	apps := make(chan string, 4)
	events.On(tr.Events(), EventFirstParticipantJoined, func(_ context.Context, id string) { firsts <- id })
	tr.Events().Add(EventDataReceived, func(_ context.Context, _ any, args ...any) {
		raw, _ := args[0].([]byte)
		data <- string(raw)
	})
	tr.Events().Add(transport.EventAppMessage, func(_ context.Context, _ any, args ...any) {
		raw, _ := args[0].(json.RawMessage)
		apps <- string(raw)
	})

	w := pipeline.NewWorker(pipeline.New(tr.Input()), pipeline.WorkerConfig{IdleTimeout: -1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	recv := func(ch chan string, what string) string {
		t.Helper()
		select {
		case v := <-ch:
			return v
		case <-time.After(3 * time.Second):
			t.Fatalf("no %s", what)
			return ""
		}
	}
	none := func(ch chan string, what string) {
		t.Helper()
		select {
		case v := <-ch:
			t.Fatalf("unexpected %s: %s", what, v)
		case <-time.After(200 * time.Millisecond):
		}
	}

	conn.deliverParticipant("a")
	if got := recv(firsts, "first participant event"); got != "a" {
		t.Fatalf("first participant = %s", got)
	}
	conn.deliverParticipant("b")
	none(firsts, "second first-participant event")
	conn.deliverParticipantLeft("a")
	conn.deliverParticipantLeft("b")
	conn.deliverParticipant("c")
	if got := recv(firsts, "first participant event after the room emptied"); got != "c" {
		t.Fatalf("first participant = %s", got)
	}

	conn.deliver([]byte(`{"message":"hi"}`), "c")
	if got := recv(apps, "app message"); got != `{"message":"hi"}` {
		t.Fatalf("app message = %s", got)
	}
	recv(data, "data received for the JSON object")
	conn.deliver([]byte("not json"), "c")
	if got := recv(data, "data received for plain bytes"); got != "not json" {
		t.Fatalf("data = %s", got)
	}
	none(apps, "app message for plain bytes")

	cancel()
	<-done
}
