package livekit

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
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

// runInput runs a pipeline of just the transport's input, recording what it
// pushes, until the test ends.
func runInput(t *testing.T, tr *Transport) *frameLog {
	t.Helper()
	log := &frameLog{}
	w := pipeline.NewWorker(pipeline.New(tr.Input()), pipeline.WorkerConfig{
		IdleTimeout: -1, ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&w.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		log.add(f)
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("the worker never finished")
		}
	})
	return log
}

// frameLog records frames.
type frameLog struct {
	mu sync.Mutex
	fs []frames.Frame
}

func (l *frameLog) add(f frames.Frame) { l.mu.Lock(); l.fs = append(l.fs, f); l.mu.Unlock() }

// dtmf is the keys of the InputDTMFFrames recorded so far.
func (l *frameLog) dtmf() []frames.KeypadEntry {
	l.mu.Lock()
	defer l.mu.Unlock()
	var keys []frames.KeypadEntry
	for _, f := range l.fs {
		if d, ok := f.(*frames.InputDTMFFrame); ok {
			keys = append(keys, d.Button)
		}
	}
	return keys
}

// awaitEvent waits for one value on ch.
func awaitEvent[T any](t *testing.T, ch chan T, what string) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(3 * time.Second):
		t.Fatalf("no %s", what)
		var zero T
		return zero
	}
}

// Every key a SIP caller presses is raised with what LiveKit reported, and a
// keypad key also enters the pipeline as an InputDTMFFrame; one that is not is
// raised but goes no further.
func TestDTMFIsRaisedAndPushed(t *testing.T) {
	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())
	raised := make(chan DTMFEvent, 4)
	events.On(tr.Events(), EventDTMF, func(_ context.Context, e DTMFEvent) { raised <- e })
	log := runInput(t, tr)

	conn.deliverDTMF(DTMFEvent{Tone: "5", Digit: "5", Code: 5, ParticipantID: "sip"})
	if e := awaitEvent(t, raised, "DTMF event"); e.Tone != "5" || e.ParticipantID != "sip" {
		t.Errorf("event = %+v, want the key and who pressed it", e)
	}
	conn.deliverDTMF(DTMFEvent{Tone: "x"})
	if e := awaitEvent(t, raised, "DTMF event for an unknown key"); e.Tone != "x" {
		t.Errorf("event = %+v, want the unknown key raised too", e)
	}
	conn.deliverDTMF(DTMFEvent{Tone: "#", Code: 11})
	awaitEvent(t, raised, "DTMF event")

	deadline := time.Now().Add(3 * time.Second)
	for len(log.dtmf()) < 2 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if got := log.dtmf(); len(got) != 2 || got[0] != frames.KeypadFive || got[1] != frames.KeypadPound {
		t.Errorf("pushed %v, want 5 then #, and nothing for the unknown key", got)
	}
}

// The room's lifecycle is raised: connected once the transport runs, the
// audio tracks as they are subscribed and unsubscribed, the loudest speaker
// as it changes, and the disconnect, once, after the handlers that run before
// it.
func TestRoomEventsAreRaised(t *testing.T) {
	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())
	got := make(chan string, 16)
	tr.Events().Add(EventConnected, func(context.Context, any, ...any) { got <- "connected" })
	events.On(tr.Events(), EventAudioTrackUnsubscribed, func(_ context.Context, id string) {
		got <- "unsubscribed:" + id
	})
	events.On(tr.Events(), EventActiveSpeakerChanged, func(_ context.Context, c Client) {
		got <- "speaker:" + c.ID
	})
	tr.Events().Add(EventBeforeDisconnect, func(context.Context, any, ...any) { got <- "before" })
	tr.Events().Add(EventDisconnected, func(context.Context, any, ...any) { got <- "disconnected" })
	runInput(t, tr)

	if e := awaitEvent(t, got, "connected"); e != "connected" {
		t.Fatalf("first event = %q, want connected", e)
	}
	conn.deliverAudioUnsubscribed("caller")
	if e := awaitEvent(t, got, "unsubscribed"); e != "unsubscribed:caller" {
		t.Errorf("event = %q, want the track unsubscribed", e)
	}
	conn.activeSpeakersChanged([]string{"caller", "other"})
	if e := awaitEvent(t, got, "speaker"); e != "speaker:caller" {
		t.Errorf("event = %q, want the loudest speaker", e)
	}

	// The server closing the room is reported once, however often it says so.
	conn.reportDisconnected()
	conn.reportDisconnected()
	if e := awaitEvent(t, got, "disconnected"); e != "disconnected" {
		t.Errorf("event = %q, want disconnected", e)
	}
	select {
	case e := <-got:
		t.Errorf("unexpected event %q after the disconnect", e)
	case <-time.After(100 * time.Millisecond):
	}
}
