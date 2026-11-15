package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/utils/events"
)

// fluxTurnServer serves a Flux session that, once it has audio, reports one turn
// through every TurnInfo event, and then closes.
func fluxTurnServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		write := func(m map[string]any) {
			data, _ := json.Marshal(m)
			_ = c.Write(ctx, websocket.MessageText, data)
		}
		write(map[string]any{"type": fluxMsgConnected})
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		turn := func(event, transcript string) {
			write(map[string]any{"type": fluxMsgTurnInfo, "event": event, "transcript": transcript})
		}
		turn(fluxEventStartOfTurn, "hi")
		turn(fluxEventUpdate, "") // no text: no result and no event
		turn(fluxEventUpdate, "hel")
		turn(fluxEventEagerEndOfTurn, "") // no text: no result, but the event fires
		turn(fluxEventTurnResumed, "")
		turn(fluxEventEndOfTurn, "hello")
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestFluxTurnEventsFireAfterTheirResults covers each turn event firing with the
// transcript its message carried, and only once the read loop has had the
// results that message produced: the results are pushed before the read loop
// calls Recv again, which is when the event fires.
func TestFluxTurnEventsFireAfterTheirResults(t *testing.T) {
	srv := fluxTurnServer(t)
	conn := newFluxConnector(FluxConfig{APIKey: "k", ListenURL: wsURL(srv.URL), Model: defaultFluxModel})
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()
	fs, ok := stream.(*fluxStream)
	if !ok {
		t.Fatalf("stream is %T", stream)
	}
	var fired []fluxTurnEvent
	fs.raise = func(_ context.Context, ev fluxTurnEvent) { fired = append(fired, ev) }

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	steps := []struct {
		name  string
		fired []fluxTurnEvent // everything fired by the time Recv returns
	}{
		{"start of turn", nil},
		{"update", []fluxTurnEvent{{EventFluxStartOfTurn, "hi"}}},
		{"turn resumed", []fluxTurnEvent{
			{EventFluxStartOfTurn, "hi"}, {EventFluxUpdate, "hel"}, {EventFluxEagerEndOfTurn, ""},
		}},
		{"end of turn", []fluxTurnEvent{
			{EventFluxStartOfTurn, "hi"},
			{EventFluxUpdate, "hel"},
			{EventFluxEagerEndOfTurn, ""},
			{EventFluxTurnResumed, ""},
		}},
	}
	for _, step := range steps {
		res, err := stream.Recv()
		if err != nil {
			t.Fatalf("%s: Recv: %v", step.name, err)
		}
		if len(res) == 0 {
			t.Fatalf("%s: Recv returned no results", step.name)
		}
		if !slices.Equal(fired, step.fired) {
			t.Fatalf("%s: fired %v, want %v", step.name, fired, step.fired)
		}
	}

	// The session ends after EndOfTurn; the next Recv fires its event first.
	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv after the session closed: want an error")
	}
	if got, want := fired[len(fired)-1], (fluxTurnEvent{EventFluxEndOfTurn, "hello"}); got != want {
		t.Fatalf("last event = %v, want %v", got, want)
	}
}

// TestFluxTurnEventsReachTheService covers NewFluxSTT declaring the turn events,
// so a handler attached to the service is given each one: the transcript to a
// typed handler, and nothing to a signal handler for TurnResumed.
func TestFluxTurnEventsReachTheService(t *testing.T) {
	svc := NewFluxSTT(FluxConfig{APIKey: "k"})
	// A connector driving the service NewFluxSTT built, as the one inside it
	// does: the events it raises are the ones the service declared.
	c := newFluxConnector(FluxConfig{APIKey: "k"})
	c.svc = svc

	var (
		mu  sync.Mutex
		got []string
	)
	record := func(s string) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, s)
	}
	for _, name := range []string{EventFluxStartOfTurn, EventFluxEndOfTurn, EventFluxEagerEndOfTurn, EventFluxUpdate} {
		if id := events.On(svc.Events(), name, func(_ context.Context, transcript string) {
			record(name + ":" + transcript)
		}); id == 0 {
			t.Fatalf("%s is not declared", name)
		}
	}
	if id := events.OnSignal(svc.Events(), EventFluxTurnResumed, func(context.Context) {
		record(EventFluxTurnResumed)
	}); id == 0 {
		t.Fatalf("%s is not declared", EventFluxTurnResumed)
	}

	ctx := context.Background()
	for _, ev := range []fluxTurnEvent{
		{EventFluxStartOfTurn, "hi"},
		{EventFluxUpdate, "hel"},
		{EventFluxEagerEndOfTurn, "hello"},
		{EventFluxTurnResumed, ""},
		{EventFluxEndOfTurn, "hello there"},
	} {
		c.raise(ctx, ev)
	}

	want := []string{
		"on_start_of_turn:hi", "on_update:hel", "on_eager_end_of_turn:hello",
		"on_turn_resumed", "on_end_of_turn:hello there",
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		mu.Lock()
		done := slices.Equal(got, want)
		snapshot := slices.Clone(got)
		mu.Unlock()
		if done {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("handlers saw %v, want %v", snapshot, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
