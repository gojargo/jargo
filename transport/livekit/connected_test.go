package livekit

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

// TestOnParticipantConnectedDeliversWhoArrivedFirst checks a participant who
// joined before the input transport registered its handler is held rather than
// dropped. The room is joined before the pipeline runs, so someone arriving in
// that window would otherwise never be reported.
func TestOnParticipantConnectedDeliversWhoArrivedFirst(t *testing.T) {
	c := &Connection{}

	c.deliverParticipant("first")
	c.deliverParticipant("second")

	var got []string
	c.OnParticipantConnected(func(identity string) { got = append(got, identity) })
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Fatalf("participants held before a handler = %v, want both in order", got)
	}

	c.deliverParticipant("third")
	if len(got) != 3 || got[2] != "third" {
		t.Errorf("participants = %v, want the third delivered straight through", got)
	}
}

// connectedRecorder records the connection frames reaching it.
type connectedRecorder struct {
	*processor.Base
	mu  sync.Mutex
	bot int
	cli int
}

func newConnectedRecorder() *connectedRecorder {
	r := &connectedRecorder{}
	r.Base = processor.New("Recorder", r)
	return r
}

func (r *connectedRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	switch f.(type) {
	case *frames.BotConnectedFrame:
		r.bot++
	case *frames.ClientConnectedFrame:
		r.cli++
	}
	r.mu.Unlock()
	return r.PushFrame(ctx, f, dir)
}

func (r *connectedRecorder) counts() (bot, cli int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.bot, r.cli
}

// waitFor polls until pred holds, failing if it never does.
func (r *connectedRecorder) waitFor(t *testing.T, what string, pred func(bot, cli int) bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if pred(r.counts()) {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	bot, cli := r.counts()
	t.Fatalf("timed out waiting for %s; bot = %d, client = %d", what, bot, cli)
}

// TestConnectionsAreReportedIntoThePipeline covers both halves of what this
// transport can say about a room: the bot is in it, and someone else has
// arrived. Neither is visible from the media, so without these frames a
// measurement of how long the call took to become answerable has nothing to
// measure.
func TestConnectionsAreReportedIntoThePipeline(t *testing.T) {
	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())
	rec := newConnectedRecorder()

	w := pipeline.NewWorker(pipeline.New(tr.Input(), rec), pipeline.WorkerConfig{IdleTimeout: -1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	// The bot joined the room when the connection was made, so the pipeline
	// starting is where that becomes reportable.
	rec.waitFor(t, "the bot's own connection", func(bot, _ int) bool { return bot == 1 })
	if _, cli := rec.counts(); cli != 0 {
		t.Errorf("client connections = %d, want none: nobody has joined yet", cli)
	}

	conn.deliverParticipant("caller")
	rec.waitFor(t, "the caller joining", func(_, cli int) bool { return cli == 1 })

	// A second arrival is a second client, and is reported as one.
	conn.deliverParticipant("observer")
	rec.waitFor(t, "the second arrival", func(_, cli int) bool { return cli == 2 })

	if bot, _ := rec.counts(); bot != 1 {
		t.Errorf("bot connections = %d, want exactly one: the bot joins the room once", bot)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the worker never finished")
	}
}

// TestTheRoomConnectionsAreTimedByAnObserver is the end-to-end reason both
// frames exist. On a room transport the setup has two halves worth telling
// apart: the bot getting into the room, and someone else turning up. An
// observer can only report them because the transport does.
func TestTheRoomConnectionsAreTimedByAnObserver(t *testing.T) {
	reports := make(chan observers.TransportTimingReport, 4)
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: func(r observers.TransportTimingReport) { reports <- r },
	})

	conn := &Connection{}
	tr := NewTransport(conn, transport.DefaultParams())
	rec := newConnectedRecorder()
	w := pipeline.NewWorker(pipeline.New(tr.Input(), rec), pipeline.WorkerConfig{
		IdleTimeout: -1,
		Observers:   []pipeline.Observer{o},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	rec.waitFor(t, "the bot's own connection", func(bot, _ int) bool { return bot == 1 })
	conn.deliverParticipant("caller")

	select {
	case r := <-reports:
		if r.BotConnected == nil {
			t.Fatal("BotConnected is unset, want the time the bot took to join the room")
		}
		if r.ClientConnected < *r.BotConnected {
			t.Errorf("the caller arrived at %s, before the bot joined at %s",
				r.ClientConnected, *r.BotConnected)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the transport timing was never reported")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the worker never finished")
	}
}
