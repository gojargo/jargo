package redis_test

// Ported from upstream's Redis bus suite. Upstream runs against an in-memory
// fake of the Redis client; these run against miniredis, an in-process Redis
// server, so the real client and protocol are exercised. Upstream sleeps for a
// fixed spell after sending; these wait for the expected messages instead.

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/gojargo/jargo/bus"
	redisbus "github.com/gojargo/jargo/bus/network/redis"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	goredis "github.com/redis/go-redis/v9"
)

// collector records every message it receives.
type collector struct {
	name string
	mu   sync.Mutex
	got  []bus.Message
}

//nolint:gochecknoglobals // names the subscribers apart across the suite
var collectorSeq atomic.Int64

func newCollector() *collector {
	return &collector{name: fmt.Sprintf("test_sub_%d", collectorSeq.Add(1))}
}

func (c *collector) Name() string { return c.name }

func (c *collector) OnBusMessage(_ context.Context, m bus.Message) {
	c.mu.Lock()
	c.got = append(c.got, m)
	c.mu.Unlock()
}

func (c *collector) received() []bus.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]bus.Message(nil), c.got...)
}

// await waits for the collector to hold n messages and returns them.
func (c *collector) await(t *testing.T, n int) []bus.Message {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if got := c.received(); len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("received %d messages, want %d", len(c.received()), n)
	return nil
}

// newClient starts a Redis server for the test and returns a client of it.
func newClient(t *testing.T) (*miniredis.Miniredis, *goredis.Client) {
	t.Helper()
	server := miniredis.RunT(t)
	client := goredis.NewClient(&goredis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return server, client
}

// published subscribes to channel on its own connection and returns what is
// published there.
func published(t *testing.T, client *goredis.Client, channel string) <-chan string {
	t.Helper()
	pubsub := client.Subscribe(t.Context(), channel)
	if _, err := pubsub.Receive(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pubsub.Close() })
	out := make(chan string, 16)
	go func() {
		for m := range pubsub.Channel() {
			out <- m.Payload
		}
	}()
	return out
}

func newBus(t *testing.T, client *goredis.Client, subs ...bus.Subscriber) *redisbus.Bus {
	t.Helper()
	b := redisbus.NewBus(redisbus.Config{Client: client, Channel: "test:bus"})
	for _, s := range subs {
		b.Subscribe(s)
	}
	if err := b.Start(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Stop)
	return b
}

func endMessage(source, reason string) *bus.EndMessage {
	m := &bus.EndMessage{Reason: reason}
	m.From = source
	return m
}

// Send serializes a message and publishes it to the bus's channel.
func TestSendPublishesToRedis(t *testing.T) {
	_, client := newClient(t)
	wire := published(t, client, "test:bus")
	b := redisbus.NewBus(redisbus.Config{Client: client, Channel: "test:bus"})

	m := &bus.JobRequestMessage{JobID: "t-1"}
	m.From, m.To = "task_a", "task_b"
	b.Send(t.Context(), m)

	select {
	case data := <-wire:
		restored, err := bus.NewJSONSerializer(nil).Deserialize([]byte(data))
		if err != nil {
			t.Fatal(err)
		}
		if restored.Source() != "task_a" || restored.Target() != "task_b" {
			t.Errorf("restored %s -> %s, want task_a -> task_b", restored.Source(), restored.Target())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("nothing was published")
	}
}

// A local-only message is delivered to the local subscribers and never
// published.
func TestLocalMessageDeliveredLocallyNotToRedis(t *testing.T) {
	_, client := newClient(t)
	wire := published(t, client, "test:bus")
	sub := newCollector()
	b := newBus(t, client, sub)

	worker := struct{ name string }{"test"}
	m := &bus.AddWorkerMessage{Worker: worker}
	m.From = "parent"
	b.Send(t.Context(), m)

	got := sub.await(t, 1)
	add, ok := got[0].(*bus.AddWorkerMessage)
	if !ok || add.Worker != worker {
		t.Fatalf("received %#v, want the AddWorkerMessage with its worker", got[0])
	}
	select {
	case data := <-wire:
		t.Errorf("the local message was published: %s", data)
	case <-time.After(100 * time.Millisecond):
	}
}

// What a bus publishes comes back to its subscribers off the channel.
func TestRoundTripViaSubscriber(t *testing.T) {
	_, client := newClient(t)
	sub := newCollector()
	b := newBus(t, client, sub)

	b.Send(t.Context(), endMessage("task_a", "done"))

	got := sub.await(t, 1)
	end, ok := got[0].(*bus.EndMessage)
	if !ok || end.Source() != "task_a" || end.Reason != "done" {
		t.Fatalf("received %#v, want the EndMessage from task_a", got[0])
	}
}

// Every subscriber receives every message.
func TestMultipleSubscribersReceive(t *testing.T) {
	_, client := newClient(t)
	a, c := newCollector(), newCollector()
	b := newBus(t, client, a, c)

	b.Send(t.Context(), endMessage("x", ""))

	a.await(t, 1)
	c.await(t, 1)
}

// A frame message, frame and direction, round-trips through Redis.
func TestFrameMessageRoundTrip(t *testing.T) {
	_, client := newClient(t)
	sub := newCollector()
	b := newBus(t, client, sub)

	m := &bus.FrameMessage{Frame: frames.NewTextFrame("hello"), Direction: processor.Downstream}
	m.From = "task_a"
	b.Send(t.Context(), m)

	got := sub.await(t, 1)
	restored, ok := got[0].(*bus.FrameMessage)
	if !ok {
		t.Fatalf("received %T, want a FrameMessage", got[0])
	}
	text, ok := restored.Frame.(*frames.TextFrame)
	if !ok || text.Text != "hello" {
		t.Errorf("frame = %#v, want the text frame", restored.Frame)
	}
	if restored.Direction != processor.Downstream {
		t.Errorf("direction = %v, want downstream", restored.Direction)
	}
}

// A job request, payload included, round-trips through Redis.
func TestJobRequestRoundTrip(t *testing.T) {
	_, client := newClient(t)
	sub := newCollector()
	b := newBus(t, client, sub)

	m := &bus.JobRequestMessage{JobID: "t-1", Payload: map[string]any{"key": "value"}}
	m.From, m.To = "parent", "worker"
	b.Send(t.Context(), m)

	got := sub.await(t, 1)
	restored, ok := got[0].(*bus.JobRequestMessage)
	if !ok {
		t.Fatalf("received %T, want a JobRequestMessage", got[0])
	}
	if restored.JobID != "t-1" || restored.Payload["key"] != "value" {
		t.Errorf("restored %+v, want job t-1 with its payload", restored)
	}
}

// Messages are published to the configured channel, and an unconfigured bus
// uses the default one.
func TestCustomChannel(t *testing.T) {
	_, client := newClient(t)
	wire := published(t, client, "custom:channel")
	b := redisbus.NewBus(redisbus.Config{Client: client, Channel: "custom:channel"})

	b.Send(t.Context(), endMessage("a", ""))

	select {
	case <-wire:
	case <-time.After(3 * time.Second):
		t.Fatal("nothing was published to the configured channel")
	}
	if got := redisbus.NewBus(redisbus.Config{Client: client}).Channel(); got != redisbus.DefaultChannel {
		t.Errorf("channel = %q, want %q", got, redisbus.DefaultChannel)
	}
}

// Stopping ends the reader and unsubscribes from the channel.
func TestStopCleansUp(t *testing.T) {
	server, client := newClient(t)
	b := redisbus.NewBus(redisbus.Config{Client: client, Channel: "test:bus"})
	if err := b.Start(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := server.PubSubNumSub("test:bus")["test:bus"]; got != 1 {
		t.Fatalf("subscribers = %d while running, want 1", got)
	}

	b.Stop()

	deadline := time.Now().Add(3 * time.Second)
	for server.PubSubNumSub("test:bus")["test:bus"] != 0 {
		if time.Now().After(deadline) {
			t.Fatal("the bus is still subscribed after Stop")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// Two buses on one channel, as two processes would have, hear each other.
func TestBusesOnOneChannelHearEachOther(t *testing.T) {
	_, client := newClient(t)
	a, c := newCollector(), newCollector()
	first := newBus(t, client, a)
	newBus(t, client, c)

	first.Send(t.Context(), endMessage("first", ""))

	if got := c.await(t, 1); got[0].Source() != "first" {
		t.Errorf("the other bus received %#v", got[0])
	}
	a.await(t, 1)
}

// A bus that cannot reach Redis fails to start.
func TestStartFailsWithoutRedis(t *testing.T) {
	server, client := newClient(t)
	server.Close()
	b := redisbus.NewBus(redisbus.Config{Client: client})

	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if err := b.Start(ctx); err == nil {
		b.Stop()
		t.Fatal("Start succeeded with Redis down")
	}
}
