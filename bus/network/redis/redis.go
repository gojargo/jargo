// Package redis is a worker bus carried over Redis pub/sub, for workers running
// in different processes.
//
// It is a module of its own, so only a program that imports it depends on a
// Redis client.
package redis

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/bus"
	goredis "github.com/redis/go-redis/v9"
)

// DefaultChannel is the pub/sub channel a Bus uses when none is configured.
const DefaultChannel = "jargo:bus"

// receiveRetryDelay is how long the reader waits after a failed read before
// reading again. The client reconnects and subscribes again on its own; the
// pause keeps a server that is down from being retried in a tight loop.
const receiveRetryDelay = 100 * time.Millisecond

// Config configures a Bus.
type Config struct {
	// Client is the Redis client to publish and subscribe with. Required.
	Client goredis.UniversalClient
	// Serializer encodes and decodes the messages; nil uses a
	// bus.JSONSerializer over the package's registered types.
	Serializer bus.MessageSerializer
	// Channel is the pub/sub channel; empty uses DefaultChannel.
	Channel string
}

// Bus is a worker bus backed by Redis pub/sub.
//
// It publishes each message, serialized, to a Redis channel, and delivers what
// arrives on the channel to its local subscribers, its own messages included. A
// local-only message (bus.LocalMessage) never reaches Redis: it is delivered
// straight to the local subscribers.
//
//	client := goredis.NewClient(&goredis.Options{Addr: "localhost:6379"})
//	b := redis.NewBus(redis.Config{Client: client, Channel: "my-session"})
type Bus struct {
	*bus.Bus

	client     goredis.UniversalClient
	serializer bus.MessageSerializer
	channel    string

	mu           sync.Mutex
	pubsub       *goredis.PubSub
	readerCancel context.CancelFunc
	readerWG     sync.WaitGroup
}

// NewBus builds a Redis bus. It subscribes to its channel when it starts.
func NewBus(cfg Config) *Bus {
	b := &Bus{client: cfg.Client, serializer: cfg.Serializer, channel: cfg.Channel}
	if b.serializer == nil {
		b.serializer = bus.NewJSONSerializer(nil)
	}
	if b.channel == "" {
		b.channel = DefaultChannel
	}
	b.Bus = bus.New(b)
	return b
}

// StartTransport subscribes to the channel and starts reading it, implementing
// bus.Transport. Bus.Start calls it.
func (b *Bus) StartTransport(ctx context.Context) error {
	pubsub := b.client.Subscribe(ctx, b.channel)
	// Subscribing is only sent above; the confirmation says it took effect, and
	// a message published after it is one the reader will see.
	if _, err := pubsub.Receive(ctx); err != nil {
		_ = pubsub.Close()
		return fmt.Errorf("redis bus: subscribe to %q: %w", b.channel, err)
	}
	readerCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	b.mu.Lock()
	b.pubsub = pubsub
	b.readerCancel = cancel
	b.mu.Unlock()
	b.readerWG.Go(func() { b.readLoop(readerCtx, pubsub) })
	return nil
}

// StopTransport stops the reader and unsubscribes from the channel,
// implementing bus.Transport. Bus.Stop calls it.
//
// The subscription is closed rather than unsubscribed: a read waiting on it is
// not interrupted by its context, only by the close, and closing the
// subscription's connection is what ends it on the server too.
func (b *Bus) StopTransport() {
	b.mu.Lock()
	pubsub, cancel := b.pubsub, b.readerCancel
	b.pubsub, b.readerCancel = nil, nil
	b.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if pubsub != nil {
		if err := pubsub.Close(); err != nil {
			slog.Debug("redis bus: close the subscription", "channel", b.channel, "err", err)
		}
	}
	b.readerWG.Wait()
}

// Publish publishes a message to the channel, implementing bus.Publisher.
// Bus.Send calls it for every message that is not local-only. A message that
// cannot be encoded or published is logged and dropped, as Send has no error
// to return it in.
func (b *Bus) Publish(ctx context.Context, m bus.Message) {
	data, err := b.serializer.Serialize(m)
	if err != nil {
		slog.Error("redis bus: failed to serialize message", "channel", b.channel,
			"message", fmt.Sprintf("%T", m), "err", err)
		return
	}
	if err := b.client.Publish(ctx, b.channel, data).Err(); err != nil {
		slog.Error("redis bus: failed to publish message", "channel", b.channel,
			"message", fmt.Sprintf("%T", m), "err", err)
	}
}

// Channel is the pub/sub channel the bus publishes to and reads.
func (b *Bus) Channel() string { return b.channel }

// readLoop reads the channel and delivers each message to the local
// subscribers, until ctx is done. A message that does not decode is logged and
// skipped.
func (b *Bus) readLoop(ctx context.Context, pubsub *goredis.PubSub) {
	for {
		raw, err := pubsub.ReceiveMessage(ctx)
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, goredis.ErrClosed) {
				return
			}
			slog.Warn("redis bus: failed to read from the channel", "channel", b.channel, "err", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(receiveRetryDelay):
			}
			continue
		}
		m, err := b.serializer.Deserialize([]byte(raw.Payload))
		if err != nil {
			slog.Error("redis bus: failed to deserialize message", "channel", b.channel, "err", err)
			continue
		}
		if m != nil {
			b.OnMessageReceived(m)
		}
	}
}
