// Package bus is the pub/sub messaging that connects workers to each other and
// to the runner that manages them.
//
// Each subscriber receives messages independently, through a queue of its own,
// so one that is slow to handle a message never holds up another. System
// messages are delivered ahead of the data messages already queued, which is
// what lets a cancel reach a worker before the work it is calling off.
//
// A Bus is the transport-independent core. AsyncQueueBus delivers in process; a
// networked one embeds Bus, overrides Publish for its transport, and calls
// OnMessageReceived as messages arrive.
package bus

import (
	"context"
	"fmt"
	"log/slog"
	"runtime/debug"
	"sync"
)

// Subscriber receives messages from a bus.
type Subscriber interface {
	// Name identifies this subscriber on the bus. Subscribing twice under the
	// same name is a no-op.
	Name() string
	// OnBusMessage handles one message. A panic in it is contained: the
	// subscriber keeps receiving.
	OnBusMessage(ctx context.Context, m Message)
}

// Publisher is what a concrete bus implements to carry a message on its
// transport. Bus.Send calls it for everything that is not local-only.
type Publisher interface {
	Publish(ctx context.Context, m Message)
}

// subscription is one subscriber's state on the bus: the queue everything
// arrives in, the queue the router hands data messages to, and the goroutines
// draining them.
type subscription struct {
	subscriber Subscriber
	// queue is where every message for this subscriber arrives.
	queue *messageQueue
	// dataQueue holds the data messages the router has separated out, so a
	// subscriber slow to handle one does not delay the system messages behind
	// it.
	dataQueue *messageQueue

	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Bus is the transport-independent core of a worker bus. Embed it in a concrete
// bus and implement Publisher:
//
//	type MyBus struct{ *bus.Bus }
//
//	func (b *MyBus) Publish(ctx context.Context, m bus.Message) { … }
//
// The zero value is not usable; build one with New.
type Bus struct {
	// self is the concrete bus, so Send reaches the overridden Publish.
	self Publisher

	mu            sync.Mutex
	subscriptions map[string]*subscription
	running       bool
	// runCtx is the lifetime the dispatch goroutines run under, live between
	// Start and Stop.
	runCtx    context.Context
	runCancel context.CancelFunc
}

// New builds a Bus. self is the embedding bus, whose Publish carries a message
// on its transport; pass nil for a bus that only delivers locally.
func New(self Publisher) *Bus {
	b := &Bus{subscriptions: make(map[string]*subscription)}
	if self != nil {
		b.self = self
	} else {
		b.self = noPublisher{}
	}
	return b
}

// noPublisher drops what it is given, for a bus with no transport of its own.
type noPublisher struct{}

func (noPublisher) Publish(context.Context, Message) {}

// Start begins dispatching to every subscriber registered so far. Messages sent
// before it are queued, not lost, and are delivered once it runs.
func (b *Bus) Start(ctx context.Context) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.running {
		return
	}
	b.running = true
	b.runCtx, b.runCancel = context.WithCancel(ctx)
	for _, sub := range b.subscriptions {
		b.startDispatchLocked(sub)
	}
}

// Stop ends dispatch and waits for the goroutines to return. Messages sent
// afterwards are queued but not delivered until Start runs again.
func (b *Bus) Stop() {
	b.mu.Lock()
	if !b.running {
		b.mu.Unlock()
		return
	}
	b.running = false
	if b.runCancel != nil {
		b.runCancel()
	}
	subs := make([]*subscription, 0, len(b.subscriptions))
	for _, sub := range b.subscriptions {
		subs = append(subs, sub)
	}
	b.mu.Unlock()

	for _, sub := range subs {
		sub.stop()
	}
}

// Subscribe registers a subscriber. It is idempotent: registering one already
// registered, by name, changes nothing.
func (b *Bus) Subscribe(s Subscriber) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.subscriptions[s.Name()]; ok {
		return
	}
	sub := &subscription{
		subscriber: s,
		queue:      newMessageQueue(),
		dataQueue:  newMessageQueue(),
	}
	b.subscriptions[s.Name()] = sub
	if b.running {
		b.startDispatchLocked(sub)
	}
}

// Unsubscribe removes a subscriber and stops delivering to it.
func (b *Bus) Unsubscribe(s Subscriber) {
	b.mu.Lock()
	sub, ok := b.subscriptions[s.Name()]
	delete(b.subscriptions, s.Name())
	b.mu.Unlock()

	if ok {
		sub.stop()
	}
}

// Send puts a message on the bus. A local-only message is delivered straight to
// the subscribers here; everything else goes to the transport, which delivers
// it back through OnMessageReceived.
func (b *Bus) Send(ctx context.Context, m Message) {
	if _, ok := m.(LocalMessage); ok {
		b.OnMessageReceived(m)
		return
	}
	b.self.Publish(ctx, m)
}

// OnMessageReceived hands a message to every local subscriber. A concrete bus
// calls it when a message arrives, whether from a local Send or off the
// network.
func (b *Bus) OnMessageReceived(m Message) {
	b.mu.Lock()
	subs := make([]*subscription, 0, len(b.subscriptions))
	for _, sub := range b.subscriptions {
		subs = append(subs, sub)
	}
	b.mu.Unlock()

	for _, sub := range subs {
		sub.queue.put(m)
	}
}

// startDispatchLocked starts a subscription's two goroutines. The caller holds
// b.mu.
func (b *Bus) startDispatchLocked(sub *subscription) {
	ctx, cancel := context.WithCancel(b.runCtx)
	sub.cancel = cancel
	sub.wg.Add(2)
	go func() {
		defer sub.wg.Done()
		b.route(ctx, sub)
	}()
	go func() {
		defer sub.wg.Done()
		b.dispatchData(ctx, sub)
	}()
}

// route handles system messages as they arrive and hands everything else to the
// data queue, so a system message is never stuck behind a subscriber still
// working through the data ahead of it.
func (b *Bus) route(ctx context.Context, sub *subscription) {
	for {
		m, ok := sub.queue.get(ctx.Done())
		if !ok {
			return
		}
		if _, isSystem := m.(SystemMessage); isSystem {
			deliver(ctx, sub.subscriber, m)
			continue
		}
		sub.dataQueue.put(m)
	}
}

// dispatchData hands the subscriber its data messages one at a time, in order.
func (b *Bus) dispatchData(ctx context.Context, sub *subscription) {
	for {
		m, ok := sub.dataQueue.get(ctx.Done())
		if !ok {
			return
		}
		deliver(ctx, sub.subscriber, m)
	}
}

// deliver hands one message to a subscriber, containing a panic.
//
// A subscriber that fails must not bring down the goroutine delivering to it:
// it would then silently stop receiving everything afterwards, including the
// cancel that would have shut it down.
func deliver(ctx context.Context, s Subscriber, m Message) {
	defer func() {
		if v := recover(); v != nil {
			slog.Error("subscriber panicked handling a bus message; keeping delivery alive",
				"subscriber", s.Name(),
				"message", fmt.Sprintf("%T", m),
				"panic", v,
				"stack", string(debug.Stack()))
		}
	}()
	s.OnBusMessage(ctx, m)
}

// stop ends a subscription's goroutines and waits for them.
func (s *subscription) stop() {
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	s.wg.Wait()
}
