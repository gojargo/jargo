package bus

import "context"

// AsyncQueueBus delivers messages in process, straight to the local
// subscribers. It is the bus a runner and its workers use when they all live in
// one process, and the only one the rest of the framework needs.
type AsyncQueueBus struct {
	*Bus
}

// NewAsyncQueueBus builds an in-process bus.
func NewAsyncQueueBus() *AsyncQueueBus {
	b := &AsyncQueueBus{}
	b.Bus = New(b)
	return b
}

// Publish delivers a message to the local subscribers. There is no transport to
// carry it to, so publishing and receiving are the same step.
func (b *AsyncQueueBus) Publish(_ context.Context, m Message) {
	b.OnMessageReceived(m)
}
