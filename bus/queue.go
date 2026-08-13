package bus

import "sync"

// messageQueue is an unbounded, concurrency-safe message queue with a single
// consumer.
//
// System messages take priority: get returns any queued system message before
// any data message, which is what lets a cancel reach a worker ahead of the
// work it is calling off. Within each band messages come back in the order they
// went in. Producers never block, so a subscriber that is slow to read cannot
// hold up whoever is sending.
type messageQueue struct {
	mu     sync.Mutex
	system []Message
	data   []Message
	notify chan struct{}
}

func newMessageQueue() *messageQueue {
	return &messageQueue{notify: make(chan struct{}, 1)}
}

// put appends a message to the band it belongs to. It never blocks.
func (q *messageQueue) put(m Message) {
	q.mu.Lock()
	if _, ok := m.(SystemMessage); ok {
		q.system = append(q.system, m)
	} else {
		q.data = append(q.data, m)
	}
	q.mu.Unlock()

	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// tryGet takes the next message, reporting whether there was one.
func (q *messageQueue) tryGet() (Message, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.system) > 0 {
		m := q.system[0]
		q.system = q.system[1:]
		return m, true
	}
	if len(q.data) > 0 {
		m := q.data[0]
		q.data = q.data[1:]
		return m, true
	}
	return nil, false
}

// get blocks until a message is available or done is closed, reporting false
// when it stopped waiting.
func (q *messageQueue) get(done <-chan struct{}) (Message, bool) {
	for {
		if m, ok := q.tryGet(); ok {
			return m, true
		}
		select {
		case <-done:
			return nil, false
		case <-q.notify:
		}
	}
}
