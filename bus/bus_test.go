package bus_test

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
)

// Ported from upstream's bus suite. Upstream sleeps for a fixed spell after
// sending and then asserts; these wait for the expected count instead, which
// says the same thing without pinning the test to a delivery latency.

// collector records every message it receives.
type collector struct {
	name string
	mu   sync.Mutex
	got  []bus.Message
}

//nolint:gochecknoglobals // names the subscribers apart across the suite
var collectorSeq atomic.Int64

func newCollector() *collector {
	return &collector{name: fmt.Sprintf("collector_%d", collectorSeq.Add(1))}
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

// awaitCount waits for the collector to hold n messages, reporting whether it
// got there.
func (c *collector) awaitCount(n int) bool {
	for range 200 {
		if len(c.received()) >= n {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return len(c.received()) >= n
}

// settle gives delivery a moment to happen, for the cases that assert nothing
// arrives.
func settle() { time.Sleep(50 * time.Millisecond) }

func data(source string) *bus.EndMessage {
	m := &bus.EndMessage{}
	m.From = source
	return m
}

func cancel(source string) *bus.CancelMessage {
	m := &bus.CancelMessage{}
	m.From = source
	return m
}

func sourcesOf(msgs []bus.Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Source())
	}
	return out
}

func TestSendDeliversToSubscriber(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	m := data("task_a")
	b.Send(t.Context(), m)

	if !sub.awaitCount(1) {
		t.Fatalf("the subscriber received %d messages, want 1", len(sub.received()))
	}
	if got := sub.received()[0]; got != bus.Message(m) {
		t.Errorf("the subscriber received %v, want the message that was sent", got)
	}
}

func TestMultipleMessagesInOrder(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	var sent []string
	for i := range 5 {
		source := fmt.Sprintf("task_%d", i)
		sent = append(sent, source)
		b.Send(t.Context(), data(source))
	}

	if !sub.awaitCount(5) {
		t.Fatalf("the subscriber received %d messages, want 5", len(sub.received()))
	}
	if got := sourcesOf(sub.received()); !equal(got, sent) {
		t.Errorf("messages arrived as %v, want them in send order %v", got, sent)
	}
}

// Dispatch runs between Start and Stop; a message sent after Stop waits.
func TestStartStopLifecycle(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Start(t.Context())

	b.Send(t.Context(), data("a"))
	if !sub.awaitCount(1) {
		t.Fatalf("the subscriber received %d messages while running, want 1", len(sub.received()))
	}

	b.Stop()
	b.Send(t.Context(), data("b"))
	settle()

	if got := len(sub.received()); got != 1 {
		t.Errorf("the subscriber received %d messages after Stop, want the 1 from before", got)
	}
}

func TestSubscribeCallsOnBusMessage(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), data("runner"))

	if !sub.awaitCount(1) {
		t.Error("subscribing did not lead to the subscriber being called")
	}
}

func TestMultipleSubscribersAreIndependent(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	a, c := newCollector(), newCollector()
	b.Subscribe(a)
	b.Subscribe(c)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), data("runner"))

	if !a.awaitCount(1) || !c.awaitCount(1) {
		t.Errorf("subscribers received %d and %d messages, want 1 each",
			len(a.received()), len(c.received()))
	}
}

func TestUnsubscribeStopsDelivery(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), data("first"))
	if !sub.awaitCount(1) {
		t.Fatal("the first message never arrived")
	}

	b.Unsubscribe(sub)
	b.Send(t.Context(), data("second"))
	settle()

	if got := len(sub.received()); got != 1 {
		t.Errorf("the subscriber received %d messages after unsubscribing, want 1", got)
	}
}

// slowSub takes its time over every message.
type slowSub struct{ delay time.Duration }

func (slowSub) Name() string { return "slow" }

func (s slowSub) OnBusMessage(context.Context, bus.Message) { time.Sleep(s.delay) }

// Each subscriber has a queue of its own, so one that is slow to handle a
// message never holds up another.
func TestSlowSubscriberDoesNotBlockOthers(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	fast := newCollector()
	b.Subscribe(slowSub{delay: 500 * time.Millisecond})
	b.Subscribe(fast)
	b.Start(t.Context())
	defer b.Stop()

	start := time.Now()
	b.Send(t.Context(), data("a"))

	if !fast.awaitCount(1) {
		t.Fatal("the fast subscriber never received the message")
	}
	if elapsed := time.Since(start); elapsed > 400*time.Millisecond {
		t.Errorf("the fast subscriber waited %v, want it not to wait on the slow one", elapsed)
	}
}

// A system message is delivered ahead of the data messages already queued.
func TestSystemMessagePreemptsDataMessages(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)

	// Queued before dispatch starts, so the whole batch is waiting when it does.
	for i := range 5 {
		b.Send(t.Context(), data(fmt.Sprintf("data_%d", i)))
	}
	b.Send(t.Context(), cancel("runner"))

	b.Start(t.Context())
	defer b.Stop()

	if !sub.awaitCount(6) {
		t.Fatalf("the subscriber received %d messages, want 6", len(sub.received()))
	}
	if _, ok := sub.received()[0].(*bus.CancelMessage); !ok {
		t.Errorf("the first message was %T, want the cancel ahead of the queued data",
			sub.received()[0])
	}
}

func TestDataMessagesPreserveFIFOOrder(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)

	var sent []string
	for i := range 5 {
		source := fmt.Sprintf("data_%d", i)
		sent = append(sent, source)
		b.Send(t.Context(), data(source))
	}

	b.Start(t.Context())
	defer b.Stop()

	if !sub.awaitCount(5) {
		t.Fatalf("the subscriber received %d messages, want 5", len(sub.received()))
	}
	if got := sourcesOf(sub.received()); !equal(got, sent) {
		t.Errorf("data messages arrived as %v, want %v", got, sent)
	}
}

func TestSystemMessagesPreserveFIFOOrder(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)

	var sent []string
	for i := range 5 {
		source := fmt.Sprintf("sys_%d", i)
		sent = append(sent, source)
		b.Send(t.Context(), cancel(source))
	}

	b.Start(t.Context())
	defer b.Stop()

	if !sub.awaitCount(5) {
		t.Fatalf("the subscriber received %d messages, want 5", len(sub.received()))
	}
	if got := sourcesOf(sub.received()); !equal(got, sent) {
		t.Errorf("system messages arrived as %v, want %v", got, sent)
	}
}

func TestMixedMessagesSystemFirst(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)

	b.Send(t.Context(), data("d1"))
	b.Send(t.Context(), cancel("s1"))
	b.Send(t.Context(), data("d2"))
	b.Send(t.Context(), cancel("s2"))

	b.Start(t.Context())
	defer b.Stop()

	if !sub.awaitCount(4) {
		t.Fatalf("the subscriber received %d messages, want 4", len(sub.received()))
	}
	want := []string{"s1", "s2", "d1", "d2"}
	if got := sourcesOf(sub.received()); !equal(got, want) {
		t.Errorf("messages arrived as %v, want the system band first: %v", got, want)
	}
}

// Calling a job off must reach the worker ahead of the work it is calling off.
func TestJobCancelIsSystemPriority(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)

	b.Send(t.Context(), data("data"))
	jc := &bus.JobCancelMessage{JobID: "t1"}
	jc.From, jc.To = "parent", "worker"
	b.Send(t.Context(), jc)

	b.Start(t.Context())
	defer b.Stop()

	if !sub.awaitCount(2) {
		t.Fatalf("the subscriber received %d messages, want 2", len(sub.received()))
	}
	if _, ok := sub.received()[0].(*bus.JobCancelMessage); !ok {
		t.Errorf("the first message was %T, want the job cancel first", sub.received()[0])
	}
}

// failingSub panics on its first message and records every one after it. It
// stands for a subscriber whose own handler throws: delivery to it must
// survive, or it silently stops receiving everything afterwards.
type failingSub struct {
	name   string
	mu     sync.Mutex
	got    []bus.Message
	failed bool
}

func newFailingSub() *failingSub {
	return &failingSub{name: fmt.Sprintf("failing_%d", collectorSeq.Add(1))}
}

func (f *failingSub) Name() string { return f.name }

func (f *failingSub) OnBusMessage(_ context.Context, m bus.Message) {
	f.mu.Lock()
	if !f.failed {
		f.failed = true
		f.mu.Unlock()
		panic("simulated subscriber failure") //nolint:forbidigo // the fault this test exists to contain
	}
	f.got = append(f.got, m)
	f.mu.Unlock()
}

func (f *failingSub) received() []bus.Message {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]bus.Message(nil), f.got...)
}

func (f *failingSub) awaitCount(n int) bool {
	for range 200 {
		if len(f.received()) >= n {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return len(f.received()) >= n
}

func TestDataDispatchSurvivesSubscriberPanic(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newFailingSub()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), data("first"))
	settle()
	b.Send(t.Context(), data("second"))

	if !sub.awaitCount(1) {
		t.Fatal("delivery stopped after the subscriber panicked on a data message")
	}
	if got := sourcesOf(sub.received()); !equal(got, []string{"second"}) {
		t.Errorf("the subscriber received %v, want only the message after the fault", got)
	}
}

func TestRouterSurvivesSubscriberPanicOnSystemMessage(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newFailingSub()
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), cancel("first"))
	settle()
	b.Send(t.Context(), cancel("second"))

	if !sub.awaitCount(1) {
		t.Fatal("delivery stopped after the subscriber panicked on a system message")
	}
	if got := sourcesOf(sub.received()); !equal(got, []string{"second"}) {
		t.Errorf("the subscriber received %v, want only the message after the fault", got)
	}
}

// Registering a subscriber already registered, by name, changes nothing.
func TestSubscribeIsIdempotent(t *testing.T) {
	b := bus.NewAsyncQueueBus()
	sub := newCollector()
	b.Subscribe(sub)
	b.Subscribe(sub)
	b.Start(t.Context())
	defer b.Stop()

	b.Send(t.Context(), data("a"))
	settle()

	if got := len(sub.received()); got != 1 {
		t.Errorf("the subscriber received %d copies, want 1", got)
	}
}

// A local-only message never reaches the transport, so a bus with a transport
// still delivers it to the subscribers here.
func TestLocalMessageSkipsTheTransport(t *testing.T) {
	published := 0
	tb := &transportBus{onPublish: func() { published++ }}
	tb.Bus = bus.New(tb)

	sub := newCollector()
	tb.Subscribe(sub)
	tb.Start(t.Context())
	defer tb.Stop()

	local := &bus.WorkerLocalErrorMessage{Error: "boom"}
	local.From = "worker"
	tb.Send(t.Context(), local)

	if !sub.awaitCount(1) {
		t.Fatal("the local message never reached the subscriber")
	}
	if published != 0 {
		t.Errorf("the local message went to the transport %d times, want 0", published)
	}

	tb.Send(t.Context(), data("a"))
	if !sub.awaitCount(2) {
		t.Fatal("the ordinary message never reached the subscriber")
	}
	if published != 1 {
		t.Errorf("the ordinary message went to the transport %d times, want 1", published)
	}
}

// transportBus is a bus with a transport of its own, for the local-message case.
type transportBus struct {
	*bus.Bus
	onPublish func()
}

func (t *transportBus) Publish(_ context.Context, m bus.Message) {
	t.onPublish()
	t.OnMessageReceived(m)
}

func equal(a, b []string) bool { return slices.Equal(a, b) }
