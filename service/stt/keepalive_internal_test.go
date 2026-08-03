package stt

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// kaStream records what was submitted on it, and stays open until the session is
// canceled so the read loop has no drop to recover from.
type kaStream struct {
	mu   sync.Mutex
	sent [][]byte
	// custom holds what SendKeepalive received, for the stream that overrides it.
	custom [][]byte
	ctx    context.Context //nolint:containedctx // the session context, set on dial
}

func (s *kaStream) Send(audio []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sent = append(s.sent, append([]byte(nil), audio...))
	return nil
}

func (s *kaStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *kaStream) Close() error { return nil }

func (s *kaStream) sends() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]byte(nil), s.sent...)
}

func (s *kaStream) keepalives() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.custom)
}

// kaProtocolStream swaps the silence for a message of its own, the way a
// provider with a protocol-level keepalive does.
type kaProtocolStream struct{ kaStream }

func (s *kaProtocolStream) SendKeepalive(silence []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.custom = append(s.custom, append([]byte(nil), silence...))
	return nil
}

// kaConnector hands out one stream and reports what keepalive the provider wants.
type kaConnector struct {
	stream Stream
	opts   KeepaliveOptions
	// wants is false for a provider that leaves an idle session alone.
	wants bool
}

func (c *kaConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	switch st := c.stream.(type) {
	case *kaStream:
		st.ctx = ctx
	case *kaProtocolStream:
		st.ctx = ctx
	}
	return c.stream, nil
}

func (c *kaConnector) Keepalive() KeepaliveOptions {
	if !c.wants {
		return KeepaliveOptions{}
	}
	return c.opts
}

// silenceBytes is one keepalive's worth of 16-bit mono silence at rate.
func silenceBytes(rate int) int {
	return int(float64(rate)*keepaliveSilence.Seconds()) * 2
}

// Silence submitted as audio is audio the provider bills for, so it counts
// towards the session's usage.
func TestKeepaliveSilenceCountsTowardUsage(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{Timeout: time.Second}}
	s := NewStream("FakeSTT", conn, 16000)
	s.sampleRate = 16000
	s.stream = st

	if err := s.sendKeepalive(); err != nil {
		t.Fatalf("sending the keepalive failed: %v", err)
	}

	want := int64(silenceBytes(16000))
	if s.audioBytes != want {
		t.Errorf("billed %d bytes of audio, want %d", s.audioBytes, want)
	}
	sends := st.sends()
	if len(sends) != 1 {
		t.Fatalf("submitted %d times, want 1", len(sends))
	}
	if len(sends[0]) != int(want) {
		t.Errorf("submitted %d bytes, want %d", len(sends[0]), want)
	}
	for i, b := range sends[0] {
		if b != 0 {
			t.Fatalf("byte %d of the keepalive is %d, want silence", i, b)
		}
	}
}

// A provider that swaps the silence for a message of its own submitted no audio,
// so there is nothing to bill for.
func TestKeepaliveProtocolMessageIsNotBilled(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{Timeout: time.Second}}
	s := NewStream("FakeSTT", conn, 16000)
	s.sampleRate = 16000
	s.stream = st

	if err := s.sendKeepalive(); err != nil {
		t.Fatalf("sending the keepalive failed: %v", err)
	}

	if s.audioBytes != 0 {
		t.Errorf("billed %d bytes of audio for a protocol keepalive, want 0", s.audioBytes)
	}
	if got := st.keepalives(); got != 1 {
		t.Errorf("sent %d protocol keepalives, want 1", got)
	}
	if got := len(st.sends()); got != 0 {
		t.Errorf("submitted audio %d times, want 0: the provider took the protocol route", got)
	}
}

// runKeepaliveService runs svc until the returned stop is called.
func runKeepaliveService(t *testing.T, svc *StreamService) (task *pipeline.Task, stop func()) {
	t.Helper()
	task = pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		if cond() {
			return
		}
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// A session carrying no audio is held open rather than left to be closed.
func TestKeepaliveFiresWhenTheSessionGoesIdle(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{
		Timeout:  20 * time.Millisecond,
		Interval: 5 * time.Millisecond,
	}}
	svc := NewStream("FakeSTT", conn, 16000)
	_, stop := runKeepaliveService(t, svc)
	defer stop()

	waitFor(t, "the idle session to be kept alive", func() bool { return st.keepalives() > 0 })
}

// A session carrying audio is not idle, so nothing is sent to hold it open.
func TestKeepaliveStaysQuietWhileAudioFlows(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{
		Timeout:  150 * time.Millisecond,
		Interval: 5 * time.Millisecond,
	}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()

	waitFor(t, "the session to open", func() bool { return svc.Connected() })
	// Audio every 20ms for well past one timeout's worth of time.
	for range 15 {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 64), 16000, 1))
		time.Sleep(20 * time.Millisecond)
	}

	if got := st.keepalives(); got != 0 {
		t.Errorf("sent %d keepalives while audio was flowing, want 0", got)
	}
}

// A provider that leaves an idle session alone is not given a keepalive.
func TestKeepaliveOffWhenTheProviderDoesNotAskForIt(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st} // does not want a keepalive
	svc := NewStream("FakeSTT", conn, 16000)
	_, stop := runKeepaliveService(t, svc)
	defer stop()

	waitFor(t, "the session to open", func() bool { return svc.Connected() })
	time.Sleep(100 * time.Millisecond)

	if got := st.keepalives(); got != 0 {
		t.Errorf("sent %d keepalives for a provider that asked for none, want 0", got)
	}
	svc.mu.Lock()
	running := svc.keepaliveCancel != nil
	svc.mu.Unlock()
	if running {
		t.Error("a keepalive is running for a provider that asked for none")
	}
}

// Shutting the session down stops the keepalive with it.
func TestKeepaliveStopsWithTheSession(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{
		Timeout:  10 * time.Millisecond,
		Interval: 5 * time.Millisecond,
	}}
	svc := NewStream("FakeSTT", conn, 16000)
	_, stop := runKeepaliveService(t, svc)

	waitFor(t, "the idle session to be kept alive", func() bool { return st.keepalives() > 0 })
	stop()

	svc.mu.Lock()
	running := svc.keepaliveCancel != nil
	svc.mu.Unlock()
	if running {
		t.Error("the keepalive outlived the session it was holding open")
	}

	settled := st.keepalives()
	time.Sleep(50 * time.Millisecond)
	if got := st.keepalives(); got != settled {
		t.Errorf("keepalives went from %d to %d after shutdown, want no more", settled, got)
	}
}

// The interval a provider leaves unset takes the default rather than spinning.
func TestKeepaliveIntervalDefaults(t *testing.T) {
	t.Parallel()

	conn := &kaConnector{stream: &kaStream{}, wants: true, opts: KeepaliveOptions{Timeout: time.Second}}
	svc := NewStream("FakeSTT", conn, 16000)

	if svc.keepalive.Interval != DefaultKeepaliveInterval {
		t.Errorf("interval = %v, want %v", svc.keepalive.Interval, DefaultKeepaliveInterval)
	}
}

var _ processor.Processor = (*StreamService)(nil)
