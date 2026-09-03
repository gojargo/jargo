package stt

import (
	"context"
	"sync"
	"testing"
	"time"
)

// wdStream records what was submitted on it and stays open until the session is
// canceled, so the read loop has no drop to recover from.
type wdStream struct {
	mu   sync.Mutex
	sent [][]byte
	ctx  context.Context //nolint:containedctx // the session context, set on dial
}

func (s *wdStream) Send(audio []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sent = append(s.sent, append([]byte(nil), audio...))
	return nil
}

func (s *wdStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *wdStream) Close() error { return nil }

func (s *wdStream) sends() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]byte(nil), s.sent...)
}

// wdConnector hands out one stream and asks for a watchdog.
type wdConnector struct {
	stream *wdStream
	opts   TurnWatchdogOptions
	wants  bool
}

func (c *wdConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

func (c *wdConnector) TurnWatchdog() TurnWatchdogOptions {
	if !c.wants {
		return TurnWatchdogOptions{}
	}
	return c.opts
}

// watchdogService builds a service with a session already in place, so the
// watchdog can be driven directly without dialing anything.
func watchdogService(t *testing.T, opts TurnWatchdogOptions) (*StreamService, *wdStream) {
	t.Helper()
	st := &wdStream{}
	s := NewStream("FakeSTT", &wdConnector{stream: st, wants: true, opts: opts}, 16000)
	s.sampleRate = 16000
	s.stream = st
	return s, st
}

// The watchdog stands in for the audio that stopped, so a provider that decides
// the end of a turn from the audio can reach one.
func TestTurnWatchdogSubmitsSilenceMidTurn(t *testing.T) {
	t.Parallel()

	s, st := watchdogService(t, TurnWatchdogOptions{
		MinTimeout: 20 * time.Millisecond, Interval: 5 * time.Millisecond,
	})
	// A turn is open and the audio stopped long enough ago to have stalled it.
	s.speaking = true
	s.lastAudio = time.Now().Add(-200 * time.Millisecond)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go s.turnWatchdogLoop(ctx)

	deadline := time.Now().Add(2 * time.Second)
	for len(st.sends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	sends := st.sends()
	if len(sends) == 0 {
		t.Fatal("no silence was submitted for a turn the audio stopped under")
	}
	for i, b := range sends[0] {
		if b != 0 {
			t.Fatalf("byte %d of the submission is %d, want silence", i, b)
		}
	}
	// Silence submitted as audio is audio the provider bills for.
	s.mu.Lock()
	billed := s.audioBytes
	s.mu.Unlock()
	if billed != int64(len(sends[0])) {
		t.Errorf("billed %d bytes, want the %d submitted", billed, len(sends[0]))
	}
}

// Outside a turn there is nothing stalled, and holding an idle session open is
// the keepalive's business rather than the watchdog's.
func TestTurnWatchdogStaysQuietOutsideATurn(t *testing.T) {
	t.Parallel()

	s, st := watchdogService(t, TurnWatchdogOptions{
		MinTimeout: 20 * time.Millisecond, Interval: 5 * time.Millisecond,
	})
	s.speaking = false
	s.lastAudio = time.Now().Add(-200 * time.Millisecond)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go s.turnWatchdogLoop(ctx)

	time.Sleep(100 * time.Millisecond)
	if n := len(st.sends()); n != 0 {
		t.Errorf("submitted %d times with no turn open, want none", n)
	}
}

// The gap waited for is at least two chunks, so a pipeline sending long chunks
// is not nudged in the ordinary gap between two of them.
func TestTurnWatchdogWaitsOutTwoChunks(t *testing.T) {
	t.Parallel()

	s, st := watchdogService(t, TurnWatchdogOptions{
		MinTimeout: 10 * time.Millisecond, Interval: 5 * time.Millisecond,
	})
	s.speaking = true
	s.lastAudio = time.Now()
	// Chunks of a second each, so the gap that counts as a stall is two seconds.
	s.lastChunk = time.Second

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go s.turnWatchdogLoop(ctx)

	time.Sleep(100 * time.Millisecond)
	if n := len(st.sends()); n != 0 {
		t.Errorf("submitted %d times inside one chunk interval, want none", n)
	}
}

// A provider that does not implement the interface gets no watchdog.
func TestTurnWatchdogIsOptIn(t *testing.T) {
	t.Parallel()

	st := &wdStream{}
	plain := NewStream("FakeSTT", &noWatchdogConnector{stream: st}, 16000)
	if plain.watchdogWanted {
		t.Error("a provider that does not ask for a watchdog was given one")
	}

	asked := NewStream("FakeSTT", &wdConnector{stream: st, wants: true}, 16000)
	if !asked.watchdogWanted {
		t.Error("a provider that asked for a watchdog was not given one")
	}
}

// noWatchdogConnector leaves a stalled turn alone, the way a provider whose turn
// boundary does not depend on the audio continuing to flow does.
type noWatchdogConnector struct{ stream *wdStream }

func (c *noWatchdogConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// A provider that asks without saying how long to wait gets the defaults.
func TestTurnWatchdogDefaults(t *testing.T) {
	t.Parallel()

	st := &wdStream{}
	s := NewStream("FakeSTT", &wdConnector{stream: st, wants: true}, 16000)
	if s.watchdog.MinTimeout != DefaultTurnWatchdogMinTimeout {
		t.Errorf("MinTimeout = %s, want %s", s.watchdog.MinTimeout, DefaultTurnWatchdogMinTimeout)
	}
	if s.watchdog.Interval != DefaultTurnWatchdogInterval {
		t.Errorf("Interval = %s, want %s", s.watchdog.Interval, DefaultTurnWatchdogInterval)
	}
}
