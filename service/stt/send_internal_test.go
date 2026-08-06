package stt

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// errSendFailed is what a session that has dropped underneath returns.
//
//nolint:gochecknoglobals // sentinel error
var errSendFailed = errors.New("session gone")

// sendStream counts what it was given and refuses it while broken, the way a
// session whose socket has gone reports every write.
type sendStream struct {
	mu     sync.Mutex
	sends  int
	broken bool
	ctx    context.Context //nolint:containedctx // the session context, set on dial
}

func (s *sendStream) Send([]byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sends++
	if s.broken {
		return errSendFailed
	}
	return nil
}

func (s *sendStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *sendStream) Close() error { return nil }

func (s *sendStream) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sends
}

// sendConnector hands out the streams it was built with, one per session.
type sendConnector struct {
	streams []*sendStream
	idx     int
}

func (c *sendConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	st := c.streams[c.idx]
	c.idx++
	st.ctx = ctx
	return st, nil
}

// A session that refuses the audio is taken out of use, so the rest of the call
// is not written into a socket that is not listening.
func TestSendStopsAtTheFirstFailure(t *testing.T) {
	t.Parallel()

	st := &sendStream{broken: true}
	s := NewStream("FakeSTT", &sendConnector{streams: []*sendStream{st}}, 16000)
	s.sampleRate = 16000
	s.stream = st

	audio := make([]byte, 320)
	for range 5 {
		s.send(audio)
	}

	if got := st.count(); got != 1 {
		t.Fatalf("sends after a failure = %d, want 1", got)
	}
}

// Reopening the session is what puts it back in use: the audio spoken after it
// has somewhere to go again.
func TestSendResumesOnANewSession(t *testing.T) {
	t.Parallel()

	broken := &sendStream{broken: true}
	fresh := &sendStream{}
	conn := &sendConnector{streams: []*sendStream{broken, fresh}}
	s := NewStream("FakeSTT", conn, 16000)
	s.sampleRate = 16000

	ctx := t.Context()
	if err := s.ConnectWebsocket(ctx); err != nil {
		t.Fatalf("ConnectWebsocket: %v", err)
	}

	audio := make([]byte, 320)
	s.send(audio)
	s.send(audio)
	if got := broken.count(); got != 1 {
		t.Fatalf("sends on the broken session = %d, want 1", got)
	}

	if err := s.DisconnectWebsocket(ctx); err != nil {
		t.Fatalf("DisconnectWebsocket: %v", err)
	}
	if err := s.ConnectWebsocket(ctx); err != nil {
		t.Fatalf("reopening: %v", err)
	}

	s.send(audio)
	s.send(audio)
	if got := fresh.count(); got != 2 {
		t.Fatalf("sends on the reopened session = %d, want 2", got)
	}
}
