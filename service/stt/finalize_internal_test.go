package stt

import (
	"context"
	"testing"
)

// confirmingStream flushes when told and confirms it did.
type confirmingStream struct {
	ctx context.Context //nolint:containedctx // the session context, set on dial
}

func (s *confirmingStream) Send([]byte) error { return nil }

func (s *confirmingStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *confirmingStream) Close() error { return nil }

func (s *confirmingStream) Finalize() error { return nil }

type confirmingConnector struct{ stream *confirmingStream }

func (c *confirmingConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// newConfirmingService builds a service whose session flushes on request, with
// the finalize already asked for the way the VAD reporting the speech ended
// asks for it.
func newConfirmingService(t *testing.T) *StreamService {
	t.Helper()
	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.sampleRate = 16000
	s.finalizeRequested = true
	return s
}

// A confirmation only counts for the transcript it belongs to. The one after it
// is an ordinary transcript again, whatever the provider said before.
func TestConfirmedFinalizeMarksOneTranscript(t *testing.T) {
	t.Parallel()

	s := newConfirmingService(t)
	s.confirmFinalize()

	if !s.takeFinalizePending() {
		t.Fatal("a confirmed finalize did not mark the transcript that answers it")
	}
	if s.takeFinalizePending() {
		t.Fatal("a confirmed finalize marked a second transcript")
	}
}

// A provider confirming a finalize nobody asked for says nothing: the flush it
// reports is not one this session is waiting on.
func TestUnaskedConfirmationIsIgnored(t *testing.T) {
	t.Parallel()

	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.confirmFinalize()

	if s.takeFinalizePending() {
		t.Fatal("a confirmation for a finalize that was never asked for marked a transcript")
	}
}

// The answer to a finalize can arrive with nothing left to say, when the
// transcript went out ahead of it. It still closes the utterance, so the
// confirmation is taken from a result carrying no text.
func TestEmptyResultStillConfirmsTheFinalize(t *testing.T) {
	t.Parallel()

	s := newConfirmingService(t)
	s.emit(context.Background(), Result{Final: true, FromFinalize: true})

	if !s.takeFinalizePending() {
		t.Fatal("an empty answer to a finalize did not confirm it")
	}
}

// An interim result says nothing about a finalize, whatever it carries.
func TestInterimResultDoesNotConfirmTheFinalize(t *testing.T) {
	t.Parallel()

	s := newConfirmingService(t)
	s.emit(context.Background(), Result{Text: "hel", FromFinalize: true})

	if s.takeFinalizePending() {
		t.Fatal("an interim result confirmed a finalize")
	}
}
