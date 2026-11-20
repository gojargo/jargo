package stt

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
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
	s.sampleRate, s.sendRate = 16000, 16000
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

// pushedFrames records what s pushes, below its own PushFrame, so the handling
// of a finalize there still runs.
func pushedFrames(s *StreamService) func() []frames.Frame {
	var mu sync.Mutex
	var got []frames.Frame
	s.Events().Add(processor.EventAfterPushFrame, func(_ context.Context, _ any, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, pushedFrame(args))
	})
	return func() []frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return append([]frames.Frame(nil), got...)
	}
}

// The answer to a finalize can arrive with nothing left to say, when the
// transcript went out ahead of it. It still closes the utterance, so the
// confirmation is taken from a result carrying no text, and a provider that
// says it confirms the transcripts already pushed has it go out as a frame of
// its own: those transcripts cannot be marked any more.
func TestEmptyResultStillConfirmsTheFinalize(t *testing.T) {
	t.Parallel()

	s := newConfirmingService(t)
	pushed := pushedFrames(s)
	s.emit(context.Background(), Result{Final: true, FromFinalize: true, PushFinalized: true})

	got := pushed()
	if len(got) != 1 {
		t.Fatalf("pushed %v, want the confirmation alone", got)
	}
	if _, ok := got[0].(*frames.TranscriptionFinalizedFrame); !ok {
		t.Fatalf("pushed %v, want a TranscriptionFinalizedFrame", got[0])
	}
	if s.takeFinalizePending() {
		t.Fatal("the confirmation was left to mark the next transcript as well")
	}
}

// The transcript arrived before the service confirmed it was the last.
func TestAConfirmationWithoutTextIsPushedAsAFrameOfItsOwn(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.userID = "session-user"
	pushed := pushedFrames(s)

	s.emit(ctx, Result{Text: "Hi.", Final: true})
	// A measurement is under way, as the VAD reporting the speech over starts.
	s.ttfb.mu.Lock()
	s.ttfb.start = time.Now()
	s.ttfb.mu.Unlock()
	s.finalizeRequested = true
	s.confirmFinalize()
	s.pushTranscriptionFinalized(ctx)

	got := pushed()
	if len(got) != 2 {
		t.Fatalf("pushed %v, want the transcript and the confirmation", got)
	}
	transcript, ok := got[0].(*frames.TranscriptionFrame)
	if !ok || transcript.Text != "Hi." || transcript.Finalized {
		t.Fatalf("first = %v, want the transcript, not finalized", got[0])
	}
	confirmation, ok := got[1].(*frames.TranscriptionFinalizedFrame)
	if !ok {
		t.Fatalf("second = %v, want a TranscriptionFinalizedFrame", got[1])
	}
	if confirmation.UserID != "session-user" {
		t.Errorf("confirmation user = %q, want session-user", confirmation.UserID)
	}
	// The wait for the final transcript ends at the confirmation.
	s.ttfb.mu.Lock()
	measuring := !s.ttfb.start.IsZero()
	s.ttfb.mu.Unlock()
	if measuring {
		t.Error("the confirmation did not end the wait for the final transcript")
	}

	// One confirmation finalizes once.
	s.pushTranscriptionFinalized(ctx)
	if got := pushed(); len(got) != 2 {
		t.Fatalf("pushed %v after a second call, want nothing more", got)
	}
}

// Nothing is finalized unless the service asked and the provider confirmed.
func TestNoFinalizedFrameWithoutAConfirmedFinalization(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	pushed := pushedFrames(s)

	s.pushTranscriptionFinalized(ctx)
	s.finalizeRequested = true
	s.pushTranscriptionFinalized(ctx)

	if got := pushed(); len(got) != 0 {
		t.Fatalf("pushed %v, want nothing", got)
	}
}

// An utterance is finalized by its transcript or by the frame, never both.
func TestAConfirmationWithTextFinalizesItsTranscriptOnly(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	s := newConfirmingService(t)
	pushed := pushedFrames(s)

	s.confirmFinalize()
	s.emit(ctx, Result{Text: "Hi.", Final: true})
	s.pushTranscriptionFinalized(ctx)

	got := pushed()
	if len(got) != 1 {
		t.Fatalf("pushed %v, want the transcript alone", got)
	}
	if tf, ok := got[0].(*frames.TranscriptionFrame); !ok || !tf.Finalized {
		t.Fatalf("pushed %v, want the finalized transcript", got[0])
	}
}

// A confirmation that arrives after the user spoke again belongs to no
// utterance.
func TestAnEmptyAnswerNobodyAskedForFinalizesNothing(t *testing.T) {
	t.Parallel()

	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	pushed := pushedFrames(s)

	s.emit(context.Background(), Result{Final: true, FromFinalize: true})

	if got := pushed(); len(got) != 0 {
		t.Fatalf("pushed %v, want nothing", got)
	}
}

// With no session open no finalize goes out, so none is waited on.
func TestAVADStopWithNoSessionRequestsNoFinalization(t *testing.T) {
	t.Parallel()

	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.finalize()

	if s.finalizeRequested {
		t.Fatal("a finalize was requested of no session")
	}
}

// An empty answer that only confirms the finalize pushes nothing: the
// confirmation is left to mark the next transcript.
func TestAnEmptyAnswerLeavesTheConfirmationToTheNextTranscript(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	s := newConfirmingService(t)
	pushed := pushedFrames(s)

	s.emit(ctx, Result{Final: true, FromFinalize: true})
	if got := pushed(); len(got) != 0 {
		t.Fatalf("pushed %v, want nothing for the confirmation", got)
	}
	s.emit(ctx, Result{Text: "Hi.", Final: true})

	got := pushed()
	if len(got) != 1 {
		t.Fatalf("pushed %v, want the next transcript", got)
	}
	if tf, ok := got[0].(*frames.TranscriptionFrame); !ok || !tf.Finalized {
		t.Fatalf("pushed %v, want the next transcript finalized", got[0])
	}
}

// The service asks the provider to finalize on a VAD stop, and records no
// request of its own: a provider that confirms its finalizes records the one it
// sends, and one that does not leaves nothing to confirm.
func TestAVADStopLeavesTheRequestToTheProvider(t *testing.T) {
	t.Parallel()

	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.stream = &confirmingStream{}
	s.finalize()

	if s.finalizeRequested {
		t.Fatal("the service recorded a finalize the provider did not")
	}
}

// A provider offering a finalize of its own records it, so the answer confirms
// it as it does the one the VAD stop asks for.
func TestARequestedFinalizeIsConfirmedByItsAnswer(t *testing.T) {
	t.Parallel()

	s := NewStream("FinalizingSTT", &confirmingConnector{stream: &confirmingStream{}}, 16000)
	s.RequestFinalize()
	s.confirmFinalize()

	if !s.takeFinalizePending() {
		t.Fatal("the answer to a requested finalize did not confirm it")
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
