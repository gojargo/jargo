package gradium

import (
	"errors"
	"testing"
	"time"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// newTestStream builds a stream with no connection under it. The reader
// goroutine is what talks to the connection, so feeding its channel drives the
// stream exactly as the server would.
func newTestStream() *sttStream {
	return &sttStream{
		lang:  "en",
		reads: make(chan sttRead, 16),
		done:  make(chan struct{}),
	}
}

// feed queues messages as the reader would have delivered them.
func (s *sttStream) feed(msgs ...sttMessage) {
	for _, m := range msgs {
		s.reads <- sttRead{msg: m}
	}
}

func text(t string) sttMessage   { return sttMessage{Type: msgText, Text: t} }
func flushed() sttMessage        { return sttMessage{Type: msgFlushed} }
func errMsg(m string) sttMessage { return sttMessage{Type: msgError, Message: m} }
func recvOne(t *testing.T, s *sttStream) stt.Result {
	t.Helper()
	got, err := s.Recv()
	if err != nil {
		t.Fatalf("recv: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d results, want 1", len(got))
	}
	return got[0]
}

// TestRecvSurfacesTheRunningTranscript covers what arrives while the user is
// still speaking. Each fragment is added to the transcript so far, and the whole
// of it goes out as an interim, so what is shown grows with the utterance rather
// than flickering between fragments.
func TestRecvSurfacesTheRunningTranscript(t *testing.T) {
	s := newTestStream()
	s.feed(text("hello"), text("there"))

	first := recvOne(t, s)
	if first.Text != "hello" || first.Final {
		t.Errorf("first result = %q (final %v), want %q interim", first.Text, first.Final, "hello")
	}
	second := recvOne(t, s)
	if second.Text != "hello there" || second.Final {
		t.Errorf("second result = %q (final %v), want %q interim", second.Text, second.Final, "hello there")
	}
}

// TestRecvKeepsTheWordsThatArriveAfterAFlush covers the end of an utterance. The
// flush says the buffered audio has been processed, but the words at the end of
// it can still be on their way. Settling the transcript the moment the flush
// arrives drops them, and because the transcript is cleared with it, they then
// turn up at the front of whatever the user says next.
func TestRecvKeepsTheWordsThatArriveAfterAFlush(t *testing.T) {
	s := newTestStream()
	// The trailing fragment is already on its way when the flush is handled.
	s.feed(text("what is the"), flushed(), text("weather today"))

	if got := recvOne(t, s); got.Text != "what is the" {
		t.Fatalf("interim = %q, want %q", got.Text, "what is the")
	}

	final := recvOne(t, s)
	if !final.Final || !final.EndOfTurn {
		t.Error("the flush did not settle the transcript")
	}
	if final.Text != "what is the weather today" {
		t.Errorf("final = %q, want %q: the words arriving after the flush were dropped",
			final.Text, "what is the weather today")
	}

	// Nothing may be carried into the next utterance.
	if len(s.accumulated) != 0 {
		t.Errorf("the stream still holds %v, which would open the next utterance", s.accumulated)
	}
}

// TestRecvSettlesOnceNothingMoreArrives covers a flush with nothing behind it.
// The transcript stays open only for the aggregation window, so an utterance
// that really has ended is not held up.
func TestRecvSettlesOnceNothingMoreArrives(t *testing.T) {
	s := newTestStream()
	s.feed(text("goodbye"), flushed())

	recvOne(t, s) // the interim

	started := time.Now()
	final := recvOne(t, s)
	elapsed := time.Since(started)

	if final.Text != "goodbye" || !final.Final {
		t.Errorf("final = %q (final %v), want %q", final.Text, final.Final, "goodbye")
	}
	if elapsed < transcriptAggregationDelay {
		t.Errorf("settled after %v, want at least the %v window", elapsed, transcriptAggregationDelay)
	}
}

// TestRecvHoldsWhatItReadWhileWaiting covers something other than text arriving
// inside the aggregation window. It was read to see whether it was a trailing
// word, and has to be acted on afterwards rather than swallowed.
func TestRecvHoldsWhatItReadWhileWaiting(t *testing.T) {
	s := newTestStream()
	s.feed(text("hello"), flushed(), errMsg("stream failed"))

	recvOne(t, s) // the interim

	final := recvOne(t, s)
	if final.Text != "hello" {
		t.Fatalf("final = %q, want %q", final.Text, "hello")
	}

	if _, err := s.Recv(); err == nil {
		t.Error("the error read during the window was swallowed")
	} else if !errors.Is(err, errSTTProtocol) {
		t.Errorf("error = %v, want the protocol error", err)
	}
}

// TestRecvIgnoresAFlushWithNothingToSettle covers the flush that follows silence.
// There is no transcript to settle, so nothing is emitted: a final carrying an
// empty transcript would end a turn the user never took.
func TestRecvIgnoresAFlushWithNothingToSettle(t *testing.T) {
	s := newTestStream()
	s.feed(flushed(), text("now speaking"))

	got := recvOne(t, s)
	if got.Text != "now speaking" || got.Final {
		t.Errorf("result = %q (final %v), want the interim that followed the empty flush",
			got.Text, got.Final)
	}
}

// TestRecvEndsWhenTheReaderStops covers the connection going away: the stream
// reports the end rather than blocking on a channel nobody will write to again.
func TestRecvEndsWhenTheReaderStops(t *testing.T) {
	s := newTestStream()
	close(s.reads)

	if _, err := s.Recv(); err == nil {
		t.Error("the stream did not report the reader having stopped")
	}
}

// TestInputFormatNamesTheRate covers the format the session is opened with. PCM
// carries its rate in the name, so the server decodes the audio at the rate it
// was captured; an unsupported rate falls back to the one the service always
// accepts rather than failing the session.
func TestInputFormatNamesTheRate(t *testing.T) {
	tests := []struct {
		name     string
		encoding string
		rate     int
		want     string
	}{
		{"8 kHz", encPCM, 8000, "pcm_8000"},
		{"16 kHz", encPCM, 16000, "pcm_16000"},
		{"24 kHz", encPCM, 24000, "pcm_24000"},
		{"an unsupported rate", encPCM, 44100, "pcm_16000"},
		{"another encoding carries no rate", "opus", 48000, "opus"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &sttConnector{cfg: STTConfig{Encoding: tt.encoding}}
			if got := c.inputFormat(tt.rate); got != tt.want {
				t.Errorf("input format = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGradiumLanguageUsesTheBaseCode covers the language hint. The service takes
// a base code, so a regional language is reduced to it, and no language at all
// leaves the hint off so the server decides.
func TestGradiumLanguageUsesTheBaseCode(t *testing.T) {
	if got := gradiumLanguage(language.FrenchCA); got != "fr" {
		t.Errorf("fr-CA mapped to %q, want %q", got, "fr")
	}
	if got := gradiumLanguage(""); got != "" {
		t.Errorf("the empty language mapped to %q, want it left off", got)
	}
}
