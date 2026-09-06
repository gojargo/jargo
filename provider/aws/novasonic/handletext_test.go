package novasonic

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// capture is a sink that records every frame reaching it. It runs in direct
// mode, so a frame pushed into it has been recorded by the time the push
// returns and a test needs no synchronization of its own.
type capture struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

func newCapture() *capture {
	c := &capture{}
	c.Base = processor.New("Capture", c, processor.WithDirectMode())
	return c
}

func (c *capture) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	c.mu.Lock()
	c.got = append(c.got, f)
	c.mu.Unlock()
	return nil
}

// names returns the recorded frame types, minus the lifecycle frames every
// processor sees, so a test can assert on what the service itself emitted.
func (c *capture) names() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, f := range c.got {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
			continue
		case *frames.LLMServiceMetadataFrame:
			// Every service describes itself once it starts; the tests assert on
			// what this one made of the model's messages.
			continue
		}
		// Names carry a per-instance suffix ("#7"); the type is what matters.
		name, _, _ := strings.Cut(f.Name(), "#")
		out = append(out, name)
	}
	return out
}

// newTestService builds a service wired to a sink and started, so the frames it
// pushes are recorded. No session is opened: these tests drive the transcript
// handler directly, which is where the service's messages become frames.
func newTestService(t *testing.T) (*Service, *capture) {
	t.Helper()
	s, sink, _ := newTestServiceBothWays(t)
	return s, sink
}

// newTestServiceBothWays builds a service with a sink on each side, since what
// the user said travels towards the user aggregator, which sits before the
// service, while the model's own output travels on.
func newTestServiceBothWays(t *testing.T) (*Service, *capture, *capture) {
	t.Helper()
	s := New(Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"})
	sink := newCapture()
	source := newCapture()
	s.Link(sink)
	source.Link(s)

	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	for _, p := range []processor.Processor{source, sink, s} {
		if err := p.Setup(ctx, setup); err != nil {
			t.Fatalf("setup %s: %v", p.Name(), err)
		}
		t.Cleanup(func() { _ = p.Cleanup(ctx) })
	}
	if err := s.Base.ProcessFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("start: %v", err)
	}
	return s, sink, source
}

// TestBargeInEmitsNoUserSpeech covers the barge-in marker, which is the only
// turn-related thing this service reports. The bot gives the floor up and the
// pipeline is interrupted, but no user-speaking frame is invented.
//
// The service reports a barge-in and never a turn starting or ending, so a start
// emitted here would never be matched by a stop, and everything keyed off those
// frames would be left believing the user is still speaking. The reference
// implementation makes the same choice for the same reason: it broadcasts the
// interruption alone.
func TestBargeInEmitsNoUserSpeech(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()

	// The bot is speaking when the barge-in lands.
	s.setSpeaking(ctx, true)
	s.handleText(ctx, "USER", `{"interrupted":true}`)

	got := sink.names()
	want := []string{"BotStartedSpeakingFrame", "BotStoppedSpeakingFrame", "InterruptionFrame"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
	}
	for _, n := range got {
		if n == "UserStartedSpeakingFrame" || n == "UserStoppedSpeakingFrame" {
			t.Errorf("the service invented a %s: it reports no turn boundaries", n)
		}
	}
}

// TestTranscriptsGoBothWays covers the two transcripts the service sends. What
// the user said becomes a transcription; what the model said becomes the model's
// own text, since it is the reply rather than the request.
func TestTranscriptsGoBothWays(t *testing.T) {
	s, sink, source := newTestServiceBothWays(t)
	ctx := context.Background()

	s.handleText(ctx, "USER", "what is the weather")
	s.handleText(ctx, "ASSISTANT", "it is raining")

	// The model's own text is the reply, so it travels on.
	if got, want := sink.names(), []string{"LLMTextFrame"}; !reflect.DeepEqual(got, want) {
		t.Errorf("frames going on = %v, want %v", got, want)
	}
	// What the user said goes back towards the user aggregator. Pushed on it
	// would reach the assistant aggregator instead, and the conversation would
	// hold no record of what the user said.
	if got, want := source.names(), []string{"TranscriptionFrame"}; !reflect.DeepEqual(got, want) {
		t.Errorf("frames going back = %v, want %v", got, want)
	}
}

// TestSpeculativeAssistantTextIsHeldBack covers the model thinking aloud. A
// speculative transcript may still be revised, so forwarding it would put words
// into the conversation that the model has not committed to saying.
func TestSpeculativeAssistantTextIsHeldBack(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()

	s.assistantSpeculative = true
	s.handleText(ctx, "ASSISTANT", "maybe this")

	if got := sink.names(); len(got) != 0 {
		t.Errorf("frames = %v, want none: the transcript was speculative", got)
	}
}

// TestFramesTravelOnce covers what the service forwards. The LLM base forwards
// what it handles, so a service pushing again would deliver two of every frame,
// and describe itself to the pipeline twice with them. The audio is the
// exception: the model consumes it, so it must not travel on at all.
func TestFramesTravelOnce(t *testing.T) {
	s, sink, _ := newTestServiceBothWays(t)
	ctx := context.Background()

	if err := s.ProcessFrame(ctx, frames.NewInputAudioRawFrame([]byte{1, 2}, 16000, 1), processor.Downstream); err != nil {
		t.Fatalf("audio: %v", err)
	}
	if err := s.ProcessFrame(ctx, frames.NewEndFrame(), processor.Downstream); err != nil {
		t.Fatalf("end: %v", err)
	}

	ends, audio := 0, 0
	for _, f := range sink.frames() {
		switch f.(type) {
		case *frames.EndFrame:
			ends++
		case *frames.InputAudioRawFrame:
			audio++
		}
	}
	if ends != 1 {
		t.Errorf("EndFrames forwarded = %d, want 1", ends)
	}
	if audio != 0 {
		t.Errorf("input audio frames forwarded = %d, want none: the model consumes them", audio)
	}
}

// TestBargeInIsBroadcast covers where the interruption goes. Both aggregators
// act on one, and the one keeping the user's turn sits before this service, so
// an interruption pushed on alone would never reach it.
func TestBargeInIsBroadcast(t *testing.T) {
	s, sink, source := newTestServiceBothWays(t)
	ctx := context.Background()

	s.setSpeaking(ctx, true)
	s.handleText(ctx, "USER", `{"interrupted":true}`)

	if !hasName(sink.names(), "InterruptionFrame") {
		t.Error("the interruption did not travel on")
	}
	if !hasName(source.names(), "InterruptionFrame") {
		t.Error("the interruption did not travel back, where the user aggregator is")
	}
}

// hasName reports whether names holds one.
func hasName(names []string, want string) bool { return slices.Contains(names, want) }

// frames returns the recorded frames themselves, for a test asserting on more
// than their types.
func (c *capture) frames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.got...)
}
