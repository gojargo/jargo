package novasonic

import (
	"context"
	"reflect"
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
	s := New(Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"})
	sink := newCapture()
	s.Link(sink)

	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	for _, p := range []processor.Processor{sink, s} {
		if err := p.Setup(ctx, setup); err != nil {
			t.Fatalf("setup %s: %v", p.Name(), err)
		}
		t.Cleanup(func() { _ = p.Cleanup(ctx) })
	}
	if err := s.Base.ProcessFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("start: %v", err)
	}
	return s, sink
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
	s, sink := newTestService(t)
	ctx := context.Background()

	s.handleText(ctx, "USER", "what is the weather")
	s.handleText(ctx, "ASSISTANT", "it is raining")

	got := sink.names()
	want := []string{"TranscriptionFrame", "LLMTextFrame"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
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
