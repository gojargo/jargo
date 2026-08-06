package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
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

// names returns the recorded frame names, minus the lifecycle frames every
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

// newTestService builds a service wired to a sink and started, so that frames it
// pushes are recorded. The session is never opened: these tests drive the
// message handler directly, which is where the server's messages become frames.
func newTestService(t *testing.T) (*Service, *capture) {
	t.Helper()
	s := New(Config{APIKey: "k", Model: "m"})
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
	// Started through the base only: the service's own ProcessFrame would open a
	// session, and these tests drive the message handler instead.
	if err := s.Base.ProcessFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("start: %v", err)
	}
	return s, sink
}

// decode builds a server message from the wire form the service receives, so
// the tests exercise the field names as well as the handling.
func decode(t *testing.T, raw string) serverMessage {
	t.Helper()
	var msg serverMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("decode server message: %v", err)
	}
	return msg
}

// audioMessage is a model turn carrying one chunk of speech.
func audioMessage(pcm []byte) string {
	return `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":"` +
		base64.StdEncoding.EncodeToString(pcm) + `"}}]}}}`
}

// TestHandleInterruptionEmitsNoUserSpeech covers the barge-in signal, which is
// the only turn-related thing this API reports. The bot gives the floor up and
// the pipeline is interrupted, but no user-speaking frame is invented.
//
// The API reports an interruption and never a turn starting or ending, so a
// start emitted here would never be matched by a stop, and everything keyed off
// those frames would be left believing the user is still speaking. The reference
// implementation makes the same choice for the same reason.
func TestHandleInterruptionEmitsNoUserSpeech(t *testing.T) {
	s, sink := newTestService(t)

	ctx := context.Background()
	// The bot is speaking when the barge-in lands.
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2, 3, 4})))
	s.handle(ctx, decode(t, `{"serverContent":{"interrupted":true}}`))

	got := sink.names()
	want := []string{
		"BotStartedSpeakingFrame",
		"TTSAudioRawFrame",
		"BotStoppedSpeakingFrame",
		"InterruptionFrame",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
	}
	for _, n := range got {
		if n == "UserStartedSpeakingFrame" || n == "UserStoppedSpeakingFrame" {
			t.Errorf("the service invented a %s: this API reports no turn boundaries", n)
		}
	}
}

// TestHandlePartEmitsAudioAndText covers a model turn carrying both: the text is
// forwarded as the model's own words, and the audio opens the bot's turn.
func TestHandlePartEmitsAudioAndText(t *testing.T) {
	s, sink := newTestService(t)

	s.handle(context.Background(), decode(t, `{"serverContent":{"modelTurn":{"parts":[
		{"text":"one moment"},
		{"inlineData":{"mimeType":"audio/pcm","data":"`+base64.StdEncoding.EncodeToString([]byte{0, 1, 0, 1})+`"}}
	]}}}`))

	got := sink.names()
	want := []string{"LLMTextFrame", "BotStartedSpeakingFrame", "TTSAudioRawFrame"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
	}
}

// TestHandlePartSkipsUnreadableAudio covers a part whose audio cannot be
// decoded, or holds nothing. Neither is speech, so neither opens the bot's turn:
// a turn opened here would never be closed by audio that never arrives.
func TestHandlePartSkipsUnreadableAudio(t *testing.T) {
	s, sink := newTestService(t)

	ctx := context.Background()
	s.handle(ctx, decode(t, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"data":"not base64!!"}}]}}}`))
	s.handle(ctx, decode(t, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"data":""}}]}}}`))

	if got := sink.names(); len(got) != 0 {
		t.Errorf("frames = %v, want none: nothing playable arrived", got)
	}
}

// TestHandleTranscriptionsGoBothWays covers the two transcripts the service
// receives. What the user said becomes a transcription; what the model said
// becomes the model's own text, since it is the reply rather than the request.
func TestHandleTranscriptionsGoBothWays(t *testing.T) {
	s, sink := newTestService(t)

	s.handle(context.Background(), decode(t, `{"serverContent":{
		"inputTranscription":{"text":"what is the weather"},
		"outputTranscription":{"text":"it is raining"}}}`))

	got := sink.names()
	want := []string{"TranscriptionFrame", "LLMTextFrame"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
	}
}

// TestHandleIgnoresEmptyTranscriptions covers the empty transcript the service
// sends while it is still listening. It carries nothing, so forwarding it would
// put an empty user message into the conversation.
func TestHandleIgnoresEmptyTranscriptions(t *testing.T) {
	s, sink := newTestService(t)

	s.handle(context.Background(), decode(t, `{"serverContent":{
		"inputTranscription":{"text":""},
		"outputTranscription":{"text":""}}}`))

	if got := sink.names(); len(got) != 0 {
		t.Errorf("frames = %v, want none: both transcripts were empty", got)
	}
}

// TestHandleGenerationCompleteClosesTheBotTurn covers the end of the reply: the
// bot gives the floor up once the model has finished generating, and does so
// once however many times the signal arrives.
func TestHandleGenerationCompleteClosesTheBotTurn(t *testing.T) {
	s, sink := newTestService(t)

	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true}}`))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true}}`))

	got := sink.names()
	want := []string{"BotStartedSpeakingFrame", "TTSAudioRawFrame", "BotStoppedSpeakingFrame"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v: the floor was given up more than once", got, want)
	}
}

// TestHandleSetupCompleteMarksReady covers the handshake: nothing may be sent to
// the model until it has acknowledged the session setup.
func TestHandleSetupCompleteMarksReady(t *testing.T) {
	s, _ := newTestService(t)

	if s.ready.Load() {
		t.Fatal("the service reported ready before the session was acknowledged")
	}
	s.handle(context.Background(), decode(t, `{"setupComplete":{}}`))
	if !s.ready.Load() {
		t.Error("the service did not report ready after the session was acknowledged")
	}
}
