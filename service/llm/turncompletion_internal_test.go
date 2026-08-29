package llm

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// Ported from upstream's turn-completion suite. Upstream drives the mixin by
// mocking push_frame on the object; jargo links capturing processors on both
// sides instead, so the frames are observed where they actually arrive and the
// completion broadcast is seen in both directions as upstream asserts.

// capture records every frame that reaches it.
type capture struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

func newCapture(name string) *capture {
	c := &capture{}
	c.Base = processor.New(name, c)
	return c
}

func (c *capture) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	c.mu.Lock()
	c.got = append(c.got, f)
	c.mu.Unlock()
	return c.PushFrame(ctx, f, dir)
}

// reset drops what has been captured so far.
func (c *capture) reset() {
	c.mu.Lock()
	c.got = nil
	c.mu.Unlock()
}

// frames returns what has been captured, once the chain has settled. The
// processors carry frames on goroutines of their own, so a read taken the
// instant after a push would race what is still in flight.
func (c *capture) frames() []frames.Frame {
	last := -1
	for range 100 {
		c.mu.Lock()
		n := len(c.got)
		c.mu.Unlock()
		if n == last {
			break
		}
		last = n
		time.Sleep(5 * time.Millisecond)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.got...)
}

// gatedService builds a service with turn-completion gating on, wrapped by a
// capturing processor on each side.
func gatedService(t *testing.T) (*Base, *capture, *capture) {
	t.Helper()
	svc := New("FakeLLM", generatorFunc(func(context.Context, *frames.LLMContext, Emit) error { return nil }))
	up, down := newCapture("Upstream"), newCapture("Downstream")
	up.Link(svc)
	svc.Link(down)
	for _, p := range []processor.Processor{up, svc, down} {
		if err := p.Setup(t.Context(), processor.Setup{}); err != nil {
			t.Fatalf("Setup: %v", err)
		}
	}
	t.Cleanup(func() {
		for _, p := range []processor.Processor{up, svc, down} {
			_ = p.Cleanup(context.Background())
		}
	})
	// A processor drains nothing until its StartFrame arrives, so one is queued
	// at the head of the chain the way a running pipeline would, and travels
	// from there to the rest.
	if err := up.QueueFrame(t.Context(), frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("StartFrame: %v", err)
	}
	if len(down.frames()) == 0 {
		t.Fatal("the StartFrame never reached the end of the chain")
	}
	down.reset()
	up.reset()

	svc.SetFilterIncompleteUserTurns(true)
	return svc, up, down
}

// generatorFunc adapts a function to Generator.
type generatorFunc func(context.Context, *frames.LLMContext, Emit) error

func (g generatorFunc) Generate(ctx context.Context, c *frames.LLMContext, e Emit) error {
	return g(ctx, c, e)
}

// textsOf returns the text of every LLMTextFrame captured, in order.
func textsOf(c *capture) []string {
	var out []string
	for _, f := range c.frames() {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// markersOf returns every marker frame captured.
func markersOf(c *capture) []*frames.LLMMarkerFrame {
	var out []*frames.LLMMarkerFrame
	for _, f := range c.frames() {
		if mf, ok := f.(*frames.LLMMarkerFrame); ok {
			out = append(out, mf)
		}
	}
	return out
}

// completionsIn counts the completion reports captured across both sides.
func completionsIn(caps ...*capture) int {
	n := 0
	for _, c := range caps {
		for _, f := range c.frames() {
			if _, ok := f.(*frames.UserTurnInferenceCompletedFrame); ok {
				n++
			}
		}
	}
	return n
}

func TestCompleteMarkerPushesText(t *testing.T) {
	svc, up, down := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Hello there!"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := textsOf(down); len(got) != 1 || got[0] != "Hello there!" {
		t.Errorf("pushed text = %v, want [Hello there!]", got)
	}
	m := markersOf(down)
	if len(m) != 1 || m[0].Marker != MarkerComplete {
		t.Fatalf("pushed markers = %v, want one complete marker", m)
	}
	if m[0].AppendToContextImmediately {
		t.Error("the complete marker should join the reply, not stand alone in the conversation")
	}
	// The completion is broadcast, so it is seen on both sides.
	if n := completionsIn(up, down); n != 2 {
		t.Errorf("completion reports = %d, want 2 (one each way)", n)
	}
}

func TestIncompleteShortMarkerSuppressesText(t *testing.T) {
	svc, up, down := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteShort); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := textsOf(down); len(got) != 0 {
		t.Errorf("pushed text = %v, want none: the reply is suppressed", got)
	}
	m := markersOf(down)
	if len(m) != 1 || m[0].Marker != MarkerIncompleteShort {
		t.Fatalf("pushed markers = %v, want one short marker", m)
	}
	if !m[0].AppendToContextImmediately {
		t.Error("an incomplete marker is the whole turn, so it stands alone in the conversation")
	}
	if n := completionsIn(up, down); n != 0 {
		t.Errorf("completion reports = %d, want 0: the turn is explicitly not complete", n)
	}
}

func TestIncompleteLongMarkerSuppressesText(t *testing.T) {
	svc, up, down := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteLong); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := textsOf(down); len(got) != 0 {
		t.Errorf("pushed text = %v, want none", got)
	}
	m := markersOf(down)
	if len(m) != 1 || m[0].Marker != MarkerIncompleteLong {
		t.Fatalf("pushed markers = %v, want one long marker", m)
	}
	if !m[0].AppendToContextImmediately {
		t.Error("an incomplete marker should stand alone in the conversation")
	}
	if n := completionsIn(up, down); n != 0 {
		t.Errorf("completion reports = %d, want 0", n)
	}
}

func TestTextBufferedUntilMarkerFound(t *testing.T) {
	svc, _, down := gatedService(t)

	for _, chunk := range []string{"Hello", " there"} {
		if err := svc.pushTurnText(t.Context(), chunk); err != nil {
			t.Fatalf("pushTurnText: %v", err)
		}
	}
	if got := len(down.frames()); got != 0 {
		t.Fatalf("%d frames were pushed before any marker, want none while buffering", got)
	}

	if err := svc.pushTurnText(t.Context(), " "+MarkerComplete+" How are you?"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := len(textsOf(down)); got != 1 {
		t.Errorf("text frames = %d, want 1 for the spoken part", got)
	}
	if got := len(markersOf(down)); got != 1 {
		t.Errorf("marker frames = %d, want 1", got)
	}
}

func TestTurnStateResetAfterResponseEnd(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Hello!"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	svc.turnCompletion.mu.Lock()
	marker := svc.turnCompletion.marker
	svc.turnCompletion.mu.Unlock()
	if marker != TurnMarkerComplete {
		t.Fatalf("marker = %v, want complete", marker)
	}

	if err := svc.PushFrame(t.Context(), frames.NewLLMFullResponseEndFrame(), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	svc.turnCompletion.mu.Lock()
	defer svc.turnCompletion.mu.Unlock()
	if svc.turnCompletion.marker != turnMarkerNone {
		t.Errorf("marker = %v, want it cleared for the next response", svc.turnCompletion.marker)
	}
	if svc.turnCompletion.buffer != "" {
		t.Errorf("buffer = %q, want it cleared", svc.turnCompletion.buffer)
	}
}

// A response starting disarms a pending re-prompt. It is the single point that
// settles the race between the timeout firing and a completion arriving.
func TestNewResponseCancelsPendingIncompleteTimeout(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteShort); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	svc.turnCompletion.mu.Lock()
	armed := svc.turnCompletion.cancelTimeout != nil
	svc.turnCompletion.mu.Unlock()
	if !armed {
		t.Fatal("the incomplete marker did not arm a re-prompt")
	}

	if err := svc.PushFrame(t.Context(), frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	svc.turnCompletion.mu.Lock()
	defer svc.turnCompletion.mu.Unlock()
	if svc.turnCompletion.cancelTimeout != nil {
		t.Error("a new response did not disarm the pending re-prompt")
	}
}

// A resume inside an open turn produces no interruption, so the re-prompt has
// to be disarmed here or it talks over the user.
func TestVADResumeCancelsPendingIncompleteTimeout(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteShort); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))

	svc.turnCompletion.mu.Lock()
	defer svc.turnCompletion.mu.Unlock()
	if svc.turnCompletion.cancelTimeout != nil {
		t.Error("a resumed speech did not disarm the pending re-prompt")
	}
}

// A turn detector can trigger several inferences per turn, each with its own
// verdict; only the first is spoken.
func TestOnlyFirstCompletionVoicedPerUserTurn(t *testing.T) {
	svc, up, down := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" How are you?"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	svc.turnReset(t.Context())

	if got := textsOf(down); len(got) != 1 || got[0] != "How are you?" {
		t.Fatalf("first inference pushed %v, want [How are you?]", got)
	}
	before := completionsIn(up, down)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" How are you?"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := textsOf(down); len(got) != 1 {
		t.Errorf("a second inference in the same turn was voiced: %v", got)
	}
	if completionsIn(up, down) != before {
		t.Error("a second inference in the same turn reported completion again")
	}
}

// The response that set the latch keeps streaming its own continuation.
func TestVoicedResponseKeepsStreamingAfterLatch(t *testing.T) {
	svc, _, down := gatedService(t)

	for _, chunk := range []string{MarkerComplete + " Hello", " there!"} {
		if err := svc.pushTurnText(t.Context(), chunk); err != nil {
			t.Fatalf("pushTurnText: %v", err)
		}
	}

	want := []string{"Hello", " there!"}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
}

func TestNewUserTurnResetsCompletionLatch(t *testing.T) {
	svc, _, _ := gatedService(t)
	svc.turnCompletion.mu.Lock()
	svc.turnCompletion.voiced = true
	svc.turnCompletion.mu.Unlock()

	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewUserStartedSpeakingFrame())

	if voicedOf(svc) {
		t.Error("a new user turn did not clear the completion latch")
	}
}

func TestVADResumeResetsCompletionLatch(t *testing.T) {
	svc, _, _ := gatedService(t)
	svc.turnCompletion.mu.Lock()
	svc.turnCompletion.voiced = true
	svc.turnCompletion.mu.Unlock()

	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))

	if voicedOf(svc) {
		t.Error("a resumed speech did not clear the completion latch")
	}
}

// A completion voiced mid-turn, then the user resumes inside the same open
// turn, then a second legitimate completion arrives. Without clearing the latch
// on the resume the second response is dropped and the bot never replies.
func TestResumedSpeechDoesNotPermanentlySilenceTheTurn(t *testing.T) {
	svc, _, down := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" First answer"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	svc.turnReset(t.Context())
	if !voicedOf(svc) {
		t.Fatal("the first completion did not set the latch")
	}

	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	// The resumed speech ends before the second inference answers. A completion
	// arriving while the user is still speaking is stale and suppressed, which
	// TestACompletionWhileTheUserSpeaksIsStale covers.
	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Second answer"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	svc.turnReset(t.Context())

	want := []string{"First answer", "Second answer"}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
}

func TestFunctionCallResetsCompletionLatch(t *testing.T) {
	svc, _, _ := gatedService(t)
	svc.turnCompletion.mu.Lock()
	svc.turnCompletion.voiced = true
	svc.turnCompletion.mu.Unlock()

	if err := svc.PushFrame(t.Context(),
		frames.NewFunctionCallsStartedFrame(nil), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	if voicedOf(svc) {
		t.Error("a tool call did not clear the completion latch")
	}
}

// A spoken acknowledgement, then a tool call, then the real answer. Without
// clearing the latch on the tool-call path the answer is dropped and the bot
// goes quiet despite the tool succeeding.
func TestPostToolResponseSpeaksAfterVoicedFiller(t *testing.T) {
	svc, _, down := gatedService(t)
	ctx := t.Context()

	if err := svc.pushTurnText(ctx, MarkerComplete+" One moment."); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	if !voicedOf(svc) {
		t.Fatal("the acknowledgement did not set the latch")
	}

	if err := svc.PushFrame(ctx, frames.NewFunctionCallsStartedFrame(nil), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}
	if err := svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	if err := svc.pushTurnText(ctx, MarkerComplete+" Here are the openings."); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	if err := svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	want := []string{"One moment.", "Here are the openings."}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
}

// A tool call grants exactly one extra spoken completion, then re-latches, so
// the reset cannot widen into an open-ended window.
func TestToolCallAllowsOnlyOneExtraCompletion(t *testing.T) {
	svc, _, down := gatedService(t)
	ctx := t.Context()

	_ = svc.pushTurnText(ctx, MarkerComplete+" One moment.")
	_ = svc.PushFrame(ctx, frames.NewFunctionCallsStartedFrame(nil), processor.Downstream)
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	_ = svc.pushTurnText(ctx, MarkerComplete+" Here are the openings.")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	// A stray inference in the same turn, with no new tool call.
	_ = svc.pushTurnText(ctx, MarkerComplete+" Duplicate!")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	want := []string{"One moment.", "Here are the openings."}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
	if !voicedOf(svc) {
		t.Error("the latch should be armed again after the extra completion")
	}
}

// An application-driven re-prompt fires while the user is silent, so no speech
// frame clears the latch for it.
func TestRequestedRunResetsCompletionLatch(t *testing.T) {
	svc, _, _ := gatedService(t)
	svc.turnCompletion.mu.Lock()
	svc.turnCompletion.voiced = true
	svc.turnCompletion.mu.Unlock()

	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleDeveloper, Text: "The user has been quiet."},
	})
	f.RunLLM = true
	svc.handleTurnCompletionProcessFrame(t.Context(), f)

	if voicedOf(svc) {
		t.Error("a requested run did not clear the completion latch")
	}
}

// Appending without asking for a run leaves the latch armed: nothing asked for
// fresh speech.
func TestAppendWithoutRunKeepsCompletionLatch(t *testing.T) {
	svc, _, _ := gatedService(t)
	svc.turnCompletion.mu.Lock()
	svc.turnCompletion.voiced = true
	svc.turnCompletion.mu.Unlock()

	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleDeveloper, Text: "Background note."},
	})
	svc.handleTurnCompletionProcessFrame(t.Context(), f)

	if !voicedOf(svc) {
		t.Error("an append with no run cleared the latch, want it left armed")
	}
}

// A completion is voiced, the user stays silent, an idle handler asks for a
// run: the response it draws must be spoken.
func TestRepromptSpeaksAfterVoicedCompletion(t *testing.T) {
	svc, _, down := gatedService(t)
	ctx := t.Context()

	_ = svc.pushTurnText(ctx, MarkerComplete+" Japan is a great pick.")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	if !voicedOf(svc) {
		t.Fatal("the first reply did not set the latch")
	}

	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleDeveloper, Text: "The user has been quiet."},
	})
	f.RunLLM = true
	svc.handleTurnCompletionProcessFrame(ctx, f)

	_ = svc.pushTurnText(ctx, MarkerComplete+" Are you still there?")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	want := []string{"Japan is a great pick.", "Are you still there?"}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
}

func TestRequestedRunAllowsOnlyOneExtraCompletion(t *testing.T) {
	svc, _, down := gatedService(t)
	ctx := t.Context()

	_ = svc.pushTurnText(ctx, MarkerComplete+" Japan is a great pick.")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleDeveloper, Text: "The user has been quiet."},
	})
	f.RunLLM = true
	svc.handleTurnCompletionProcessFrame(ctx, f)

	_ = svc.pushTurnText(ctx, MarkerComplete+" Are you still there?")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	// A stray inference in the same turn, with no run requested.
	_ = svc.pushTurnText(ctx, MarkerComplete+" Duplicate!")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	want := []string{"Japan is a great pick.", "Are you still there?"}
	if got := textsOf(down); !equalStrings(got, want) {
		t.Errorf("pushed text = %v, want %v", got, want)
	}
	if !voicedOf(svc) {
		t.Error("the latch should be armed again after the extra completion")
	}
}

// A response that carried no marker at all has its text pushed rather than
// dropped, so a model that ignored the protocol still says something.
func TestUnmarkedResponseIsPushedAnyway(t *testing.T) {
	svc, _, down := gatedService(t)

	_ = svc.pushTurnText(t.Context(), "no marker here")
	if got := len(down.frames()); got != 0 {
		t.Fatalf("%d frames pushed while buffering, want none", got)
	}

	svc.turnReset(t.Context())

	if got := textsOf(down); len(got) != 1 || got[0] != "no marker here" {
		t.Errorf("pushed text = %v, want the unmarked response pushed anyway", got)
	}
}

// Text arriving after an incomplete verdict stays suppressed, in case the model
// disobeys the protocol and keeps talking past the marker.
func TestTextAfterIncompleteMarkerStaysSuppressed(t *testing.T) {
	svc, _, down := gatedService(t)

	_ = svc.pushTurnText(t.Context(), MarkerIncompleteShort)
	_ = svc.pushTurnText(t.Context(), " Please continue...")

	if got := textsOf(down); len(got) != 0 {
		t.Errorf("pushed text = %v, want none after an incomplete verdict", got)
	}
}

// The instructions are composed onto the prompt exactly while gating is on.
func TestSystemInstructionComposition(t *testing.T) {
	svc := New("FakeLLM", generatorFunc(func(context.Context, *frames.LLMContext, Emit) error { return nil }))

	svc.SetSystemInstruction("You are a helpful assistant.")
	if got := svc.SystemInstruction(); got != "You are a helpful assistant." {
		t.Fatalf("SystemInstruction() = %q before gating", got)
	}

	svc.SetFilterIncompleteUserTurns(true)
	want := "You are a helpful assistant.\n\n" + UserTurnCompletionInstructions
	if got := svc.SystemInstruction(); got != want {
		t.Errorf("the protocol was not composed onto the prompt when gating was enabled")
	}

	// Changing the prompt while gating is on recomposes from the new base.
	svc.SetSystemInstruction("New prompt.")
	want = "New prompt.\n\n" + UserTurnCompletionInstructions
	if got := svc.SystemInstruction(); got != want {
		t.Errorf("changing the prompt did not recompose with the protocol")
	}

	// A configuration carrying its own instructions replaces the default.
	svc.SetUserTurnCompletionConfig(UserTurnCompletionConfig{Instructions: "Custom turn instructions."})
	if got := svc.SystemInstruction(); got != "New prompt.\n\nCustom turn instructions." {
		t.Errorf("SystemInstruction() = %q, want the custom instructions composed on", got)
	}

	// Disabling takes the protocol back out and restores the prompt.
	svc.SetFilterIncompleteUserTurns(false)
	if got := svc.SystemInstruction(); got != "New prompt." {
		t.Errorf("SystemInstruction() = %q, want the base prompt restored", got)
	}
}

// With no base prompt, enabling gating leaves the protocol alone as the whole
// instruction, and disabling leaves nothing.
func TestSystemInstructionCompositionWithNoBasePrompt(t *testing.T) {
	svc := New("FakeLLM", generatorFunc(func(context.Context, *frames.LLMContext, Emit) error { return nil }))

	svc.SetFilterIncompleteUserTurns(true)
	if got := svc.SystemInstruction(); got != UserTurnCompletionInstructions {
		t.Error("the protocol alone should be the instruction when there is no base prompt")
	}
	svc.SetFilterIncompleteUserTurns(false)
	if got := svc.SystemInstruction(); got != "" {
		t.Errorf("SystemInstruction() = %q, want empty again", got)
	}
}

// The re-prompt delivery has no upstream test, since upstream mocks the timeout
// out; this one is ours. It is the path the whole incomplete branch exists for,
// so it is worth covering end to end.
func TestIncompleteTimeoutRepromptsTheModel(t *testing.T) {
	svc, _, down := gatedService(t)
	svc.SetUserTurnCompletionConfig(UserTurnCompletionConfig{
		IncompleteShortTimeout: 20 * time.Millisecond,
		IncompleteShortPrompt:  "Go on.",
	})

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteShort); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	var appended *frames.LLMMessagesAppendFrame
	sawRun := false
	for range 100 {
		for _, f := range down.frames() {
			switch v := f.(type) {
			case *frames.LLMMessagesAppendFrame:
				appended = v
			case *frames.LLMRunFrame:
				sawRun = true
			}
		}
		if appended != nil && sawRun {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if appended == nil {
		t.Fatal("the re-prompt never reached the pipeline")
	}
	if len(appended.Messages) != 1 || appended.Messages[0].Text != "Go on." {
		t.Errorf("re-prompt messages = %+v, want the configured prompt", appended.Messages)
	}
	if appended.Messages[0].Role != frames.RoleDeveloper {
		t.Errorf("re-prompt role = %q, want %q", appended.Messages[0].Role, frames.RoleDeveloper)
	}
	if !sawRun {
		t.Error("the re-prompt did not ask for a run")
	}

	// The re-prompt resets the response state, so the answer it draws is parsed
	// as a fresh one rather than being suppressed by the incomplete verdict.
	svc.turnCompletion.mu.Lock()
	defer svc.turnCompletion.mu.Unlock()
	if svc.turnCompletion.marker != turnMarkerNone {
		t.Errorf("marker = %v, want it cleared before the re-prompt ran", svc.turnCompletion.marker)
	}
}

// The strategy that drives the protocol configures the service over a settings
// update once the pipeline is running, so both halves of that update have to
// land: the enable flag and the configuration it runs under.
func TestSettingsUpdateEnablesGatingAndCarriesItsConfig(t *testing.T) {
	svc := New("FakeLLM", generatorFunc(func(context.Context, *frames.LLMContext, Emit) error { return nil }))
	svc.SetSystemInstruction("Base prompt.")

	var delta settings.LLM
	delta.FilterIncompleteUserTurns = settings.Set(true)
	delta.UserTurnCompletionConfig = settings.Set[any](UserTurnCompletionConfig{
		Instructions:           "Custom turn instructions.",
		IncompleteShortTimeout: 3 * time.Second,
	})
	svc.updateSettings(t.Context(), frames.NewLLMUpdateSettingsFrame(&delta))

	if !svc.FilterIncompleteUserTurns() {
		t.Error("the update did not enable the gating")
	}
	if got := svc.UserTurnCompletionConfig().IncompleteShortTimeout; got != 3*time.Second {
		t.Errorf("IncompleteShortTimeout = %v, want the configured 3s", got)
	}
	if got := svc.SystemInstruction(); got != "Base prompt.\n\nCustom turn instructions." {
		t.Errorf("SystemInstruction() = %q, want the custom protocol composed on", got)
	}
}

func voicedOf(b *Base) bool {
	b.turnCompletion.mu.Lock()
	defer b.turnCompletion.mu.Unlock()
	return b.turnCompletion.voiced
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestACompletionWhileTheUserSpeaksIsStale covers a complete verdict that
// arrives after the user has resumed. The inference was triggered when the turn
// looked over; by the time it answers the turn is not over after all, so the
// response is suppressed and the short timeout re-armed, exactly as if the model
// had reported the turn incomplete.
//
// Ported from upstream's stale-complete handling.
func TestACompletionWhileTheUserSpeaksIsStale(t *testing.T) {
	svc, _, down := gatedService(t)

	svc.handleTurnCompletionProcessFrame(t.Context(), frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Answer"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	if got := textsOf(down); len(got) != 0 {
		t.Errorf("pushed text = %v, want none: the completion was stale", got)
	}
	if got := markersOf(down); len(got) != 1 || got[0].Marker != MarkerIncompleteShort {
		t.Errorf("markers = %v, want the short incomplete one", got)
	}
}

// TestConfiguredMarkersAreTheOnesRead covers a service told to use a marker set
// of its own: it is those the response is parsed for, and those the prompts
// teach.
func TestConfiguredMarkersAreTheOnesRead(t *testing.T) {
	svc, _, down := gatedService(t)
	cfg := UserTurnCompletionConfig{
		CompleteMarker:        "A",
		IncompleteShortMarker: "B",
		IncompleteLongMarker:  "C",
	}
	svc.SetUserTurnCompletionConfig(cfg)

	if !strings.Contains(cfg.CompletionInstructions(), "A") ||
		strings.Contains(cfg.CompletionInstructions(), MarkerComplete) {
		t.Error("the instructions still teach the default markers")
	}

	if err := svc.pushTurnText(t.Context(), "A Answer"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	if got := textsOf(down); len(got) != 1 || got[0] != "Answer" {
		t.Errorf("pushed text = %v, want the answer past the configured marker", got)
	}
	if got := markersOf(down); len(got) != 1 || got[0].Marker != "A" {
		t.Errorf("markers = %v, want the configured complete marker", got)
	}
}
