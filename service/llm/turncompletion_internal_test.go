package llm

import (
	"context"
	"slices"
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
	// A queued frame is delivered by the processor's own loop, so it is waited
	// for rather than assumed to have arrived by the time the queue call
	// returns.
	waitUntil(t, "the StartFrame never reached the end of the chain",
		func() bool { return len(down.frames()) > 0 })
	down.reset()
	up.reset()

	svc.SetFilterIncompleteUserTurns(true)
	return svc, up, down
}

// waitUntil blocks until cond holds, failing with msg if it never does.
func waitUntil(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal(msg)
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

	instructions := cfg.CompletionInstructions()
	for _, want := range []string{
		"Write A, a space, then your full reply",
		"Respond with only B",
		"Respond with only C",
	} {
		if !strings.Contains(instructions, want) {
			t.Errorf("the instructions do not say %q", want)
		}
	}
	for _, def := range []string{MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong} {
		if strings.Contains(instructions, def) {
			t.Errorf("the instructions still teach the default marker %q", def)
		}
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

// The report the service makes of a whole response is a diagnostic, for a
// consumer checking how well the model followed the protocol it was given. It
// carries the marker, what it meant, and the text before anything was held back.
func TestMarkerResponseReportsWhatTheResponseCarried(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Hello there!"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	report, ok := svc.markerResponse()
	if !ok {
		t.Fatal("gating is on, so there is a protocol to report against")
	}
	if report.Raw != MarkerComplete+" Hello there!" {
		t.Errorf("raw = %q, want the response as the model produced it", report.Raw)
	}
	if report.Marker != MarkerComplete || report.Kind != "complete" {
		t.Errorf("marker = %q kind = %q, want the complete marker", report.Marker, report.Kind)
	}
	want := []string{MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong}
	if !slices.Equal(report.Markers, want) {
		t.Errorf("markers = %v, want every marker the protocol uses (%v)", report.Markers, want)
	}
}

// An incomplete turn produces no speech, so the marker is the whole of what the
// response carried, and the report says which of the two reasons it was.
func TestMarkerResponseReportsAHeldTurn(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), MarkerIncompleteLong); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	report, _ := svc.markerResponse()
	if report.Marker != MarkerIncompleteLong || report.Kind != "long" {
		t.Errorf("marker = %q kind = %q, want the long marker", report.Marker, report.Kind)
	}
	if report.Raw != MarkerIncompleteLong {
		t.Errorf("raw = %q, want the marker alone", report.Raw)
	}
}

// A response carrying no marker at all is reported as such, which is what tells
// a model that ignored the protocol from one that followed it.
func TestMarkerResponseReportsNoMarker(t *testing.T) {
	svc, _, _ := gatedService(t)

	if err := svc.pushTurnText(t.Context(), "I forgot the protocol"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}
	report, _ := svc.markerResponse()
	if report.Marker != "" || report.Kind != "" {
		t.Errorf("marker = %q kind = %q, want neither", report.Marker, report.Kind)
	}
	if report.Raw != "I forgot the protocol" {
		t.Errorf("raw = %q, want what the model produced", report.Raw)
	}
}

// With gating off there is no protocol to report against, so nothing is
// reported: the markers are the gating's own convention with the model.
func TestNoMarkerResponseWithoutGating(t *testing.T) {
	svc := New("LLM", &markerGenerator{})
	if _, ok := svc.markerResponse(); ok {
		t.Error("a service that reads no markers should report none")
	}
}

// markerGenerator stands in for a model in a service built without gating.
type markerGenerator struct{}

func (*markerGenerator) Generate(context.Context, *frames.LLMContext, Emit) error { return nil }

func TestTextAfterACompleteMarkerStreamsStraightThrough(t *testing.T) {
	svc, _, down := gatedService(t)

	for _, chunk := range []string{MarkerComplete + " Hello", " there!"} {
		if err := svc.pushTurnText(t.Context(), chunk); err != nil {
			t.Fatalf("pushTurnText: %v", err)
		}
	}

	if got := textsOf(down); !slices.Equal(got, []string{"Hello", " there!"}) {
		t.Errorf("pushed text = %v, want [Hello  there!]", got)
	}
}

// The turn-completion judgment is pushed upstream as a UserTurnCompletionFrame.

// reportsIn returns every turn-completion report captured.
func reportsIn(c *capture) []*frames.UserTurnCompletionFrame {
	var out []*frames.UserTurnCompletionFrame
	for _, f := range c.frames() {
		if r, ok := f.(*frames.UserTurnCompletionFrame); ok {
			out = append(out, r)
		}
	}
	return out
}

func TestEachMarkerIsReportedUpstream(t *testing.T) {
	for _, tt := range []struct{ marker, kind string }{
		{MarkerComplete, "complete"},
		{MarkerIncompleteShort, "short"},
		{MarkerIncompleteLong, "long"},
	} {
		t.Run(tt.kind, func(t *testing.T) {
			svc, up, down := gatedService(t)

			if err := svc.pushTurnText(t.Context(), tt.marker+" Hello"); err != nil {
				t.Fatalf("pushTurnText: %v", err)
			}

			reports := reportsIn(up)
			if len(reports) != 1 || reports[0].Kind != tt.kind || reports[0].Marker != tt.marker {
				t.Fatalf("reports upstream = %v, want one %q", reports, tt.kind)
			}
			if got := reportsIn(down); len(got) != 0 {
				t.Errorf("reports downstream = %v, want none", got)
			}
		})
	}
}

func TestReportsCarryTheContextFrameID(t *testing.T) {
	svc, up, _ := gatedService(t)
	contextFrame := frames.NewLLMContextFrame(frames.NewLLMContext(""))
	svc.handleTurnCompletionProcessFrame(t.Context(), contextFrame)

	if err := svc.pushTurnText(t.Context(), MarkerComplete+" Hello"); err != nil {
		t.Fatalf("pushTurnText: %v", err)
	}

	reports := reportsIn(up)
	if len(reports) != 1 || reports[0].ContextFrameID != contextFrame.ID() {
		t.Fatalf("reports = %v, want one naming frame %d", reports, contextFrame.ID())
	}
}

func TestAToolCallIsReportedAsComplete(t *testing.T) {
	svc, up, down := gatedService(t)

	if err := svc.PushFrame(t.Context(), frames.NewFunctionCallsStartedFrame(nil), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	reports := reportsIn(up)
	if len(reports) != 1 || reports[0].Kind != "complete" || reports[0].Marker != "" {
		t.Fatalf("reports = %v, want one complete with no marker", reports)
	}
	if n := completionsIn(up, down); n != 2 {
		t.Errorf("completion broadcasts = %d, want 2", n)
	}
}

func TestAResponseWithoutAMarkerIsReportedWithNoKind(t *testing.T) {
	svc, up, _ := gatedService(t)

	ctx := t.Context()
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = svc.pushTurnText(ctx, "Hello there")
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	reports := reportsIn(up)
	if len(reports) != 1 || reports[0].Kind != "" || reports[0].Marker != "" {
		t.Fatalf("reports = %v, want one with no kind", reports)
	}
}

func TestEachInferenceIsReportedOnce(t *testing.T) {
	svc, up, _ := gatedService(t)
	ctx := t.Context()

	svc.handleTurnCompletionProcessFrame(ctx, frames.NewLLMContextFrame(frames.NewLLMContext("")))
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = svc.pushTurnText(ctx, MarkerComplete+" Let me check")
	_ = svc.PushFrame(ctx, frames.NewFunctionCallsStartedFrame(nil), processor.Downstream)
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	// The next inference is reported afresh.
	svc.handleTurnCompletionProcessFrame(ctx, frames.NewLLMContextFrame(frames.NewLLMContext("")))
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = svc.pushTurnText(ctx, MarkerIncompleteShort)
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	var kinds []string
	for _, r := range reportsIn(up) {
		kinds = append(kinds, r.Kind)
	}
	if !slices.Equal(kinds, []string{"complete", "short"}) {
		t.Fatalf("kinds = %v, want [complete short]", kinds)
	}
}

func TestNothingIsReportedWhileFilteringIsOff(t *testing.T) {
	svc, up, _ := gatedService(t)
	svc.SetFilterIncompleteUserTurns(false)
	ctx := t.Context()

	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = svc.PushFrame(ctx, frames.NewFunctionCallsStartedFrame(nil), processor.Downstream)
	_ = svc.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	if got := reportsIn(up); len(got) != 0 {
		t.Fatalf("reports = %v, want none", got)
	}
}
