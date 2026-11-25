package llm

import (
	"context"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// pushedFrame is one frame the extension let through or pushed, with its
// direction.
type pushedFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// reader runs the turn-completion extension the way its service does,
// recording what goes out: what the extension lets through or adds in
// FilterPushFrame, and what it pushes or broadcasts itself, in order.
type reader struct {
	extension *TurnCompletionExtension
	mu        sync.Mutex
	pushed    []pushedFrame
}

func newReader(cfg UserTurnCompletionConfig) *reader {
	r := &reader{extension: NewTurnCompletionExtension(cfg)}
	r.extension.extensionBase().wire(nil,
		func(_ context.Context, f frames.Frame, dir processor.Direction) error {
			r.record(f, dir)
			return nil
		},
		func(_ context.Context, build func() frames.Frame) error {
			r.record(build(), processor.Downstream)
			r.record(build(), processor.Upstream)
			return nil
		})
	return r
}

func (r *reader) record(f frames.Frame, dir processor.Direction) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pushed = append(r.pushed, pushedFrame{f, dir})
}

// push pushes a frame as the service would, through the extension's filter.
func (r *reader) push(t *testing.T, f frames.Frame) {
	t.Helper()
	out, err := r.extension.FilterPushFrame(t.Context(), f, processor.Downstream)
	if err != nil {
		t.Fatalf("FilterPushFrame: %v", err)
	}
	for _, filtered := range out {
		r.record(filtered, processor.Downstream)
	}
}

func (r *reader) text(t *testing.T, text string) {
	t.Helper()
	r.push(t, frames.NewLLMTextFrame(text))
}

func (r *reader) process(t *testing.T, f frames.Frame) {
	t.Helper()
	if err := r.extension.ProcessFrame(t.Context(), f, processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}
}

func (r *reader) frames() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]frames.Frame, 0, len(r.pushed))
	for _, p := range r.pushed {
		out = append(out, p.frame)
	}
	return out
}

// framesOf are the frames of type T the reader recorded, in order.
func framesOf[T frames.Frame](r *reader) []T {
	var out []T
	for _, f := range r.frames() {
		if typed, ok := f.(T); ok {
			out = append(out, typed)
		}
	}
	return out
}

func textsOf(r *reader) []string {
	var out []string
	for _, f := range framesOf[*frames.LLMTextFrame](r) {
		out = append(out, f.Text)
	}
	return out
}

// Reading the marker from the model's text.

func TestCompleteMarkerPushesText(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerComplete+" Hello there!")

	// The marker rides as an LLMMarkerFrame that joins the reply; only the
	// spoken text is pushed as an LLMTextFrame.
	texts := framesOf[*frames.LLMTextFrame](r)
	if len(texts) != 1 || texts[0].Text != "Hello there!" {
		t.Fatalf("pushed text = %v, want [Hello there!]", textsOf(r))
	}
	if texts[0].SkipTTS != nil && *texts[0].SkipTTS {
		t.Error("the spoken text should not skip TTS")
	}
	markers := framesOf[*frames.LLMMarkerFrame](r)
	if len(markers) != 1 || markers[0].Marker != MarkerComplete {
		t.Fatalf("pushed markers = %v, want one complete marker", markers)
	}
	if markers[0].AppendToContextImmediately {
		t.Error("the complete marker should join the reply, not stand alone in the conversation")
	}
	// UserTurnInferenceCompletedFrame is broadcast in both directions.
	if n := len(framesOf[*frames.UserTurnInferenceCompletedFrame](r)); n != 2 {
		t.Errorf("completion broadcasts = %d, want 2", n)
	}
}

func TestIncompleteShortMarkerSuppressesText(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerIncompleteShort)

	if got := textsOf(r); len(got) != 0 {
		t.Errorf("pushed text = %v, want none: the response is suppressed", got)
	}
	markers := framesOf[*frames.LLMMarkerFrame](r)
	if len(markers) != 1 || markers[0].Marker != MarkerIncompleteShort {
		t.Fatalf("pushed markers = %v, want one short marker", markers)
	}
	if !markers[0].AppendToContextImmediately {
		t.Error("an incomplete marker is the whole turn, so it stands alone in the conversation")
	}
	// Incomplete markers do not emit UserTurnInferenceCompletedFrame.
	if n := len(framesOf[*frames.UserTurnInferenceCompletedFrame](r)); n != 0 {
		t.Errorf("completion broadcasts = %d, want none", n)
	}
}

func TestIncompleteLongMarkerSuppressesText(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerIncompleteLong)

	if got := textsOf(r); len(got) != 0 {
		t.Errorf("pushed text = %v, want none", got)
	}
	markers := framesOf[*frames.LLMMarkerFrame](r)
	if len(markers) != 1 || markers[0].Marker != MarkerIncompleteLong {
		t.Fatalf("pushed markers = %v, want one long marker", markers)
	}
	if !markers[0].AppendToContextImmediately {
		t.Error("an incomplete marker should stand alone in the conversation")
	}
	if n := len(framesOf[*frames.UserTurnInferenceCompletedFrame](r)); n != 0 {
		t.Errorf("completion broadcasts = %d, want none", n)
	}
}

func TestTextBufferedUntilMarkerFound(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, "Hello")
	r.text(t, " there")
	if got := r.frames(); len(got) != 0 {
		t.Fatalf("pushed %d frames before any marker, want none while buffering", len(got))
	}

	r.text(t, " "+MarkerComplete+" How are you?")

	// One LLMTextFrame for the spoken portion, one LLMMarkerFrame for the
	// marker.
	if n := len(framesOf[*frames.LLMTextFrame](r)); n != 1 {
		t.Errorf("text frames = %d, want 1", n)
	}
	if n := len(framesOf[*frames.LLMMarkerFrame](r)); n != 1 {
		t.Errorf("marker frames = %d, want 1", n)
	}
}

func TestTurnStateResetAfterLLMFullResponseEndFrame(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerComplete+" Hello!")
	r.extension.mu.Lock()
	marker := r.extension.marker
	r.extension.mu.Unlock()
	if marker != TurnMarkerComplete {
		t.Fatalf("marker = %v, want complete", marker)
	}

	r.push(t, frames.NewLLMFullResponseEndFrame())

	// The marker is cleared, ready for the next response.
	r.extension.mu.Lock()
	defer r.extension.mu.Unlock()
	if r.extension.marker != turnMarkerNone {
		t.Errorf("marker = %v, want it cleared", r.extension.marker)
	}
	if r.extension.buffer != "" {
		t.Errorf("buffer = %q, want it cleared", r.extension.buffer)
	}
}

func TestResponseEndReportsTheRawTextAndMarker(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, "Well, ")
	r.text(t, MarkerIncompleteShort+" go on")
	r.text(t, " please")
	r.push(t, frames.NewLLMFullResponseEndFrame())

	reports := framesOf[*frames.LLMMarkerResponseFrame](r)
	if len(reports) != 1 {
		t.Fatalf("reports = %d, want 1", len(reports))
	}
	report := reports[0]
	// Everything the model produced, including what the extension held back.
	if want := "Well, " + MarkerIncompleteShort + " go on please"; report.Raw != want {
		t.Errorf("raw = %q, want %q", report.Raw, want)
	}
	if report.Marker != MarkerIncompleteShort || report.Kind != "short" {
		t.Errorf("marker = %q kind = %q, want the short marker", report.Marker, report.Kind)
	}
	if want := []string{MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong}; !slices.Equal(report.Markers, want) {
		t.Errorf("markers = %v, want %v", report.Markers, want)
	}
	// The report precedes the end-of-response frame.
	all := r.frames()
	reportAt := slices.IndexFunc(all, func(f frames.Frame) bool { return f == frames.Frame(report) })
	endAt := slices.IndexFunc(all, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMFullResponseEndFrame)
		return ok
	})
	if reportAt < 0 || endAt < 0 || reportAt > endAt {
		t.Errorf("the report is at %d and the end of the response at %d, want the report first", reportAt, endAt)
	}
}

func TestAResponseWithoutAMarkerIsReportedAsSuch(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, "Hello there")
	r.push(t, frames.NewLLMFullResponseEndFrame())

	reports := framesOf[*frames.LLMMarkerResponseFrame](r)
	if len(reports) != 1 {
		t.Fatalf("reports = %d, want 1", len(reports))
	}
	if got := reports[0]; got.Raw != "Hello there" || got.Marker != "" || got.Kind != "" {
		t.Errorf("report = (%q, %q, %q), want (Hello there, none, none)", got.Raw, got.Marker, got.Kind)
	}
}

func TestAResponseTheExtensionNeverSawIsNotReported(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.push(t, frames.NewLLMFullResponseEndFrame())

	if reports := framesOf[*frames.LLMMarkerResponseFrame](r); len(reports) != 0 {
		t.Errorf("reported %v for a response that carried nothing", reports)
	}
}

func TestTextAfterACompleteMarkerStreamsStraightThrough(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerComplete+" Hello")
	r.text(t, " there!")

	if got := textsOf(r); !slices.Equal(got, []string{"Hello", " there!"}) {
		t.Errorf("pushed text = %v, want [Hello  there!]", got)
	}
}

// Text the model wrote without a marker is not lost: it goes out at the end.
func TestUnmarkedTextIsPushedWhenTheResponseEnds(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, "Hello there")
	if got := textsOf(r); len(got) != 0 {
		t.Fatalf("pushed text = %v while buffering, want none", got)
	}

	r.push(t, frames.NewLLMFullResponseEndFrame())

	if got := textsOf(r); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("pushed text = %v, want [Hello there]", got)
	}
}

// An interruption resets the response too, and text buffered without a marker
// is pushed then rather than lost.
func TestUnmarkedTextIsPushedOnAnInterruption(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.text(t, "no marker here")

	r.process(t, frames.NewInterruptionFrame())

	if got := textsOf(r); !slices.Equal(got, []string{"no marker here"}) {
		t.Errorf("pushed text = %v, want the unmarked response pushed anyway", got)
	}
}

// Text arriving after an incomplete verdict stays suppressed, in case the model
// disobeys the protocol and keeps talking past the marker.
func TestTextAfterIncompleteMarkerStaysSuppressed(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.text(t, MarkerIncompleteShort)
	r.text(t, " Please continue...")

	if got := textsOf(r); len(got) != 0 {
		t.Errorf("pushed text = %v, want none after an incomplete verdict", got)
	}
}

// The turn-completion instructions composed into the service's system
// instruction.

// plainService builds a service with no adapter, for composing instructions.
func plainService(prompt string) *Base {
	svc := New("FakeLLM", generatorFunc(func(context.Context, *frames.LLMContext, Emit) error { return nil }))
	if prompt != "" {
		svc.SetSystemInstruction(prompt)
	}
	return svc
}

// generatorFunc adapts a function to Generator.
type generatorFunc func(context.Context, *frames.LLMContext, Emit) error

func (g generatorFunc) Generate(ctx context.Context, c *frames.LLMContext, e Emit) error {
	return g(ctx, c, e)
}

// updateLLMSettings applies a settings delta to the service as an
// LLMUpdateSettingsFrame would.
func updateLLMSettings(t *testing.T, svc *Base, delta *settings.LLM) error {
	t.Helper()
	return svc.updateSettings(t.Context(), frames.NewLLMUpdateSettingsFrame(delta))
}

// enableTurnCompletion installs the extension the way the stop strategy does:
// over a settings update, with a function that builds it.
func enableTurnCompletion(t *testing.T, svc *Base, cfg UserTurnCompletionConfig) *TurnCompletionExtension {
	t.Helper()
	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions{
		func() Extension { return NewTurnCompletionExtension(cfg) },
	})
	if err := updateLLMSettings(t, svc, &delta); err != nil {
		t.Fatalf("updateSettings: %v", err)
	}
	extension, ok := GetExtension[*TurnCompletionExtension](svc)
	if !ok {
		t.Fatal("the update did not attach the turn-completion extension")
	}
	return extension
}

func TestEnableTurnCompletionSetsSystemInstruction(t *testing.T) {
	svc := plainService("")
	if got := svc.SystemInstruction(); got != "" {
		t.Fatalf("SystemInstruction() = %q, want none", got)
	}

	enableTurnCompletion(t, svc, UserTurnCompletionConfig{})

	if got := svc.SystemInstruction(); got != UserTurnCompletionInstructions {
		t.Error("the protocol alone should be the instruction when there is no base prompt")
	}
}

func TestEnableTurnCompletionAppendsToExistingSystemInstruction(t *testing.T) {
	svc := plainService("You are a helpful assistant.")

	enableTurnCompletion(t, svc, UserTurnCompletionConfig{})

	want := "You are a helpful assistant.\n\n" + UserTurnCompletionInstructions
	if got := svc.SystemInstruction(); got != want {
		t.Error("the protocol was not composed onto the prompt")
	}
}

func TestUpdateSystemInstructionWhileTurnCompletionActive(t *testing.T) {
	svc := plainService("Original prompt.")
	enableTurnCompletion(t, svc, UserTurnCompletionConfig{})

	var delta settings.LLM
	delta.SystemInstruction = settings.Set("New prompt.")
	if err := updateLLMSettings(t, svc, &delta); err != nil {
		t.Fatal(err)
	}

	if got, want := svc.SystemInstruction(), "New prompt.\n\n"+UserTurnCompletionInstructions; got != want {
		t.Error("changing the prompt did not recompose with the protocol")
	}
}

func TestCustomInstructionsComposeInPlaceOfTheDefault(t *testing.T) {
	svc := plainService("Base prompt.")

	enableTurnCompletion(t, svc, UserTurnCompletionConfig{Instructions: "Custom turn instructions."})

	if got := svc.SystemInstruction(); got != "Base prompt.\n\nCustom turn instructions." {
		t.Errorf("SystemInstruction() = %q, want the custom instructions composed on", got)
	}
}

// Enabling turn completion and changing the system instruction in the same
// delta composes on the new prompt.
func TestSimultaneousEnableAndSystemInstructionChange(t *testing.T) {
	svc := plainService("Original prompt.")

	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions{
		func() Extension { return NewTurnCompletionExtension(UserTurnCompletionConfig{}) },
	})
	delta.SystemInstruction = settings.Set("New prompt.")
	if err := updateLLMSettings(t, svc, &delta); err != nil {
		t.Fatal(err)
	}

	if got, want := svc.SystemInstruction(), "New prompt.\n\n"+UserTurnCompletionInstructions; got != want {
		t.Errorf("SystemInstruction() = %q, want the protocol on the new prompt", got)
	}
}

// The deprecated settings fields still switch the extension on and off, with a
// warning.

// Markers are configurable, and the prompts follow whatever is configured.

func TestDefaultMarkersAreTheFillGradient(t *testing.T) {
	// Full is a finished turn, half is cut off mid-thought, empty is not
	// started.
	if MarkerComplete != "●" || MarkerIncompleteShort != "◐" || MarkerIncompleteLong != "○" {
		t.Errorf("markers = %q %q %q, want ● ◐ ○", MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong)
	}
}

func TestPromptsFollowConfiguredMarkers(t *testing.T) {
	cfg := UserTurnCompletionConfig{CompleteMarker: "Y", IncompleteShortMarker: "N", IncompleteLongMarker: "W"}

	for _, prompt := range []string{cfg.CompletionInstructions(), cfg.ShortPrompt(), cfg.LongPrompt()} {
		for _, def := range []string{MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong} {
			if strings.Contains(prompt, def) {
				t.Errorf("a prompt still teaches the default marker %q", def)
			}
		}
	}
	for _, want := range []string{"Write Y, a space, then your full reply", "Respond with only N", "Respond with only W"} {
		if !strings.Contains(cfg.CompletionInstructions(), want) {
			t.Errorf("the instructions do not say %q", want)
		}
	}
	if !strings.Contains(cfg.ShortPrompt(), "respond with Y") || !strings.Contains(cfg.LongPrompt(), "respond with Y") {
		t.Error("the re-prompts do not ask for the configured complete marker")
	}
}

func TestCustomInstructionsStillWin(t *testing.T) {
	cfg := UserTurnCompletionConfig{Instructions: "do it my way"}
	if got := cfg.CompletionInstructions(); got != "do it my way" {
		t.Errorf("CompletionInstructions() = %q, want the configured ones", got)
	}
}

func TestParserDetectsConfiguredMarkers(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{CompleteMarker: "Y", IncompleteShortMarker: "N", IncompleteLongMarker: "W"})

	r.text(t, "Y Hello there!")

	if got := textsOf(r); !slices.Equal(got, []string{"Hello there!"}) {
		t.Errorf("pushed text = %v, want the answer past the configured marker", got)
	}
	if got := framesOf[*frames.LLMMarkerFrame](r); len(got) != 1 || got[0].Marker != "Y" {
		t.Errorf("markers = %v, want the configured complete marker", got)
	}
}

// The turn-completion judgment pushed upstream as a UserTurnCompletionFrame.

// reportsOf are the reports recorded, with their directions.
func reportsOf(r *reader) []pushedFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []pushedFrame
	for _, p := range r.pushed {
		if _, ok := p.frame.(*frames.UserTurnCompletionFrame); ok {
			out = append(out, p)
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
			r := newReader(UserTurnCompletionConfig{})

			r.text(t, tt.marker+" Hello")

			reports := reportsOf(r)
			if len(reports) != 1 {
				t.Fatalf("reports = %d, want 1", len(reports))
			}
			report, ok := reports[0].frame.(*frames.UserTurnCompletionFrame)
			if !ok {
				t.Fatalf("recorded %T, want a UserTurnCompletionFrame", reports[0].frame)
			}
			if report.Kind != tt.kind || report.Marker != tt.marker {
				t.Errorf("report = (%q, %q), want (%q, %q)", report.Kind, report.Marker, tt.kind, tt.marker)
			}
			if reports[0].dir != processor.Upstream {
				t.Error("the report should be pushed upstream")
			}
		})
	}
}

func TestReportsCarryTheContextFrameID(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	contextFrame := frames.NewLLMContextFrame(frames.NewLLMContext(""))
	r.process(t, contextFrame)

	r.text(t, MarkerComplete+" Hello")

	reports := framesOf[*frames.UserTurnCompletionFrame](r)
	if len(reports) != 1 || reports[0].ContextFrameID != contextFrame.ID() {
		t.Fatalf("reports = %v, want one naming frame %d", reports, contextFrame.ID())
	}
}

func TestAToolCallIsReportedAsComplete(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})

	r.push(t, frames.NewFunctionCallsStartedFrame(nil))

	reports := framesOf[*frames.UserTurnCompletionFrame](r)
	if len(reports) != 1 || reports[0].Kind != "complete" || reports[0].Marker != "" {
		t.Fatalf("reports = %v, want one complete with no marker", reports)
	}
	if n := len(framesOf[*frames.UserTurnInferenceCompletedFrame](r)); n != 2 {
		t.Errorf("completion broadcasts = %d, want 2", n)
	}
}

func TestAResponseWithoutAMarkerIsReportedWithNoKind(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, "Hello there")
	r.push(t, frames.NewLLMFullResponseEndFrame())

	reports := framesOf[*frames.UserTurnCompletionFrame](r)
	if len(reports) != 1 || reports[0].Kind != "" || reports[0].Marker != "" {
		t.Fatalf("reports = %v, want one with no kind", reports)
	}
}

func TestEachInferenceIsReportedOnce(t *testing.T) {
	r := newReader(UserTurnCompletionConfig{})
	r.process(t, frames.NewLLMContextFrame(frames.NewLLMContext("")))
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, MarkerComplete+" Let me check")
	r.push(t, frames.NewFunctionCallsStartedFrame(nil))
	r.push(t, frames.NewLLMFullResponseEndFrame())

	// The next inference is reported afresh.
	r.process(t, frames.NewLLMContextFrame(frames.NewLLMContext("")))
	r.push(t, frames.NewLLMFullResponseStartFrame())
	r.text(t, MarkerIncompleteShort)
	r.push(t, frames.NewLLMFullResponseEndFrame())

	var kinds []string
	for _, report := range framesOf[*frames.UserTurnCompletionFrame](r) {
		kinds = append(kinds, report.Kind)
	}
	if !slices.Equal(kinds, []string{"complete", "short"}) {
		t.Fatalf("kinds = %v, want [complete short]", kinds)
	}
}

// captureLog collects what is logged at warning level while run runs.
func captureLog(t *testing.T, run func()) string {
	t.Helper()
	var buf strings.Builder
	var mu sync.Mutex
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(writerFunc(func(p []byte) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return buf.Write(p)
	}), &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	run()
	mu.Lock()
	defer mu.Unlock()
	return buf.String()
}

type writerFunc func([]byte) (int, error)

func (w writerFunc) Write(p []byte) (int, error) { return w(p) }

// The strategy's own update, which installs the extension, is how turn
// completion is meant to arrive, so it warns of nothing.
func TestTheStrategysSettingsUpdateIsSilent(t *testing.T) {
	svc := plainService("")
	logged := captureLog(t, func() { enableTurnCompletion(t, svc, UserTurnCompletionConfig{}) })

	if strings.Contains(logged, "deprecated") {
		t.Errorf("the strategy's update warned: %s", logged)
	}
	if _, ok := svc.Extensions()[0].(*TurnCompletionExtension); !ok {
		t.Error("the turn-completion extension is not the closest to the service")
	}
}

// The extension reads the response the service generates: the marker is
// stripped from the spoken text, reported upstream, and the response's report
// precedes its end.
func TestTheExtensionReadsTheServicesResponse(t *testing.T) {
	svc := New("FakeLLM", generatorFunc(func(_ context.Context, _ *frames.LLMContext, emit Emit) error {
		if err := emit("Well "); err != nil {
			return err
		}
		return emit(MarkerComplete + " Hello there!")
	}))
	if err := svc.AddExtension(NewTurnCompletionExtension(UserTurnCompletionConfig{})); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var down, up []frames.Frame
	sink := func(f frames.Frame, dir processor.Direction) {
		mu.Lock()
		defer mu.Unlock()
		if dir == processor.Upstream {
			up = append(up, f)
		} else {
			down = append(down, f)
		}
	}
	after := newRecordingLink("After", sink)
	before := newRecordingLink("Before", sink)
	before.Link(svc)
	svc.Link(after)
	for _, p := range []processor.Processor{before, svc, after} {
		if err := p.Setup(t.Context(), processor.Setup{}); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		for _, p := range []processor.Processor{before, svc, after} {
			_ = p.Cleanup(context.Background())
		}
	})

	// A processor drains nothing until its StartFrame arrives, so one is queued
	// at the head of the chain the way a running pipeline would.
	if err := before.QueueFrame(t.Context(), frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	seen := func(match func(frames.Frame) bool, in *[]frames.Frame) func() bool {
		return func() bool {
			mu.Lock()
			defer mu.Unlock()
			return slices.ContainsFunc(*in, match)
		}
	}
	isType := func(want frames.Frame) func(frames.Frame) bool {
		return func(f frames.Frame) bool { return reflect.TypeOf(f) == reflect.TypeOf(want) }
	}
	waitUntil(t, "the StartFrame never reached the end of the chain",
		seen(isType(frames.NewStartFrame()), &down))

	if err := svc.run(t.Context(), frames.NewLLMContext("")); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the response never ended downstream", seen(isType(frames.NewLLMFullResponseEndFrame()), &down))
	waitUntil(t, "nothing was reported upstream", seen(isType(frames.NewUserTurnCompletionFrame("", "", 0)), &up))

	mu.Lock()
	defer mu.Unlock()
	var texts []string
	reportAt, endAt := -1, -1
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.LLMTextFrame:
			texts = append(texts, fr.Text)
		case *frames.LLMMarkerResponseFrame:
			reportAt = i
		case *frames.LLMFullResponseEndFrame:
			endAt = i
		}
	}
	if !slices.Equal(texts, []string{"Hello there!"}) {
		t.Errorf("spoken text = %v, want [Hello there!]", texts)
	}
	if reportAt < 0 || reportAt > endAt {
		t.Errorf("the response's report is at %d and its end at %d, want the report first", reportAt, endAt)
	}
	var kinds []string
	for _, f := range up {
		if report, ok := f.(*frames.UserTurnCompletionFrame); ok {
			kinds = append(kinds, report.Kind)
		}
	}
	if !slices.Equal(kinds, []string{"complete"}) {
		t.Errorf("reports upstream = %v, want one complete", kinds)
	}
}

// recordingLink hands every frame that reaches it to a sink, and passes it on.
type recordingLink struct {
	*processor.Base
	sink func(frames.Frame, processor.Direction)
}

func newRecordingLink(name string, sink func(frames.Frame, processor.Direction)) *recordingLink {
	l := &recordingLink{sink: sink}
	l.Base = processor.New(name, l)
	return l
}

func (l *recordingLink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	l.sink(f, dir)
	return l.PushFrame(ctx, f, dir)
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

// Enabling turn completion composes after an appended guide, once, and
// disabling restores the base prompt and the guide.
func TestTurnCompletionComposesAfterTheAppendedGuide(t *testing.T) {
	svc := plainService("APP")
	svc.AppendSystemInstruction("GUIDE")

	enableTurnCompletion(t, svc, UserTurnCompletionConfig{})
	composed := svc.SystemInstruction()
	if !strings.HasPrefix(composed, "APP\n\nGUIDE\n\n") || strings.Count(composed, "GUIDE") != 1 {
		t.Errorf("SystemInstruction() = %q, want APP, GUIDE once, then the protocol", composed)
	}
}
