package turns_test

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

const (
	reassuranceInterval = 7 * time.Second
	mark                = turns.WaitReassuranceSilentMark
)

// passingGen stands in for a model that is never asked to generate.
type passingGen struct{}

func (passingGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

// reassuringService builds an LLM service, which pushes every frame it does not
// handle onward, with the wait-reassurance extension attached.
func reassuringService(t *testing.T, cfg turns.WaitReassuranceConfig) *llm.Base {
	t.Helper()
	if cfg.Interval == 0 {
		cfg.Interval = reassuranceInterval
	}
	svc := llm.New("FakeLLM", passingGen{})
	if err := svc.AddExtension(turns.NewWaitReassuranceLLMExtension(cfg)); err != nil {
		t.Fatalf("AddExtension: %v", err)
	}
	return svc
}

// runService runs p alone in a pipeline, sends it each frame in order (a
// time.Duration sleeps that long instead), lets it finish, and returns the
// frames that came out of it downstream and upstream, without the StartFrame,
// the EndFrame and the service's metadata.
func runService(t *testing.T, p processor.Processor, send ...any) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	noRTVI := false
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	keep := func(f frames.Frame) bool {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame, *frames.LLMServiceMetadataFrame:
			return false
		}
		return true
	}
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			down = append(down, f)
			mu.Unlock()
		}
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			up = append(up, f)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range send {
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v)
		default:
			t.Fatalf("cannot send %T", s)
		}
	}
	task.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

// response is one reply of the model: its start, its texts and its end.
func response(texts ...string) []any {
	out := []any{frames.NewLLMFullResponseStartFrame()}
	for _, text := range texts {
		out = append(out, frames.NewLLMTextFrame(text))
	}
	return append(out, frames.NewLLMFullResponseEndFrame())
}

// sends joins frames and responses into one sequence to send.
func sends(parts ...any) []any {
	var out []any
	for _, p := range parts {
		if list, ok := p.([]any); ok {
			out = append(out, list...)
			continue
		}
		out = append(out, p)
	}
	return out
}

func texts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

func prompts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if af, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			out = append(out, af.Messages[0].Text)
		}
	}
	return out
}

func reports(fs []frames.Frame) []*frames.LLMMarkerResponseFrame {
	var out []*frames.LLMMarkerResponseFrame
	for _, f := range fs {
		if rf, ok := f.(*frames.LLMMarkerResponseFrame); ok {
			out = append(out, rf)
		}
	}
	return out
}

func kinds(rs []*frames.LLMMarkerResponseFrame) []string {
	out := []string{}
	for _, r := range rs {
		out = append(out, r.Kind)
	}
	return out
}

// withTurnCompletion is the settings update that attaches the turn-completion
// extension, built for the service it reaches.
func withTurnCompletion() frames.Frame {
	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions{
		func() llm.Extension { return llm.NewTurnCompletionExtension(llm.UserTurnCompletionConfig{}) },
	})
	return frames.NewLLMUpdateSettingsFrame(&delta)
}

// What the LLM is told, and how the wait is configured.

func TestTheInstructionExplainsThePromptAndTheMark(t *testing.T) {
	instruction := reassuringService(t, turns.WaitReassuranceConfig{}).SystemInstruction()
	if !strings.Contains(instruction, mark) {
		t.Errorf("instruction %q does not carry the mark", instruction)
	}
	if strings.Contains(instruction, "{mark}") {
		t.Errorf("instruction %q still holds the {mark} field", instruction)
	}
	// The instruction names the message by its opening.
	opening, _, _ := strings.Cut(turns.WaitReassurancePrompt, ",")
	if !strings.Contains(instruction, `"`+opening) {
		t.Errorf("instruction %q does not name the prompt by its opening %q", instruction, opening)
	}
}

func TestTheWaitingTimeoutIsSetUpstreamAtStart(t *testing.T) {
	_, up := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}))
	var updates []*frames.UserWaitingTimeoutUpdateFrame
	for _, f := range up {
		if u, ok := f.(*frames.UserWaitingTimeoutUpdateFrame); ok {
			updates = append(updates, u)
		}
	}
	if len(updates) != 1 || updates[0].Timeout != reassuranceInterval {
		t.Fatalf("upstream waiting timeout updates = %v, want one of %v", updates, reassuranceInterval)
	}
}

// When the model is asked.

func TestTheModelIsAskedWhenTheUserHasWaited(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}), frames.NewUserWaitingFrame(12.4))
	var appends []*frames.LLMMessagesAppendFrame
	for _, f := range down {
		if a, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			appends = append(appends, a)
		}
	}
	if len(appends) != 1 {
		t.Fatalf("appends = %d, want one", len(appends))
	}
	got := appends[0]
	if !got.RunLLM {
		t.Error("RunLLM = false, want the model run")
	}
	if got.Messages[0].Role != frames.RoleDeveloper {
		t.Errorf("role = %q, want developer", got.Messages[0].Role)
	}
	want := strings.ReplaceAll(turns.WaitReassurancePrompt, "{quiet_secs}", "12")
	if got.Messages[0].Text != want {
		t.Errorf("prompt = %q, want %q", got.Messages[0].Text, want)
	}
}

func TestTheInstructionPromptAndMarkAreTheApps(t *testing.T) {
	svc := reassuringService(t, turns.WaitReassuranceConfig{
		Instruction: "Say {mark} to pass.",
		Prompt:      "Quiet for {quiet_secs}s; say {mark} to pass.",
		SilentMark:  "PASS",
	})
	if got := svc.SystemInstruction(); got != "Say PASS to pass." {
		t.Errorf("SystemInstruction() = %q, want the app's instruction with its mark", got)
	}

	down, _ := runService(t, svc, sends(frames.NewUserWaitingFrame(3), response("PASS"))...)

	if got := prompts(down); !slices.Equal(got, []string{"Quiet for 3s; say PASS to pass."}) {
		t.Errorf("prompts = %q, want the app's prompt", got)
	}
	if got := texts(down); len(got) != 0 {
		t.Errorf("texts = %q, want none", got)
	}
	if rs := reports(down); len(rs) != 1 || rs[0].Marker != "PASS" {
		t.Errorf("reports = %v, want one for PASS", rs)
	}
}

// Reading the reply for the mark.

func TestAReplyOpeningWithTheMarkIsDroppedWholeAndReportedBothWays(t *testing.T) {
	down, up := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		response(" ", mark, " and more")...)

	if got := texts(down); len(got) != 0 {
		t.Errorf("texts = %q, want none", got)
	}
	rs := reports(down)
	if len(rs) != 1 {
		t.Fatalf("downstream reports = %d, want one", len(rs))
	}
	report := rs[0]
	if report.Marker != mark || report.Kind != frames.LLMMarkerKindSilent || report.Raw != " "+mark ||
		!slices.Equal(report.Markers, []string{mark}) {
		t.Errorf("report = %+v, want the silent mark, raw %q", report, " "+mark)
	}
	// The user aggregator, upstream, starts its timer over on the same report.
	ups := reports(up)
	if len(ups) != 1 || ups[0].Kind != frames.LLMMarkerKindSilent {
		t.Fatalf("upstream reports = %v, want one silent", ups)
	}
	if sibling, ok := ups[0].BroadcastSiblingID(); !ok || sibling != report.ID() {
		t.Errorf("upstream report's sibling = %d (%v), want the downstream report %d", sibling, ok, report.ID())
	}
}

func TestAnyOtherReplyGoesOutAsWritten(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}), response("Still ", "on it.")...)

	if got := texts(down); !slices.Equal(got, []string{"Still ", "on it."}) {
		t.Errorf("texts = %q, want the reply as written", got)
	}
	if rs := reports(down); len(rs) != 0 {
		t.Errorf("reports = %v, want none", rs)
	}
}

func TestHeldWhitespaceGoesOutWithTheTextThatDecidedIt(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		response(" ", "\n", "Nearly there.")...)

	if got := texts(down); !slices.Equal(got, []string{" \nNearly there."}) {
		t.Errorf("texts = %q, want the held whitespace with the text", got)
	}
}

func TestASilentReplyDoesNotShadowTheNext(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		sends(response(mark), response("Hello."))...)

	if got := texts(down); !slices.Equal(got, []string{"Hello."}) {
		t.Errorf("texts = %q, want the next reply", got)
	}
	if rs := reports(down); len(rs) != 1 {
		t.Errorf("reports = %d, want one", len(rs))
	}
}

func TestAnInterruptionForgetsAHeldOpening(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}), sends(
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame(" "),
		frames.NewInterruptionFrame(),
		response("Hello."),
	)...)

	if got := texts(down); !slices.Equal(got, []string{"Hello."}) {
		t.Errorf("texts = %q, want only the next reply", got)
	}
}

// With turn completion on, its marker is stripped first, so the mark after it
// still reads.
func TestATurnMarkerBeforeTheMarkIsReadAsSilentToo(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}), sends(
		withTurnCompletion(),
		response(frames.UserTurnCompleteMarker+" "+mark),
	)...)

	if got := texts(down); len(got) != 0 {
		t.Errorf("texts = %q, want none", got)
	}
	if got := kinds(reports(down)); !slices.Equal(got, []string{frames.LLMMarkerKindSilent, "complete"}) {
		t.Errorf("report kinds = %q, want silent then complete", got)
	}
}

// interruptedWhileTurnCompletionHolds is a reply without a turn marker,
// interrupted while turn completion still holds its text.
func interruptedWhileTurnCompletionHolds(text string) []any {
	return []any{
		withTurnCompletion(),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame(text),
		// The interruption is a system frame: without the pause it would reach
		// the service ahead of the text.
		50 * time.Millisecond,
		frames.NewInterruptionFrame(),
	}
}

// Turn completion sits closer to the service, so the text it pushes passes this
// extension.
func TestAMarkTurnCompletionLetsGoOfOnAnInterruptionIsReadAsSilent(t *testing.T) {
	down, up := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		interruptedWhileTurnCompletionHolds(mark)...)

	if got := texts(down); len(got) != 0 {
		t.Errorf("texts = %q, want none", got)
	}
	if got := kinds(reports(down)); !slices.Equal(got, []string{frames.LLMMarkerKindSilent}) {
		t.Errorf("downstream report kinds = %q, want silent", got)
	}
	if got := kinds(reports(up)); !slices.Equal(got, []string{frames.LLMMarkerKindSilent}) {
		t.Errorf("upstream report kinds = %q, want silent", got)
	}
}

func TestOtherTextTurnCompletionLetsGoOfOnAnInterruptionGoesOutAsWritten(t *testing.T) {
	down, up := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		interruptedWhileTurnCompletionHolds("Hello there")...)

	if got := texts(down); !slices.Equal(got, []string{"Hello there"}) {
		t.Errorf("texts = %q, want the text as written", got)
	}
	if rs := reports(down); len(rs) != 0 {
		t.Errorf("downstream reports = %v, want none", rs)
	}
	if rs := reports(up); len(rs) != 0 {
		t.Errorf("upstream reports = %v, want none", rs)
	}
}

func TestABareMarkUnderTurnCompletionIsReadAsSilentWhenTheReplyEnds(t *testing.T) {
	down, _ := runService(t, reassuringService(t, turns.WaitReassuranceConfig{}),
		sends(withTurnCompletion(), response(mark))...)

	if got := texts(down); len(got) != 0 {
		t.Errorf("texts = %q, want none", got)
	}
	// Turn completion reports first: its "no marker" report and the held text
	// come out together at the end of the reply, and the mark is read from the
	// text.
	if got := kinds(reports(down)); !slices.Equal(got, []string{"", frames.LLMMarkerKindSilent}) {
		t.Errorf("report kinds = %q, want none then silent", got)
	}
}
