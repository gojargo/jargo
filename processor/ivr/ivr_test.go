package ivr_test

// Ported from upstream's IVR navigation suite: each case runs the processor in
// a pipeline and checks the frames reaching each end, and the events it raised.

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/ivr"
	"github.com/gojargo/jargo/utils/events"
)

// step is a frame to queue, or a pause when f is nil.
type step struct {
	f     frames.Frame
	pause time.Duration
}

func send(fs ...frames.Frame) []step {
	out := make([]step, len(fs))
	for i, f := range fs {
		out[i] = step{f: f}
	}
	return out
}

// recorded is what reached each end of the pipeline and what was raised.
type recorded struct {
	mu            sync.Mutex
	down, up      []frames.Frame
	conversations [][]frames.Message
	statuses      []ivr.Status
	sources       []any
}

func (r *recorded) add(dst *[]frames.Frame, f frames.Frame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	*dst = append(*dst, f)
}

// names lists the type names of fs, Start and End frames left out.
func names(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		n := strings.TrimPrefix(fmt.Sprintf("%T", f), "*frames.")
		if n == "StartFrame" || n == "EndFrame" {
			continue
		}
		out = append(out, n)
	}
	return out
}

// newProcessor is the processor every case runs, with the fixture's prompts.
func newProcessor() *ivr.Processor {
	params := vad.DefaultParams()
	params.StopSecs = 2.0
	return ivr.New(ivr.Config{
		ClassifierPrompt: "Classify as IVR or conversation",
		IVRPrompt:        "Navigate to the billing department",
		IVRVADParams:     &params,
	})
}

// run runs p over steps, then ends the pipeline.
func run(t *testing.T, p *ivr.Processor, steps []step) *recorded {
	t.Helper()
	r := &recorded{}
	p.Events().Add(ivr.EventConversationDetected, func(_ context.Context, src any, args ...any) {
		r.mu.Lock()
		defer r.mu.Unlock()
		history, _ := args[0].([]frames.Message)
		r.conversations = append(r.conversations, history)
		r.sources = append(r.sources, src)
	})
	p.Events().Add(ivr.EventIVRStatusChanged, func(_ context.Context, src any, args ...any) {
		r.mu.Lock()
		defer r.mu.Unlock()
		s, _ := args[0].(ivr.Status)
		r.statuses = append(r.statuses, s)
		r.sources = append(r.sources, src)
	})
	// No RTVI: its messages to the client would sit among the frames checked.
	noRTVI := false
	w := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&w.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.add(&r.down, f)
	})
	events.On(&w.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		r.add(&r.up, f)
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	for _, s := range steps {
		if s.f == nil {
			time.Sleep(s.pause)
			continue
		}
		w.QueueFrame(s.f)
	}
	w.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not end")
	}
	return r
}

func expectFrames(t *testing.T, r *recorded, wantDown, wantUp []string) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if got := names(r.down); !reflect.DeepEqual(got, wantDown) {
		t.Errorf("down = %v, want %v", got, wantDown)
	}
	if got := names(r.up); !reflect.DeepEqual(got, wantUp) {
		t.Errorf("up = %v, want %v", got, wantUp)
	}
}

// test_switch_to_ivr_mode: the classifier prompt goes up at start, then the
// navigation prompt and the IVR timing on the verdict.
func TestSwitchToIVRMode(t *testing.T) {
	r := run(t, newProcessor(), send(frames.NewLLMTextFrame("<mode>ivr</mode>")))
	expectFrames(t, r, nil, []string{"LLMMessagesUpdateFrame", "LLMMessagesUpdateFrame", "VADParamsUpdateFrame"})

	r.mu.Lock()
	defer r.mu.Unlock()
	update, _ := r.up[1].(*frames.LLMMessagesUpdateFrame)
	if update == nil || !update.RunLLM || update.Messages[0].Text != "Navigate to the billing department" {
		t.Errorf("navigation update = %+v, want the navigation prompt, run at once", r.up[1])
	}
	if vadUpdate, _ := r.up[2].(*frames.VADParamsUpdateFrame); vadUpdate == nil || vadUpdate.Params.StopSecs != 2.0 {
		t.Errorf("VAD update = %+v, want a stop of 2 s", r.up[2])
	}
	if !reflect.DeepEqual(r.statuses, []ivr.Status{ivr.StatusDetected}) {
		t.Errorf("statuses = %v, want detected", r.statuses)
	}
}

// test_switch_to_conversation_mode: a person on the line is raised with the
// conversation so far, empty at first.
func TestSwitchToConversationMode(t *testing.T) {
	p := newProcessor()
	r := run(t, p, send(frames.NewLLMTextFrame("<mode>conversation</mode>")))
	expectFrames(t, r, nil, []string{"LLMMessagesUpdateFrame"})

	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.conversations) != 1 {
		t.Fatalf("conversation detected %d times, want once", len(r.conversations))
	}
	if r.sources[0] != p {
		t.Errorf("the event came from %v, want the processor", r.sources[0])
	}
	if len(r.conversations[0]) != 0 {
		t.Errorf("history = %v, want empty at first", r.conversations[0])
	}
}

// test_basic_dtmf_navigation: a key is pressed at once and recorded.
func TestBasicDTMFNavigation(t *testing.T) {
	r := run(t, newProcessor(), send(frames.NewLLMTextFrame("<dtmf>1</dtmf>")))
	expectFrames(t, r, []string{"OutputDTMFUrgentFrame", "TextFrame"}, []string{"LLMMessagesUpdateFrame"})

	r.mu.Lock()
	defer r.mu.Unlock()
	if key, _ := r.down[1].(*frames.OutputDTMFUrgentFrame); key == nil || key.Buttons[0] != frames.KeypadOne {
		t.Errorf("key = %+v, want 1", r.down[1])
	}
	tf, _ := r.down[2].(*frames.TextFrame)
	if tf == nil || tf.Text != "<dtmf>1</dtmf>" || tf.SkipTTS == nil || !*tf.SkipTTS {
		t.Errorf("recorded = %+v, want the tag, skipped by the synthesizer", r.down[2])
	}
}

// test_multiple_dtmf_navigation: each key pressed and recorded in turn.
func TestMultipleDTMFNavigation(t *testing.T) {
	r := run(t, newProcessor(), send(
		frames.NewLLMTextFrame("<dtmf>1</dtmf>"), frames.NewLLMTextFrame("<dtmf>2</dtmf>"),
		frames.NewLLMTextFrame("<dtmf>3</dtmf>"), frames.NewLLMTextFrame("<dtmf>4</dtmf>"),
	))
	var want []string
	for range 4 {
		want = append(want, "OutputDTMFUrgentFrame", "TextFrame")
	}
	expectFrames(t, r, want, []string{"LLMMessagesUpdateFrame"})
}

// A value that is not one keypad key presses nothing.
func TestInvalidDTMFIsRefused(t *testing.T) {
	r := run(t, newProcessor(), send(frames.NewLLMTextFrame("<dtmf>12</dtmf>"), frames.NewLLMTextFrame("<dtmf>x</dtmf>")))
	expectFrames(t, r, nil, []string{"LLMMessagesUpdateFrame"})
}

// test_ivr_wait: recorded, but raises nothing.
func TestIVRWait(t *testing.T) {
	r := run(t, newProcessor(), send(frames.NewLLMTextFrame("<ivr>wait</ivr>")))
	expectFrames(t, r, []string{"TextFrame"}, []string{"LLMMessagesUpdateFrame"})
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.statuses) != 0 {
		t.Errorf("statuses = %v, want none for wait", r.statuses)
	}
}

// test_ivr_stuck and test_ivr_completed: recorded and raised.
func TestIVRStuckAndCompleted(t *testing.T) {
	for _, status := range []ivr.Status{ivr.StatusStuck, ivr.StatusCompleted} {
		p := newProcessor()
		r := run(t, p, send(frames.NewLLMTextFrame("<ivr>"+string(status)+"</ivr>")))
		expectFrames(t, r, []string{"TextFrame"}, []string{"LLMMessagesUpdateFrame"})
		r.mu.Lock()
		if !reflect.DeepEqual(r.statuses, []ivr.Status{status}) || r.sources[0] != p {
			t.Errorf("%s: statuses = %v from %v, want %s from the processor", status, r.statuses, r.sources, status)
		}
		r.mu.Unlock()
	}
}

// An unknown status is refused: nothing recorded, nothing raised.
func TestUnknownIVRStatusIsRefused(t *testing.T) {
	r := run(t, newProcessor(), send(frames.NewLLMTextFrame("<ivr>lost</ivr>")))
	expectFrames(t, r, nil, []string{"LLMMessagesUpdateFrame"})
}

// test_normal_text_passthrough: text without tags is spoken as a sentence.
func TestNormalTextPassthrough(t *testing.T) {
	r := run(t, newProcessor(), send(
		frames.NewLLMTextFrame("Hello, I'm trying to reach billing."), frames.NewLLMFullResponseEndFrame(),
	))
	expectFrames(t, r, []string{"AggregatedTextFrame", "LLMFullResponseEndFrame"}, []string{"LLMMessagesUpdateFrame"})
}

// test_interruption_discards_partial_response: a tag left open by an
// interrupted response is dropped, not spoken with the next one.
func TestInterruptionDiscardsPartialResponse(t *testing.T) {
	steps := []step{
		{f: frames.NewLLMTextFrame("<dtmf>")},
		{pause: 100 * time.Millisecond},
		{f: frames.NewInterruptionFrame()},
		{pause: 100 * time.Millisecond},
		{f: frames.NewLLMTextFrame("Yes.")},
		{f: frames.NewLLMFullResponseEndFrame()},
	}
	r := run(t, newProcessor(), steps)
	expectFrames(t, r, []string{"InterruptionFrame", "AggregatedTextFrame", "LLMFullResponseEndFrame"},
		[]string{"LLMMessagesUpdateFrame"})
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, f := range r.down {
		if at, ok := f.(*frames.AggregatedTextFrame); ok && at.Text != "Yes." {
			t.Errorf("spoke %q, want only Yes.", at.Text)
		}
	}
}

// The navigator saves the conversation on its way to the LLM, and hands it over
// when a person answers.
func TestNavigatorHandsTheConversationOver(t *testing.T) {
	llm := &scriptedLLM{reply: "<mode>conversation</mode>"}
	llm.Base = processor.New("ScriptedLLM", llm)
	nav := ivr.NewNavigator(ivr.NavigatorConfig{LLM: llm, Goal: "reach billing"})
	got := make(chan []frames.Message, 1)
	events.On(nav.Processor().Events(), ivr.EventConversationDetected, func(_ context.Context, h []frames.Message) {
		got <- h
	})
	w := pipeline.NewWorker(nav.Pipeline, pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()

	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleDeveloper, Text: "classify"},
		{Role: frames.RoleUser, Text: "Hello, this is the front desk."},
	})
	w.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case h := <-got:
		if len(h) != 1 || h[0].Text != "Hello, this is the front desk." {
			t.Errorf("history = %+v, want the conversation without its system message", h)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no conversation detected")
	}
	w.StopWhenDone()
	<-done
}

// scriptedLLM answers every conversation with a fixed reply.
type scriptedLLM struct {
	*processor.Base
	reply string
}

func (l *scriptedLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMContextFrame); ok {
		if err := l.PushFrame(ctx, frames.NewLLMTextFrame(l.reply), processor.Downstream); err != nil {
			return err
		}
		return l.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	}
	return l.PushFrame(ctx, f, dir)
}
