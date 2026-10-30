package aggregators_test

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/text"
)

// collectAggregated drains the aggregated-text frames after a short grace
// period, in the order they arrived.
func collectAggregated(seen chan frames.Frame) []*frames.AggregatedTextFrame {
	var got []*frames.AggregatedTextFrame
	for _, f := range drainAfterGrace(seen) {
		if af, ok := f.(*frames.AggregatedTextFrame); ok {
			got = append(got, af)
		}
	}
	return got
}

func TestLLMTextGroupsTokensIntoSentences(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, token := range []string{"Hello", ",", " world", ". ", "Next", " one"} {
		task.QueueFrame(frames.NewLLMTextFrame(token))
	}

	got := collectAggregated(seen)
	if len(got) != 1 {
		t.Fatalf("aggregated %d units, want 1: only the first sentence has ended", len(got))
	}
	// The boundary is confirmed by the token after it, and the whitespace that
	// separated them belongs to the unit still being gathered.
	if got[0].Text != "Hello, world." {
		t.Errorf("text = %q, want the completed sentence", got[0].Text)
	}
	if got[0].TextType != frames.TextTypeSentence {
		t.Errorf("aggregated by %q, want %q", got[0].TextType, frames.TextTypeSentence)
	}

	task.StopWhenDone()
	<-runDone
}

func TestLLMTextFlushesWhenTheResponseEnds(t *testing.T) {
	// The last unit of a response has no boundary after it, so it would be held
	// back for good without the flush.
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("no full stop here"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	got := collectAggregated(seen)
	if len(got) != 1 || got[0].Text != "no full stop here" {
		t.Errorf("aggregated %+v, want what was held to be flushed", got)
	}

	task.StopWhenDone()
	<-runDone
}

func TestLLMTextConsumesTheTokensItGathers(t *testing.T) {
	// The tokens are what it converts, so they do not travel on beside the
	// aggregation they became.
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))

	task.QueueFrame(frames.NewLLMTextFrame("Hello. "))

	for _, f := range drainAfterGrace(seen) {
		if _, ok := f.(*frames.LLMTextFrame); ok {
			t.Error("a token traveled on as well as being aggregated")
		}
	}

	task.StopWhenDone()
	<-runDone
}

func TestLLMTextDropsWhatAnInterruptionCutOff(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))

	task.QueueFrame(frames.NewLLMTextFrame("half a "))
	awaitNothing(seen)
	task.QueueFrame(frames.NewInterruptionFrame())
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	if got := collectAggregated(seen); len(got) != 0 {
		t.Errorf("aggregated %+v, want the cut-off text dropped", got)
	}

	task.StopWhenDone()
	<-runDone
}

func TestLLMTextCarriesTheSkipTTSDecision(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))

	skip := true
	token := frames.NewLLMTextFrame("Hello. ")
	token.SkipTTS = &skip
	task.QueueFrame(token)
	// A sentence is only confirmed by what follows it, so the response has to
	// end for the last one to be flushed.
	end := frames.NewLLMFullResponseEndFrame()
	end.SkipTTS = &skip
	task.QueueFrame(end)

	got := collectAggregated(seen)
	if len(got) != 1 {
		t.Fatalf("aggregated %d units, want 1", len(got))
	}
	if got[0].SkipTTS == nil || !*got[0].SkipTTS {
		t.Error("the aggregation lost the decision to keep the text unspoken")
	}

	task.StopWhenDone()
	<-runDone
}

// clearingSpy records which of the aggregator's two clearing hooks it was
// given, so a test can tell an interruption from a reset.
type clearingSpy struct {
	*text.SimpleAggregator

	mu            sync.Mutex
	interruptions int
	resets        int
}

func newClearingSpy(t *testing.T) *clearingSpy {
	t.Helper()
	return &clearingSpy{SimpleAggregator: text.NewSimpleAggregator(frames.TextTypeSentence, "")}
}

func (a *clearingSpy) HandleInterruption() {
	a.mu.Lock()
	a.interruptions++
	a.mu.Unlock()
	a.SimpleAggregator.HandleInterruption()
}

func (a *clearingSpy) Reset() {
	a.mu.Lock()
	a.resets++
	a.mu.Unlock()
	a.SimpleAggregator.Reset()
}

func (a *clearingSpy) counts() (interruptions, resets int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.interruptions, a.resets
}

// An interruption reaches the aggregator as an interruption, not as a reset.
// The two are separate hooks so an aggregator can treat them differently, and
// an aggregator that does would never hear about the barge-in otherwise.
func TestLLMTextTellsTheAggregatorItWasInterrupted(t *testing.T) {
	spy := newClearingSpy(t)
	task, seen, runDone := runProcessor(t, aggregators.NewLLMTextWith("LLMText", spy))

	task.QueueFrame(frames.NewLLMTextFrame("half a "))
	awaitNothing(seen)
	task.QueueFrame(frames.NewInterruptionFrame())

	task.StopWhenDone()
	<-runDone

	interruptions, resets := spy.counts()
	if interruptions != 1 {
		t.Errorf("aggregator was interrupted %d times, want 1", interruptions)
	}
	if resets != 0 {
		t.Errorf("aggregator was reset %d times, want 0: an interruption is not a reset", resets)
	}
}

// Reset, in contrast, is what the processor's own Reset hands on.
func TestLLMTextResetIsNotAnInterruption(t *testing.T) {
	spy := newClearingSpy(t)
	p := aggregators.NewLLMTextWith("LLMText", spy)

	p.Reset()

	interruptions, resets := spy.counts()
	if resets != 1 {
		t.Errorf("aggregator was reset %d times, want 1", resets)
	}
	if interruptions != 0 {
		t.Errorf("aggregator was interrupted %d times, want 0", interruptions)
	}
}

// runLLMText sends a response through an LLMText and returns the aggregations
// that reached the end of the pipeline and the errors that reached its start.
func runLLMText(
	t *testing.T, p *aggregators.LLMText, texts ...string,
) ([]*frames.AggregatedTextFrame, []*frames.ErrorFrame) {
	t.Helper()
	var (
		mu         sync.Mutex
		aggregated []*frames.AggregatedTextFrame
		errors     []*frames.ErrorFrame
	)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if af, ok := f.(*frames.AggregatedTextFrame); ok {
			mu.Lock()
			aggregated = append(aggregated, af)
			mu.Unlock()
		}
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errors = append(errors, ef)
			mu.Unlock()
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, s := range texts {
		task.QueueFrame(frames.NewLLMTextFrame(s))
	}
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	if err := <-runDone; err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	return aggregated, errors
}

func transformer(textType frames.TextType, transform func(string) (string, error)) tts.TextTransformer {
	return tts.TextTransformer{
		TextType: textType,
		Transform: func(_ context.Context, text string, _ frames.TextType) (string, error) {
			return transform(text)
		},
	}
}

func upper(text string) (string, error) { return strings.ToUpper(text), nil }

var errBoom = errors.New("boom")

// A "*" transform rewrites every aggregation, flushed text included.
func TestLLMTextTransformRewritesTextAndRawText(t *testing.T) {
	plain, _ := runLLMText(t, aggregators.NewLLMText("LLMText"), "Hello there. ", "Bye")
	p := aggregators.NewLLMText("LLMText")
	p.SetTextTransformers(transformer(frames.AnyTextType, upper))
	got, _ := runLLMText(t, p, "Hello there. ", "Bye")

	if len(got) != len(plain) || len(got) != 2 {
		t.Fatalf("%d aggregations, want %d", len(got), len(plain))
	}
	for i := range got {
		want := strings.ToUpper(plain[i].Text)
		if got[i].Text != want || got[i].RawText != want || got[i].TextType != plain[i].TextType {
			t.Errorf("aggregation %d = %q (raw %q, %s), want %q (raw %q, %s)",
				i, got[i].Text, got[i].RawText, got[i].TextType, want, want, plain[i].TextType)
		}
	}
}

func codeAggregator(t *testing.T) *text.PatternPairAggregator {
	t.Helper()
	a := text.NewPatternPairAggregator(frames.TextTypeSentence, "")
	if err := a.AddPattern("code", "<code>", "</code>", text.MatchAggregate); err != nil {
		t.Fatal(err)
	}
	return a
}

// A transform for one type leaves the other types alone and keeps delimiters.
func TestLLMTextTransformAppliesToItsTypeOnly(t *testing.T) {
	const reply = "Run this <code>print(1)</code> now. "
	plain, _ := runLLMText(t, aggregators.NewLLMTextWith("LLMText", codeAggregator(t)), reply)
	p := aggregators.NewLLMTextWith("LLMText", codeAggregator(t))
	p.SetTextTransformers(transformer("code", upper))
	got, _ := runLLMText(t, p, reply)

	var code []*frames.AggregatedTextFrame
	type pair struct{ text, raw string }
	var others, plainOthers []pair
	for _, f := range got {
		if f.TextType == "code" {
			code = append(code, f)
		} else {
			others = append(others, pair{f.Text, f.RawText})
		}
	}
	for _, f := range plain {
		if f.TextType != "code" {
			plainOthers = append(plainOthers, pair{f.Text, f.RawText})
		}
	}
	if len(code) != 1 || code[0].Text != "PRINT(1)" || code[0].RawText != "<code>PRINT(1)</code>" {
		t.Fatalf("code aggregations = %+v, want one PRINT(1) inside its delimiters", code)
	}
	if !slices.Equal(others, plainOthers) || len(others) != 2 {
		t.Errorf("other aggregations = %+v, want them untouched: %+v", others, plainOthers)
	}
}

func TestLLMTextTransformLeavingWhitespaceDropsTheAggregation(t *testing.T) {
	p := aggregators.NewLLMText("LLMText")
	p.SetTextTransformers(transformer(frames.AnyTextType, func(string) (string, error) { return "", nil }))
	if got, _ := runLLMText(t, p, "Leaked reasoning. "); len(got) != 0 {
		t.Errorf("aggregated %+v, want nothing", got)
	}
}

func TestLLMTextFailingTransformDropsTheAggregationAndReports(t *testing.T) {
	p := aggregators.NewLLMText("LLMText")
	p.SetTextTransformers(transformer(frames.AnyTextType, func(string) (string, error) {
		return "", errBoom
	}))
	got, errs := runLLMText(t, p, "Hello there. ")
	if len(got) != 0 {
		t.Errorf("aggregated %+v, want nothing", got)
	}
	if len(errs) != 1 || !strings.Contains(errs[0].Error, "boom") {
		t.Errorf("errors = %+v, want one reporting the failure", errs)
	}
}

// Go function values cannot be compared, so a transform is withdrawn by setting
// the list again without it.
func TestLLMTextTransformerCanBeAddedAndRemoved(t *testing.T) {
	p := aggregators.NewLLMText("LLMText")
	p.AddTextTransformer(transformer(frames.AnyTextType, upper))
	p.SetTextTransformers()
	got, _ := runLLMText(t, p, "Hello there. ")
	if len(got) != 1 || got[0].Text != "Hello there." {
		t.Errorf("aggregated %+v, want the text untouched", got)
	}
}

// A speech-to-speech service stops its speech before it ends the response. The
// last sentence has no following text to close it, so the processor still holds
// it when the speech stops. It was spoken before the stop and goes out ahead of
// it.
func TestLLMTextPushesHeldTextBeforeTTSStopped(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewLLMText("LLMText"))
	task.QueueFrame(frames.NewLLMTextFrame("Hello there. "))
	task.QueueFrame(frames.NewLLMTextFrame("It's provider-agnostic."))
	task.QueueFrame(frames.NewTTSStoppedFrame())
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	var names, texts []string
	for _, f := range drainAfterGrace(seen) {
		switch fr := f.(type) {
		case *frames.AggregatedTextFrame:
			names = append(names, "AggregatedTextFrame")
			texts = append(texts, fr.Text)
		case *frames.TTSStoppedFrame:
			names = append(names, "TTSStoppedFrame")
		case *frames.LLMFullResponseEndFrame:
			names = append(names, "LLMFullResponseEndFrame")
		}
	}
	task.StopWhenDone()
	<-runDone

	wantNames := []string{"AggregatedTextFrame", "AggregatedTextFrame", "TTSStoppedFrame", "LLMFullResponseEndFrame"}
	if !slices.Equal(names, wantNames) {
		t.Errorf("frames = %v, want %v", names, wantNames)
	}
	if want := []string{"Hello there.", "It's provider-agnostic."}; !slices.Equal(texts, want) {
		t.Errorf("texts = %q, want %q", texts, want)
	}
}

// messageOutput is an output transport that takes every audio frame without
// playing it and records the RTVI messages it is asked to send.
type messageOutput struct {
	*transport.BaseOutput
	mu   sync.Mutex
	sent []rtvi.Message
}

func newMessageOutput() *messageOutput {
	o := &messageOutput{}
	o.BaseOutput = transport.NewBaseOutput("Output", transport.Params{AudioOutEnabled: true}, o)
	return o
}

func (o *messageOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	return true, nil
}

func (o *messageOutput) SendMessage(_ context.Context, f frames.OutputTransportMessage) error {
	if m, ok := f.TransportMessage().(rtvi.Message); ok {
		o.mu.Lock()
		o.sent = append(o.sent, m)
		o.mu.Unlock()
	}
	return nil
}

func (o *messageOutput) botOutput() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	var out []string
	for _, m := range o.sent {
		if d, ok := m.Data.(rtvi.BotOutputData); ok && m.Type == rtvi.TypeBotOutput {
			out = append(out, d.Text)
		}
	}
	return out
}

func speech(contextID string) *frames.TTSAudioRawFrame {
	f := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x10, 0x27}, 1600), 16000, 1)
	f.ContextID = contextID
	return f
}

// A reply's last sentence reaches clients before the user speaks again. Text
// that reaches the output transport after the bot stopped speaking waits for the
// next speech, and the user's next turn interrupts before that.
func TestLLMTextSpeechToSpeechReplyEndsWithItsLastSentence(t *testing.T) {
	proc := rtvi.NewProcessor()
	out := newMessageOutput()
	task := pipeline.NewWorker(pipeline.New(proc, aggregators.NewLLMText("LLMText"), out), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{rtvi.NewObserver(proc)},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	started := frames.NewTTSStartedFrame()
	started.ContextID = "reply"
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = "reply"
	for _, f := range []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		started,
		speech("reply"),
		frames.NewLLMTextFrame("Hello there. "),
		speech("reply"),
		frames.NewLLMTextFrame("It's provider-agnostic."),
		stopped,
		frames.NewLLMFullResponseEndFrame(),
	} {
		task.QueueFrame(f)
	}
	time.Sleep(500 * time.Millisecond)
	// The user starts their next turn.
	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	if err := <-runDone; err != nil {
		t.Fatal(err)
	}

	if got, want := out.botOutput(), []string{"Hello there.", "It's provider-agnostic."}; !slices.Equal(got, want) {
		t.Errorf("bot output = %q, want %q", got, want)
	}
}
