package aggregators_test

import (
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
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
	if got[0].AggregatedBy != frames.AggregationSentence {
		t.Errorf("aggregated by %q, want %q", got[0].AggregatedBy, frames.AggregationSentence)
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
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return &clearingSpy{SimpleAggregator: text.NewSimpleAggregator(frames.AggregationSentence, tok)}
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
