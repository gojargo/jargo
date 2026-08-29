package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// codeType is the aggregation type the pattern below cuts its units under.
const codeType = frames.AggregationType("code")

// runWithSkippedType feeds one turn through a base that cuts <code> blocks out
// under their own type and is told not to speak them. It returns what the
// provider was asked to say and the text frames that reached the pipeline, in
// the order they arrived.
func runWithSkippedType(t *testing.T, skip bool, text string) (spoken []string, down []frames.Frame) {
	t.Helper()

	agg := ttstext.NewPatternPairAggregator(frames.AggregationSentence, newTokenizer(t))
	if err := agg.AddPattern(codeType, "<code>", "</code>", ttstext.MatchAggregate); err != nil {
		t.Fatal(err)
	}

	syn := &spacedSynth{}
	base := tts.New("SkipTTS", syn)
	base.SetTextAggregator(agg)
	if skip {
		base.SetSkipAggregatorTypes(codeType)
	}

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.AggregatedTextFrame, *frames.TTSTextFrame:
			mu.Lock()
			down = append(down, f)
			mu.Unlock()
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame(text))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return syn.texts(), append([]frames.Frame(nil), down...)
}

// A block cut out under a type that is not spoken never reaches the provider.
func TestSkippedAggregationTypeIsNotSynthesized(t *testing.T) {
	spoken, _ := runWithSkippedType(t, true, "Here is code <code>print here</code> done.")

	for _, said := range spoken {
		if said == "print here" {
			t.Fatalf("the skipped block was spoken: %q", spoken)
		}
	}
	if len(spoken) == 0 {
		t.Fatalf("nothing was spoken at all, so the skip proves nothing")
	}
}

// Not being spoken is not the same as being dropped: the block still reaches
// the pipeline, and it goes into the conversation as it stands, since nothing
// later will put it there.
func TestSkippedAggregationTypeStillReachesTheConversation(t *testing.T) {
	_, down := runWithSkippedType(t, true, "Here is code <code>print here</code> done.")

	var skipped *frames.AggregatedTextFrame
	for _, f := range down {
		if af, ok := f.(*frames.AggregatedTextFrame); ok && af.AggregatedBy == codeType {
			skipped = af
		}
	}
	if skipped == nil {
		t.Fatalf("the skipped block never reached the pipeline: %v", down)
	}
	if skipped.Text != "print here" {
		t.Errorf("skipped text = %q, want %q", skipped.Text, "print here")
	}
	if !skipped.AppendToContext {
		t.Error("a block that is never spoken has to go into the conversation as it stands")
	}
	if skipped.WillBeSpoken {
		t.Error("a skipped block is marked as one that will be spoken")
	}
}

// Left off the skip list, the block is spoken like any other unit, which is what
// says the test above measures the setting rather than the aggregator.
func TestUnskippedAggregationTypeIsSynthesized(t *testing.T) {
	spoken, _ := runWithSkippedType(t, false, "Here is code <code>print here</code> done.")

	var found bool
	for _, said := range spoken {
		if said == "print here" {
			found = true
		}
	}
	if !found {
		t.Errorf("spoken = %q, want the block among them", spoken)
	}
}

// The skipped block lands after the audio it follows, not ahead of it: the
// sequencer holds it until whatever was spoken before it has been said. Only a
// service that times its words emits them through the sequencer, so it is the
// one that can order a skipped frame against them.
func TestSkippedAggregationTypeLandsInOrder(t *testing.T) {
	agg := ttstext.NewPatternPairAggregator(frames.AggregationSentence, newTokenizer(t))
	if err := agg.AddPattern(codeType, "<code>", "</code>", ttstext.MatchAggregate); err != nil {
		t.Fatal(err)
	}

	syn := &fakeTimedSynth{rate: 24000, words: []timedWord{
		{text: "Hello", offset: 0, pcm: []byte{1, 2, 3, 4}},
		{text: "there.", offset: 0.1, pcm: []byte{5, 6, 7, 8}},
	}}
	base := tts.New("SkipOrderTTS", syn)
	base.SetTextAggregator(agg)
	base.SetSkipAggregatorTypes(codeType)

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	var down []frames.Frame
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.AggregatedTextFrame, *frames.TTSTextFrame:
			mu.Lock()
			down = append(down, f)
			mu.Unlock()
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there. <code>print here</code>"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	lastWord, skipped := -1, -1
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.TTSTextFrame:
			lastWord = i
		case *frames.AggregatedTextFrame:
			if fr.AggregatedBy == codeType {
				skipped = i
			}
		}
	}
	if skipped < 0 {
		t.Fatalf("the skipped block never reached the pipeline: %v", down)
	}
	if lastWord < 0 {
		t.Fatalf("no spoken word reached the pipeline, so the ordering proves nothing: %v", down)
	}
	if skipped < lastWord {
		t.Errorf("the skipped block landed at %d, ahead of the last spoken word at %d", skipped, lastWord)
	}
}
