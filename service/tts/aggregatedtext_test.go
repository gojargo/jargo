package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// collectAggregated runs a TTS base over feed and returns every
// AggregatedTextFrame that reached the pipeline.
func collectAggregated(
	t *testing.T, aggregator ttstext.Aggregator, feed func(task *pipeline.Task),
) []*frames.AggregatedTextFrame {
	t.Helper()

	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 16)}
	go func() {
		for range syn.spoken {
		}
	}()
	base := tts.New("AggTTS", syn)
	if aggregator != nil {
		base.SetTextAggregator(aggregator)
	}

	var mu sync.Mutex
	var got []*frames.AggregatedTextFrame
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if af, ok := f.(*frames.AggregatedTextFrame); ok {
				mu.Lock()
				got = append(got, af)
				mu.Unlock()
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	feed(task)
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]*frames.AggregatedTextFrame(nil), got...)
}

func newTokenizer(t *testing.T) ttstext.SentenceTokenizer {
	t.Helper()
	tok, err := ttstext.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// The frame announcing what is about to be spoken says how the text was
// grouped. A service passing tokens straight through says so, rather than
// describing every token as a sentence.
func TestAggregatedTextCarriesHowItWasGrouped(t *testing.T) {
	agg := ttstext.NewSimpleAggregator(frames.AggregationToken, newTokenizer(t))
	got := collectAggregated(t, agg, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) == 0 {
		t.Fatal("no aggregated text reached the pipeline")
	}
	if got[0].AggregatedBy != frames.AggregationToken {
		t.Errorf("aggregated by = %q, want %q", got[0].AggregatedBy, frames.AggregationToken)
	}
	if got[0].RawText != "Hello there." {
		t.Errorf("raw text = %q, want the text as written", got[0].RawText)
	}
}

// A unit cut from a matched pattern carries the written form it came from, so a
// consumer recording the conversation has the delimiters the model wrote even
// though only the content is spoken.
func TestAggregatedTextCarriesTheTextItWasCutFrom(t *testing.T) {
	agg := ttstext.NewPatternPairAggregator(frames.AggregationSentence, newTokenizer(t))
	if err := agg.AddPattern("code", "<code>", "</code>", ttstext.MatchAggregate); err != nil {
		t.Fatal(err)
	}

	got := collectAggregated(t, agg, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Here is code <code>print here</code> done."))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	var code *frames.AggregatedTextFrame
	for _, f := range got {
		if f.AggregatedBy == "code" {
			code = f
		}
	}
	if code == nil {
		t.Fatalf("no frame carried the pattern's own type, got %+v", got)
	}
	if code.Text != "print here" {
		t.Errorf("text = %q, want the content between the delimiters", code.Text)
	}
	if code.RawText != "<code>print here</code>" {
		t.Errorf("raw text = %q, want the delimiters kept", code.RawText)
	}
}
