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

// collectTextAggregation runs a TTS base over feed and returns every text
// aggregation time it reported.
func collectTextAggregation(
	t *testing.T,
	metricsEnabled bool,
	aggregator ttstext.Aggregator,
	feed func(task *pipeline.Task),
) []frames.TextAggregationMetricsData {
	t.Helper()

	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 8)}
	go func() {
		for range syn.spoken {
		}
	}()
	base := tts.New("AggTTS", syn)
	if aggregator != nil {
		base.SetTextAggregator(aggregator)
	}

	var mu sync.Mutex
	var got []frames.TextAggregationMetricsData
	off := false
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		EnableMetrics:           metricsEnabled,
		SendInitialEmptyMetrics: &off,
		OnReachedDownstream: func(f frames.Frame) {
			mf, ok := f.(*frames.MetricsFrame)
			if !ok {
				return
			}
			mu.Lock()
			for _, d := range mf.Data {
				if agg, ok := d.(frames.TextAggregationMetricsData); ok {
					got = append(got, agg)
				}
			}
			mu.Unlock()
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
	return append([]frames.TextAggregationMetricsData(nil), got...)
}

// speakOneSentence feeds a turn whose text completes a sentence.
func speakOneSentence(task *pipeline.Task) {
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("world. "))
	task.QueueFrame(frames.NewLLMTextFrame("And more."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
}

func TestTextAggregationMeasuredPerSentence(t *testing.T) {
	got := collectTextAggregation(t, true, nil, speakOneSentence)

	// One report for the sentence the response completed.
	if len(got) != 1 {
		t.Fatalf("got %d text aggregation reports, want 1: %+v", len(got), got)
	}
	if got[0].Processor == "" {
		t.Error("report does not name the processor that measured it")
	}
	if got[0].Value <= 0 {
		t.Errorf("aggregation time = %v, want a positive duration", got[0].Value)
	}
}

func TestTextAggregationMeasuredFromTheFirstToken(t *testing.T) {
	// Text keeps arriving while the sentence is being waited for. The wait runs
	// from the first of it, so a later token must not push the clock forward.
	const gap = 200 * time.Millisecond

	got := collectTextAggregation(t, true, nil, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Hello "))
		time.Sleep(gap)
		task.QueueFrame(frames.NewLLMTextFrame("world. "))
		// A following non-whitespace character is what confirms the boundary.
		task.QueueFrame(frames.NewLLMTextFrame("Next"))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) == 0 {
		t.Fatal("no text aggregation reported")
	}
	if got[0].Value < gap {
		t.Errorf("aggregation time = %v, want at least %v: it must run from the first token, not the last",
			got[0].Value, gap)
	}
}

func TestTextAggregationNotMeasuredWhenMetricsDisabled(t *testing.T) {
	if got := collectTextAggregation(t, false, nil, speakOneSentence); len(got) != 0 {
		t.Errorf("got %d reports with metrics off, want none: %+v", len(got), got)
	}
}

func TestTextAggregationNotMeasuredWhenStreamingTokens(t *testing.T) {
	// Grouping by token completes no sentences, so there is no aggregation delay
	// to measure and none is reported, not even at the end of the response.
	tok, err := ttstext.NewPunktEnglish()
	if err != nil {
		t.Fatalf("NewPunktEnglish: %v", err)
	}
	byToken := ttstext.NewSimpleAggregator(frames.AggregationToken, tok)

	if got := collectTextAggregation(t, true, byToken, speakOneSentence); len(got) != 0 {
		t.Errorf("got %d reports while streaming tokens, want none: %+v", len(got), got)
	}
}

func TestTextAggregationReportedWhenNoSentenceCompletes(t *testing.T) {
	// A response that never terminates a sentence still gets measured, when the
	// end of the response flushes what is left.
	got := collectTextAggregation(t, true, nil, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("no terminator here"))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if len(got) != 1 {
		t.Fatalf("got %d reports, want 1: %+v", len(got), got)
	}
}
