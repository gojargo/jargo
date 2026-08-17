package aggregators_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
)

// runProcessor runs one processor and collects the text frames that reach the
// end of the pipeline.
func runProcessor(t *testing.T, p processor.Processor) (*pipeline.Worker, chan frames.Frame, chan error) {
	t.Helper()
	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, seen, runDone
}

// drainAfterGrace returns every frame that reached the end of the pipeline
// within a short grace period, in the order it arrived.
func drainAfterGrace(seen chan frames.Frame) []frames.Frame {
	deadline := time.After(500 * time.Millisecond)
	var got []frames.Frame
	for {
		select {
		case f := <-seen:
			got = append(got, f)
		case <-deadline:
			return got
		}
	}
}

// awaitNothing gives the pipeline a moment to carry what was queued, for a test
// whose next frame must arrive after it.
func awaitNothing(seen chan frames.Frame) {
	deadline := time.After(200 * time.Millisecond)
	for {
		select {
		case <-seen:
		case <-deadline:
			return
		}
	}
}

// collectText drains the collected text frames after a short grace period, in
// the order they arrived.
func collectText(seen chan frames.Frame) []string {
	var got []string
	for _, f := range drainAfterGrace(seen) {
		if tf, ok := f.(*frames.TextFrame); ok {
			got = append(got, tf.Text)
		}
	}
	return got
}

func TestSentenceAggregatorHoldsUntilASentenceEnds(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewTextFrame("Hello,"))
	task.QueueFrame(frames.NewTextFrame(" world. "))

	got := collectText(seen)
	if len(got) != 1 || got[0] != "Hello, world. " {
		t.Errorf("text = %q, want the one whole sentence", got)
	}

	task.StopWhenDone()
	<-runDone
}

func TestSentenceAggregatorDropsInterimTranscriptions(t *testing.T) {
	// An interim transcription is a guess that will be revised, so folding it in
	// would repeat what the final one says.
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewInterimTranscriptionFrame("Hel", "user-1", "ts"))
	task.QueueFrame(frames.NewTextFrame("Hello. "))

	got := collectText(seen)
	if len(got) != 1 || got[0] != "Hello. " {
		t.Errorf("text = %q, want the final sentence alone", got)
	}

	task.StopWhenDone()
	<-runDone
}

func TestSentenceAggregatorFlushesWhatIsHeldAtTheEnd(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewTextFrame("half a thought"))
	task.StopWhenDone()
	<-runDone

	got := collectText(seen)
	if len(got) != 1 || got[0] != "half a thought" {
		t.Errorf("text = %q, want what was held to go out unfinished", got)
	}
}

func TestSentenceAggregatorForwardsWhatItDoesNotGather(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewTTSSpeakFrame("say this"))

	deadline := time.After(2 * time.Second)
	for {
		select {
		case f := <-seen:
			if _, ok := f.(*frames.TTSSpeakFrame); ok {
				task.StopWhenDone()
				<-runDone
				return
			}
		case <-deadline:
			task.StopWhenDone()
			<-runDone
			t.Fatal("a frame the aggregator does not gather never reached the end")
		}
	}
}

// TestSentenceAggregatorSplitsAStreamIntoSentences is upstream's own case: a
// paragraph fed word by word comes out one sentence at a time.
func TestSentenceAggregatorSplitsAStreamIntoSentences(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	for word := range strings.SplitSeq("Hello, world. How are you? I am fine!", " ") {
		task.QueueFrame(frames.NewTextFrame(word + " "))
	}

	got := collectText(seen)
	want := []string{"Hello, world. ", "How are you? ", "I am fine! "}
	if len(got) != len(want) {
		t.Fatalf("sentences = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sentence %d = %q, want %q", i, got[i], want[i])
		}
	}

	task.StopWhenDone()
	<-runDone
}
