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

// TestSentenceAggregatorGathersLLMTokens checks the tokens an LLM streams are
// gathered, which is what the aggregator is placed after an LLM for, and the
// unfinished rest goes out at the end.
func TestSentenceAggregatorGathersLLMTokens(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrames([]frames.Frame{
		frames.NewLLMTextFrame("Hello,"), frames.NewLLMTextFrame(" world."), frames.NewLLMTextFrame(" Bye"),
	})
	task.StopWhenDone()
	<-runDone

	var got []string
	for _, f := range drainAfterGrace(seen) {
		switch fr := f.(type) {
		case *frames.TextFrame:
			got = append(got, fr.Text)
		case *frames.LLMTextFrame:
			t.Errorf("an LLM token went past ungathered: %q", fr.Text)
		}
	}
	if len(got) != 2 || got[0] != "Hello, world." || got[1] != " Bye" {
		t.Errorf("text = %q, want the sentence, then the rest at the end", got)
	}
}

// TestSentenceAggregatorGathersTranscriptions checks a final transcription is
// gathered and an eager one, which the final one will repeat, is dropped.
func TestSentenceAggregatorGathersTranscriptions(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewEagerTranscriptionFrame("book a", "user-1", "ts"))
	task.QueueFrame(frames.NewTranscriptionFrame("Book a flight.", "user-1", "ts"))

	got := collectText(seen)
	if len(got) != 1 || got[0] != "Book a flight." {
		t.Errorf("text = %q, want the final transcription as a sentence", got)
	}

	task.StopWhenDone()
	<-runDone
}

// TestSentenceAggregatorDropsAHalfSentenceOnAnInterruption checks a sentence cut
// off by an interruption is dropped rather than joined to the next one.
func TestSentenceAggregatorDropsAHalfSentenceOnAnInterruption(t *testing.T) {
	task, seen, runDone := runProcessor(t, aggregators.NewSentence("Sentence"))

	task.QueueFrame(frames.NewLLMTextFrame("Let me check the"))
	awaitNothing(seen)
	task.QueueFrame(frames.NewInterruptionFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Okay."))

	got := collectText(seen)
	if len(got) != 1 || got[0] != "Okay." {
		t.Errorf("text = %q, want only the sentence said after the interruption", got)
	}

	task.StopWhenDone()
	<-runDone
}
