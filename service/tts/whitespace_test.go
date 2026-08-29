package tts_test

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// speakUnits feeds each of texts as its own model token in one turn and returns
// the units the provider was actually asked to speak. With feedAsSpeak set the
// texts are sent as fixed utterances instead, which skips the aggregator.
func speakUnits(t *testing.T, agg ttstext.Aggregator, feedAsSpeak bool, texts ...string) []string {
	t.Helper()

	syn := &spacedSynth{}
	base := tts.New("GatingTTS", syn)
	if agg != nil {
		base.SetTextAggregator(agg)
	}
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})

	stopped := make(chan struct{}, 1)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.TTSStoppedFrame); ok {
			select {
			case stopped <- struct{}{}:
			default:
			}
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	if feedAsSpeak {
		for _, text := range texts {
			task.QueueFrame(frames.NewTTSSpeakFrame(text))
		}
	} else {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		for _, text := range texts {
			task.QueueFrame(frames.NewLLMTextFrame(text))
		}
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	}

	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("nothing was ever spoken")
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	return syn.texts()
}

func equalUnits(got, want []string) bool {
	return slices.Equal(got, want)
}

// Streaming tokens, the text is sent exactly as the model wrote it. The space
// opening the third token holds it apart from the second, and a provider that
// neither inserts nor strips whitespace between the units it is sent would run
// the words together without it.
func TestStreamedTokensAreSentAsWritten(t *testing.T) {
	agg := ttstext.NewSimpleAggregator(frames.AggregationToken, newTokenizer(t))
	got := speakUnits(t, agg, false, "Unbelieva", "ble", " isn't it?")

	want := []string{"Unbelieva", "ble", " isn't it?"}
	if !equalUnits(got, want) {
		t.Errorf("spoken = %q, want %q", got, want)
	}
}

// Whitespace between two words is what holds them apart, so once a token
// carrying something has been sent, a token of nothing but whitespace goes to
// the provider rather than being dropped as empty.
func TestWhitespaceBetweenTokensIsSpoken(t *testing.T) {
	agg := ttstext.NewSimpleAggregator(frames.AggregationToken, newTokenizer(t))
	got := speakUnits(t, agg, false, "Hi", " ", "there")

	want := []string{"Hi", " ", "there"}
	if !equalUnits(got, want) {
		t.Errorf("spoken = %q, want %q", got, want)
	}
}

// The whitespace opening a context attaches to nothing, so it comes off rather
// than being sent as a unit of its own or as a lead-in to the first word.
func TestWhitespaceOpeningAContextIsDropped(t *testing.T) {
	agg := ttstext.NewSimpleAggregator(frames.AggregationToken, newTokenizer(t))
	got := speakUnits(t, agg, false, "  ", "\n  Hi", " there")

	want := []string{"Hi", " there"}
	if !equalUnits(got, want) {
		t.Errorf("spoken = %q, want %q", got, want)
	}
}

// Aggregating sentences, a leading newline is layout rather than speech and
// comes off, but the spaces inside the sentence are left as they are.
func TestSentenceAggregationStripsLeadingNewlines(t *testing.T) {
	got := speakUnits(t, nil, true, "\n\nHello there.")

	want := []string{"Hello there."}
	if !equalUnits(got, want) {
		t.Errorf("spoken = %q, want %q", got, want)
	}
}

// A unit of nothing but whitespace is not worth speaking when sentences are
// what is being aggregated: there is no neighboring token for it to hold apart.
func TestWhitespaceOnlySentenceIsNotSpoken(t *testing.T) {
	got := speakUnits(t, nil, true, "   \n  ", "Hello there.")

	want := []string{"Hello there."}
	if !equalUnits(got, want) {
		t.Errorf("spoken = %q, want %q", got, want)
	}
}
