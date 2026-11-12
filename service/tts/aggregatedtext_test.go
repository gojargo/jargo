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

// collectAggregated runs a TTS base over feed and returns every
// AggregatedTextFrame that reached the pipeline.
func collectAggregated(
	t *testing.T, aggregator ttstext.Aggregator, feed func(task *pipeline.Worker),
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
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if af, ok := f.(*frames.AggregatedTextFrame); ok {
			mu.Lock()
			got = append(got, af)
			mu.Unlock()
		}
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

// A service passing tokens straight through announces what is about to be
// spoken once, as the sentence the tokens make, rather than token by token.
func TestAggregatedTextInTokenModeIsTheSentence(t *testing.T) {
	agg := ttstext.NewSimpleAggregator(frames.TextTypeToken, "")
	got := collectAggregated(t, agg, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	var announced []*frames.AggregatedTextFrame
	for _, f := range got {
		if f.WillBeSpoken {
			announced = append(announced, f)
		}
	}
	if len(announced) != 1 {
		t.Fatalf("announcements = %d, want one", len(announced))
	}
	if announced[0].TextType == frames.TextTypeToken || announced[0].Text != "Hello there." {
		t.Errorf("announced %q grouped by %q, want the sentence", announced[0].Text, announced[0].TextType)
	}
}

// A unit cut from a matched pattern carries the written form it came from, so a
// consumer recording the conversation has the delimiters the model wrote even
// though only the content is spoken.
func TestAggregatedTextCarriesTheTextItWasCutFrom(t *testing.T) {
	agg := ttstext.NewPatternPairAggregator(frames.TextTypeSentence, "")
	if err := agg.AddPattern("code", "<code>", "</code>", ttstext.MatchAggregate); err != nil {
		t.Fatal(err)
	}

	got := collectAggregated(t, agg, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Here is code <code>print here</code> done."))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	var code *frames.AggregatedTextFrame
	for _, f := range got {
		if f.TextType == "code" {
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

// TestASpeakFrameCarriesItsTextType checks the text a TTSSpeakFrame is spoken
// as carries the frame's text type, so a custom type such as "status" can be
// told apart from the bot's replies, and kept from the client.
func TestASpeakFrameCarriesItsTextType(t *testing.T) {
	got := collectAggregated(t, nil, func(task *pipeline.Worker) {
		speak := frames.NewTTSSpeakFrame("One moment.")
		speak.TextType = "status"
		task.QueueFrame(speak)
	})

	if len(got) == 0 {
		t.Fatal("no aggregated text reached the pipeline")
	}
	if got[0].TextType != "status" {
		t.Errorf("text type = %q, want the speak frame's status", got[0].TextType)
	}
}

// TestASpeakFrameIsASentenceByDefault checks a TTSSpeakFrame left at its
// default is spoken as a sentence.
func TestASpeakFrameIsASentenceByDefault(t *testing.T) {
	got := collectAggregated(t, nil, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	})

	if len(got) == 0 || got[0].TextType != frames.TextTypeSentence {
		t.Errorf("aggregated = %v, want a sentence", got)
	}
}

// TestABotBackchannelIsSpokenAsAnUninterruptibleBackchannel checks a
// BotBackchannelFrame is spoken as backchannel text, which interruptions do not
// cut off and which stays out of the conversation, in token mode too.
func TestABotBackchannelIsSpokenAsAnUninterruptibleBackchannel(t *testing.T) {
	for _, mode := range []frames.TextType{frames.TextTypeSentence, frames.TextTypeToken} {
		t.Run(string(mode), func(t *testing.T) {
			got := collectAggregated(t, ttstext.NewSimpleAggregator(mode, ""), func(task *pipeline.Worker) {
				task.QueueFrame(frames.NewBotBackchannelFrame("Mm-hmm."))
			})

			if len(got) == 0 {
				t.Fatal("no aggregated text reached the pipeline")
			}
			for _, f := range got {
				if f.TextType != frames.TextTypeBackchannel || frames.Interruptible(f) || f.AppendToContext {
					t.Errorf("%v: type %q, interruptible %t, append to context %t; want an uninterruptible "+
						"backchannel kept out of the context", f, f.TextType, frames.Interruptible(f), f.AppendToContext)
				}
			}
		})
	}
}

// TestABotBackchannelMadeInterruptibleIsSpokenSo checks a backchannel made
// interruptible is spoken as interruptible text.
func TestABotBackchannelMadeInterruptibleIsSpokenSo(t *testing.T) {
	got := collectAggregated(t, nil, func(task *pipeline.Worker) {
		f := frames.NewBotBackchannelFrame("Mm-hmm.")
		f.SetInterruptible(true)
		task.QueueFrame(f)
	})

	if len(got) == 0 || !frames.Interruptible(got[0]) {
		t.Errorf("aggregated = %v, want interruptible text", got)
	}
}
