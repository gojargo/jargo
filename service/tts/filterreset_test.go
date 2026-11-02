package tts_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// TestFilterStateDoesNotCarryIntoTheNextReply checks a stateful filter is reset
// when a reply ends: a code block one reply left open must not mute the next.
func TestFilterStateDoesNotCarryIntoTheNextReply(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 8)}
	base := tts.New("FilterTTS", syn)
	opts := ttstext.DefaultMarkdownFilterOptions() //nolint:staticcheck // the deprecated filter under test
	opts.FilterCode = true
	base.SetTextFilters(ttstext.NewMarkdownFilter(opts)) //nolint:staticcheck // the deprecated filter under test

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		task.Cancel(context.Background(), "")
		<-done
	})

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("In Markdown, type ``` to start a code block."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("How else can I help?"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	deadline := time.After(3 * time.Second)
	for {
		select {
		case text := <-syn.spoken:
			if text == "How else can I help?" {
				return
			}
		case <-deadline:
			t.Fatal("the next reply was never spoken: the open code block muted it")
		}
	}
}

// TestATransformLeavingOnlyWhitespaceLeavesTheTextUnspoken checks a transform
// whose output is only whitespace drops the unit, rather than sending a lone
// space to the synthesizer, and that the sentences around it are still spoken.
func TestATransformLeavingOnlyWhitespaceLeavesTheTextUnspoken(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 8)}
	base := tts.New("TransformTTS", syn)
	base.AddTextTransformer(tts.TextTransformer{
		TextType: frames.AnyTextType,
		Transform: func(_ context.Context, text string, _ frames.TextType) (string, error) {
			if text == "Skip me." {
				return " ", nil
			}
			return text, nil
		},
	})

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		task.Cancel(context.Background(), "")
		<-done
	})

	for _, text := range []string{"Skip me.", "Say this."} {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame(text))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	}

	select {
	case text := <-syn.spoken:
		if text != "Say this." {
			t.Errorf("first text synthesized = %q, want %q", text, "Say this.")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("nothing was synthesized")
	}
}
