package processor_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// TestTextTransformerKeepsFrameTypeAndFields checks every kind of text frame is
// rewritten in place: an LLM token stays one, with its spacing flag, and a
// transcription stays one, with its user.
func TestTextTransformerKeepsFrameTypeAndFields(t *testing.T) {
	up := processor.NewStatelessTextTransformer("Upper", strings.ToUpper)
	token := frames.NewLLMTextFrame("hello")
	heard := frames.NewTranscriptionFrame("hi", "user1", "now")

	got := collectDownstream(t, []processor.Processor{up}, func(task *pipeline.Worker) {
		task.QueueFrames([]frames.Frame{token, heard})
		time.Sleep(50 * time.Millisecond)
	})

	var gotToken *frames.LLMTextFrame
	var gotHeard *frames.TranscriptionFrame
	for _, f := range got {
		switch fr := f.(type) {
		case *frames.LLMTextFrame:
			gotToken = fr
		case *frames.TranscriptionFrame:
			gotHeard = fr
		}
	}
	if gotToken == nil || gotToken.Text != "HELLO" || !gotToken.IncludesInterFrameSpaces {
		t.Errorf("LLM token = %+v, want it rewritten and still an LLM token with its spacing", gotToken)
	}
	if gotHeard == nil || gotHeard.Text != "HI" || gotHeard.UserID != "user1" {
		t.Errorf("transcription = %+v, want it rewritten and still carrying its user", gotHeard)
	}
}

// bouncer sends each text frame it is given back upstream.
type bouncer struct{ *processor.Base }

func (b *bouncer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream {
		return b.PushFrame(ctx, frames.NewTextFrame(tf.Text), processor.Upstream)
	}
	return b.PushFrame(ctx, f, dir)
}

// TestTextTransformerKeepsDirection checks text traveling upstream is rewritten
// and keeps going upstream. The bouncer sends the text back after the
// transformer, so what reaches the start of the pipeline went through it both
// ways.
func TestTextTransformerKeepsDirection(t *testing.T) {
	exclaim := processor.NewStatelessTextTransformer("Exclaim", func(s string) string { return s + "!" })
	b := &bouncer{}
	b.Base = processor.New("Bouncer", b)

	var mu sync.Mutex
	var reached []string
	task := pipeline.NewWorker(pipeline.New(exclaim, b),
		pipeline.WorkerConfig{ReachedUpstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TextFrame); ok {
			mu.Lock()
			reached = append(reached, tf.Text)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTextFrame("hello"))
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	<-done

	mu.Lock()
	defer mu.Unlock()
	if len(reached) != 1 || reached[0] != "hello!!" {
		t.Errorf("reached upstream = %q, want the text rewritten on its way back too", reached)
	}
}
