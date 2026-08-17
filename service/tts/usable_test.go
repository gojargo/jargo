package tts_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// countingSynth counts how often it is asked to synthesize.
type countingSynth struct {
	calls atomic.Int64
}

func (s *countingSynth) SampleRate() int { return 24000 }

func (s *countingSynth) RunTTS(_ context.Context, _, _ string, yield func(f frames.Frame) error) error {
	s.calls.Add(1)
	emit := tts.PCMYielder(yield, s.SampleRate())
	return emit(make([]byte, 640))
}

// speakOnce runs one sentence through a TTS service and returns how many times
// the provider was asked to synthesize.
func speakOnce(t *testing.T, usable bool) int64 {
	t.Helper()
	syn := &countingSynth{}
	svc := tts.New("CountingTTS", syn)
	if !usable {
		svc.SetUsable(context.Background(), false)
	}

	// The turn completes whether or not anything was spoken, so waiting on the
	// end of the response is what tells us the service is done with it.
	done := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			select {
			case done <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the TTS service never finished the response")
	}
	task.StopWhenDone()
	<-runDone

	return syn.calls.Load()
}

func TestTextIsSynthesizedWhileTheServiceIsHealthy(t *testing.T) {
	if got := speakOnce(t, true); got != 1 {
		t.Errorf("synthesized %d times, want 1", got)
	}
}

func TestTextIsDroppedOnceTheServiceIsUnusable(t *testing.T) {
	if got := speakOnce(t, false); got != 0 {
		t.Errorf("synthesized %d times, want 0", got)
	}
}

func TestASynthesizerBroughtBackSpeaksAgain(t *testing.T) {
	syn := &countingSynth{}
	svc := tts.New("CountingTTS", syn)
	ctx := context.Background()

	svc.SetUsable(ctx, false)
	svc.SetUsable(ctx, true)

	if !svc.Usable() {
		t.Fatal("a service brought back should be able to take work again")
	}
}
