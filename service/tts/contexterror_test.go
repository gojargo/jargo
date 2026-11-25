package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// erroringSynth reports a failure in its stream, then answers with audio.
type erroringSynth struct{}

func (erroringSynth) SampleRate() int { return 24000 }

func (erroringSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	if err := yield(frames.NewErrorFrame("one event failed")); err != nil {
		return err
	}
	return yield(frames.NewTTSAudioRawFrame(make([]byte, 480), 24000, 1))
}

// TestAnErrorInTheStreamIsPushedAsAnError checks an error a provider yields
// among its audio is pushed upstream as the error it is when its turn in the
// context comes, rather than sent on downstream with the audio, and that the
// audio around it still plays.
func TestAnErrorInTheStreamIsPushedAsAnError(t *testing.T) {
	var mu sync.Mutex
	var down, up []frames.Frame
	task := pipeline.NewWorker(pipeline.New(tts.New("ErroringTTS", erroringSynth{})), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()

	var upErrors []string
	for _, f := range up {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			upErrors = append(upErrors, ef.Error)
		}
	}
	if len(upErrors) != 1 || upErrors[0] != "one event failed" {
		t.Errorf("errors upstream = %q, want the one yielded", upErrors)
	}
	audio := 0
	for _, f := range down {
		switch f.(type) {
		case *frames.ErrorFrame:
			t.Errorf("an error went downstream: %v", f)
		case *frames.TTSAudioRawFrame:
			audio++
		}
	}
	if audio == 0 {
		t.Error("the audio after the error was not played")
	}
}
