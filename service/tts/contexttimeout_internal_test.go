package tts

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// runBase drives base with send, honoring sleepFrames between them, and returns
// every frame that reached the end of the pipeline.
func runBase(t *testing.T, base *Base, send ...frames.Frame) []frames.Frame {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	var down []frames.Frame
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, f := range send {
		if sleep, ok := f.(*sleepFrame); ok {
			if sleep.release != nil {
				close(sleep.release)
			}
			time.Sleep(sleep.d)
			continue
		}
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), down...)
}

// slowSynth answers inline, the way an HTTP provider does, but only after delay,
// which can be longer than the audio context's idle timeout.
type slowSynth struct{ delay time.Duration }

func (s *slowSynth) SampleRate() int { return alignSampleRate }

func (s *slowSynth) RunTTS(_ context.Context, _, contextID string, yield func(frames.Frame) error) error {
	time.Sleep(s.delay)
	f := frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1)
	f.ContextID = contextID
	return yield(f)
}

// TestAudioAfterTheContextTimedOutIsStillPlayed covers a provider slower to
// send its first byte than the idle timeout. The drain loop takes the context
// as finished, but the turn is still being spoken, so the audio that arrives
// afterwards opens it again rather than being dropped, and the sentence is
// still heard and recorded.
func TestAudioAfterTheContextTimedOutIsStillPlayed(t *testing.T) {
	base := New("SlowTTS", &slowSynth{delay: 250 * time.Millisecond})
	base.stopFrameTimeout = 100 * time.Millisecond

	down := runBase(t, base,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("Hello there."),
		frames.NewLLMFullResponseEndFrame(),
		sleepFor(500*time.Millisecond),
	)

	audio, text, lastAudio, lastStop := 0, 0, -1, -1
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			audio++
			lastAudio = i
		case *frames.TTSTextFrame:
			if fr.Text == "Hello there." {
				text++
			}
		case *frames.TTSStoppedFrame:
			lastStop = i
		}
	}
	if audio != 1 {
		t.Fatalf("%d audio frames reached the output, want 1: %s", audio, frameNames(down))
	}
	if text != 1 {
		t.Fatalf("the sentence was recorded %d times, want once: %s", text, frameNames(down))
	}
	if lastStop < lastAudio {
		t.Fatalf("no stop frame follows the late audio: %s", frameNames(down))
	}
}

func frameNames(fs []frames.Frame) string {
	out := ""
	for _, f := range fs {
		out += f.Name() + " "
	}
	return out
}
