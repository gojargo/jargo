package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// fakeSynth records the text it was asked to speak and emits fixed PCM.
type fakeSynth struct {
	rate   int
	chunk  []byte
	spoken chan string
}

func (s *fakeSynth) SampleRate() int { return s.rate }

func (s *fakeSynth) Synthesize(_ context.Context, text string, emit func(pcm []byte) error) error {
	s.spoken <- text
	return emit(s.chunk)
}

// runTTS wires a fake synthesizer into a task and records the downstream frame
// sequence and audio sample rates.
func runTTS(t *testing.T, syn *fakeSynth, feed func(task *pipeline.Task)) []string {
	t.Helper()
	var mu sync.Mutex
	var seq []string
	stopped := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(tts.New("FakeTTS", syn)), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.TTSStartedFrame:
				seq = append(seq, "started")
			case *frames.TTSAudioRawFrame:
				seq = append(seq, "audio")
				if fr.SampleRate != syn.rate {
					t.Errorf("audio sample rate = %d, want %d", fr.SampleRate, syn.rate)
				}
			case *frames.TTSStoppedFrame:
				seq = append(seq, "stopped")
				select {
				case stopped <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	feed(task)

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("TTS base did not finish a synthesis")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return seq
}

func TestSynthesizesCompletedSentence(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 1)}
	seq := runTTS(t, syn, func(task *pipeline.Task) {
		// Split across frames; synthesis fires once the sentence terminates.
		task.QueueFrame(frames.NewLLMTextFrame("Hello "))
		task.QueueFrame(frames.NewLLMTextFrame("world."))
	})

	if got := <-syn.spoken; got != "Hello world." {
		t.Fatalf("synthesized text = %q, want %q", got, "Hello world.")
	}
	want := []string{"started", "audio", "stopped"}
	if !equal(seq, want) {
		t.Fatalf("frame sequence = %v, want %v", seq, want)
	}
}

func TestFlushSynthesizesTrailingText(t *testing.T) {
	syn := &fakeSynth{rate: 16000, chunk: []byte{9}, spoken: make(chan string, 1)}
	seq := runTTS(t, syn, func(task *pipeline.Task) {
		// No sentence terminator: only the end-of-response flush speaks it.
		task.QueueFrame(frames.NewTextFrame("no period here"))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if got := <-syn.spoken; got != "no period here" {
		t.Fatalf("synthesized text = %q, want %q", got, "no period here")
	}
	want := []string{"started", "audio", "stopped"}
	if !equal(seq, want) {
		t.Fatalf("frame sequence = %v, want %v", seq, want)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
