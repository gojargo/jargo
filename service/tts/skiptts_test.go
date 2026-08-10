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

const skipSampleRate = 16000

// recordingSynth records the text it was asked to speak, so a test can tell
// whether a turn reached synthesis at all.
type recordingSynth struct {
	mu     sync.Mutex
	spoken []string
}

func (s *recordingSynth) SampleRate() int { return skipSampleRate }

func (s *recordingSynth) RunTTS(_ context.Context, text, _ string, yield func(frames.Frame) error) error {
	s.mu.Lock()
	s.spoken = append(s.spoken, text)
	s.mu.Unlock()
	return yield(frames.NewTTSAudioRawFrame(make([]byte, 640), skipSampleRate, 1))
}

func (s *recordingSynth) texts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.spoken...)
}

// runResponse pushes one complete LLM response through a TTS service, stamped
// as skipStamp says, and reports what reached the synthesizer and what reached
// the end of the pipeline.
func runResponse(t *testing.T, skipStamp *bool) (spoken []string, reached []frames.Frame) {
	t.Helper()

	synth := &recordingSynth{}
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(
		pipeline.New(tts.New("SkipTTS", synth)),
		pipeline.TaskParams{ReachedDownstreamFilter: pipeline.AnyFrame, OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		}},
	)
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	start := frames.NewLLMFullResponseStartFrame()
	start.SkipTTS = skipStamp
	text := frames.NewLLMTextFrame("the reply")
	text.SkipTTS = skipStamp
	end := frames.NewLLMFullResponseEndFrame()
	end.SkipTTS = skipStamp
	task.QueueFrames([]frames.Frame{start, text, end})

	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return synth.texts(), append([]frames.Frame(nil), got...)
}

// hasText reports whether the response's text reached the end of the pipeline,
// so the conversation is built from it whether or not it was spoken.
func hasText(got []frames.Frame, want string) bool {
	for _, f := range got {
		if tf, ok := f.(*frames.LLMTextFrame); ok && tf.Text == want {
			return true
		}
	}
	return false
}

// TestUnstampedResponseIsSpoken is the control: a response nothing configured is
// synthesized as usual.
func TestUnstampedResponseIsSpoken(t *testing.T) {
	spoken, _ := runResponse(t, nil)
	if len(spoken) == 0 {
		t.Error("the response never reached the synthesizer")
	}
}

// TestStampedResponseSkipsSynthesis checks a response stamped to skip TTS is
// passed through instead of spoken. The text itself still travels on, which is
// what puts the reply in the conversation: on the spoken path the service
// replaces it with the words it actually said, and there are none here.
func TestStampedResponseSkipsSynthesis(t *testing.T) {
	skip := true
	spoken, got := runResponse(t, &skip)
	if len(spoken) != 0 {
		t.Errorf("the response was stamped to skip TTS but reached the synthesizer: %q", spoken)
	}
	if !hasText(got, "the reply") {
		t.Error("the response text did not reach the end of the pipeline")
	}
}

// TestStampedToSpeakIsSpoken checks a response stamped the other way is
// synthesized, so putting the configuration back restores speech.
func TestStampedToSpeakIsSpoken(t *testing.T) {
	skip := false
	spoken, _ := runResponse(t, &skip)
	if len(spoken) == 0 {
		t.Error("the response was stamped to be spoken but never reached the synthesizer")
	}
}
