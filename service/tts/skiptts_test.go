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
	task := pipeline.NewWorker(
		pipeline.New(tts.New("SkipTTS", synth)),
		pipeline.WorkerConfig{
			ReachedDownstreamFilter: pipeline.AnyFrame,
		},
	)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
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
// so the conversation is built from it whether or not it was spoken. A reply
// that is not spoken arrives cut into sentences, each an AggregatedTextFrame
// marked as skipping synthesis, which a client shows as the bot's output.
func hasText(got []frames.Frame, want string) bool {
	for _, f := range got {
		if af, ok := f.(*frames.AggregatedTextFrame); ok && af.Text == want &&
			af.SkipTTS != nil && *af.SkipTTS && af.AppendToContext {
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
// not spoken. Its text still travels on, cut into sentences, which is what puts
// the reply in the conversation and before a client: on the spoken path the
// service replaces it with the words it actually said, and there are none here.
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

// runAggregated pushes a response whose text arrives already aggregated (as an
// LLMText processor ahead of the service sends it) through a TTS service, and
// reports what reached the synthesizer and what reached the end of the
// pipeline.
func runAggregated(
	t *testing.T, skipTypes []frames.TextType, units ...*frames.AggregatedTextFrame,
) (spoken []string, reached []frames.Frame) {
	t.Helper()

	synth := &recordingSynth{}
	svc := tts.New("AggregatedTTS", synth)
	svc.SetSkipTextTypes(skipTypes...)
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(
		pipeline.New(svc),
		pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame},
	)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	queued := []frames.Frame{frames.NewLLMFullResponseStartFrame()}
	for _, u := range units {
		queued = append(queued, u)
	}
	task.QueueFrames(append(queued, frames.NewLLMFullResponseEndFrame()))

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

// TestAggregatedTextIsSpoken checks text that reaches the service already
// aggregated is spoken as the unit it is, rather than passed on unspoken.
func TestAggregatedTextIsSpoken(t *testing.T) {
	spoken, got := runAggregated(t, nil, frames.NewAggregatedTextFrame("Hello there.", frames.TextTypeSentence))
	if len(spoken) != 1 || spoken[0] != "Hello there." {
		t.Fatalf("spoken = %q, want the unit as it arrived", spoken)
	}
	started := false
	for _, f := range got {
		if _, ok := f.(*frames.TTSStartedFrame); ok {
			started = true
		}
	}
	if !started {
		t.Error("no TTSStartedFrame: the unit was not synthesized")
	}
}

// TestAggregatedTextStampedToSkipIsNotSpoken checks a unit stamped to skip TTS
// travels on as it is, unspoken.
func TestAggregatedTextStampedToSkipIsNotSpoken(t *testing.T) {
	skip := true
	unit := frames.NewAggregatedTextFrame("Not said.", frames.TextTypeSentence)
	unit.SkipTTS = &skip
	spoken, got := runAggregated(t, nil, unit)
	if len(spoken) != 0 {
		t.Errorf("spoken = %q, want nothing synthesized", spoken)
	}
	found := false
	for _, f := range got {
		if f == unit {
			found = true
		}
	}
	if !found {
		t.Error("the stamped unit did not travel on")
	}
}

// TestAggregatedTextOfASkippedTypeIsNotSpoken checks a unit grouped a way the
// service does not speak (a code block, say) travels on unspoken, marked as
// left out, after the spoken text before it.
func TestAggregatedTextOfASkippedTypeIsNotSpoken(t *testing.T) {
	code := frames.NewAggregatedTextFrame("x := 1", "code")
	spoken, got := runAggregated(t, []frames.TextType{"code"},
		frames.NewAggregatedTextFrame("Here it is.", frames.TextTypeSentence), code)
	if len(spoken) != 1 || spoken[0] != "Here it is." {
		t.Errorf("spoken = %q, want only the sentence", spoken)
	}
	skipped := false
	for _, f := range got {
		if af, ok := f.(*frames.AggregatedTextFrame); ok && af.Text == "x := 1" {
			skipped = af.SkipTTS != nil && *af.SkipTTS
		}
	}
	if !skipped {
		t.Error("the code unit did not travel on marked as left out")
	}
}
