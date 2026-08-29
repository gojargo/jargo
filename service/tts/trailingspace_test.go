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

// spacedSynth records the text it was handed and declares whether the provider
// behind it needs that text to end in a space.
type spacedSynth struct {
	requires bool

	mu   sync.Mutex
	said []string
}

func (s *spacedSynth) SampleRate() int { return 24000 }

func (s *spacedSynth) RequiresTrailingSpace() bool { return s.requires }

func (s *spacedSynth) RunTTS(
	_ context.Context, text, _ string, yield func(f frames.Frame) error,
) error {
	s.mu.Lock()
	s.said = append(s.said, text)
	s.mu.Unlock()
	return tts.PCMYielder(yield, s.SampleRate())([]byte{1, 2, 3, 4})
}

func (s *spacedSynth) texts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.said...)
}

// unspacedSynth says nothing about spacing at all, which is what most providers
// do. It deliberately does not implement tts.TrailingSpaceRequirer.
type unspacedSynth struct {
	mu   sync.Mutex
	said []string
}

func (s *unspacedSynth) SampleRate() int { return 24000 }

func (s *unspacedSynth) RunTTS(
	_ context.Context, text, _ string, yield func(f frames.Frame) error,
) error {
	s.mu.Lock()
	s.said = append(s.said, text)
	s.mu.Unlock()
	return tts.PCMYielder(yield, s.SampleRate())([]byte{1, 2, 3, 4})
}

func (s *unspacedSynth) texts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.said...)
}

// speakThrough runs one LLM turn carrying text through a base built on syn and
// returns the text the provider was actually asked to speak.
func speakThrough(t *testing.T, syn tts.Synthesizer, agg ttstext.Aggregator, text string) {
	t.Helper()

	base := tts.New("SpacingTTS", syn)
	if agg != nil {
		base.SetTextAggregator(agg)
	}
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})

	// A turn is finished once its audio has been heard, which is the point every
	// unit of it has been synthesized.
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

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame(text))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("the turn was never spoken")
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
}

// A provider that reads trailing punctuation aloud is handed text ending in a
// space, so the full stop closing a sentence is heard as a pause rather than
// spoken as "dot".
func TestTrailingSpaceIsAppendedForAProviderThatNeedsIt(t *testing.T) {
	syn := &spacedSynth{requires: true}
	speakThrough(t, syn, nil, "Hi there!")

	got := syn.texts()
	if len(got) != 1 || got[0] != "Hi there! " {
		t.Errorf("spoken = %q, want [\"Hi there! \"]", got)
	}
}

// Text that already ends in a space is left alone: appending a second one would
// change what the provider is given for no reason.
func TestTrailingSpaceIsNotDoubled(t *testing.T) {
	syn := &spacedSynth{requires: true}
	// The aggregator hands sentences over without their trailing space, so the
	// text is fed straight to the service instead.
	base := tts.New("SpacingTTS", syn)
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
	task.QueueFrame(frames.NewTTSSpeakFrame("Hi there! "))
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("the utterance was never spoken")
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	got := syn.texts()
	if len(got) != 1 || got[0] != "Hi there! " {
		t.Errorf("spoken = %q, want the text unchanged", got)
	}
}

// A provider that says nothing about spacing gets the text as it was aggregated.
func TestTrailingSpaceIsNotAppendedByDefault(t *testing.T) {
	syn := &unspacedSynth{}
	speakThrough(t, syn, nil, "Hi there!")

	got := syn.texts()
	if len(got) != 1 || got[0] != "Hi there!" {
		t.Errorf("spoken = %q, want [\"Hi there!\"]", got)
	}
}

// Streaming tokens, the text's own whitespace is what separates the words, so a
// space appended to every token would break them apart. The provider needing one
// between sentences does not get one between tokens.
func TestTrailingSpaceIsNotAppendedWhenStreamingTokens(t *testing.T) {
	syn := &spacedSynth{requires: true}
	agg := ttstext.NewSimpleAggregator(frames.AggregationToken, newTokenizer(t))
	speakThrough(t, syn, agg, "Hi there!")

	for _, got := range syn.texts() {
		if got != "Hi there!" {
			t.Errorf("spoken %q, want the token as written", got)
		}
	}
}
