package tts

// Text reaches the conversation only once audio shows it was spoken. A context
// that produces no audio adds none of its text, and frames queued behind text
// still waiting for its audio keep their place.
//
// Ported from upstream's tests for the same behavior. Its two tests of a skipped
// code block sent to the service as an AggregatedTextFrame are not: this
// service does not yet speak or skip an incoming AggregatedTextFrame by its
// type, and queues it as it is.

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// noAudioStopTimeout is how long a context in these tests waits for more audio
// before it is taken as finished.
const noAudioStopTimeout = 100 * time.Millisecond

// utteranceSynth answers inline, the way an HTTP provider does, with audio only
// for the chosen utterances. Every other request is accepted and answered with
// nothing, like a provider given a voice it doesn't know.
type utteranceSynth struct {
	mu      sync.Mutex
	n       int
	speakOn map[int]bool
}

func (s *utteranceSynth) SampleRate() int { return alignSampleRate }

func (s *utteranceSynth) RunTTS(_ context.Context, _, contextID string, yield func(frames.Frame) error) error {
	s.mu.Lock()
	s.n++
	speak := s.speakOn[s.n]
	s.mu.Unlock()
	if !speak {
		return nil
	}
	f := frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1)
	f.ContextID = contextID
	return yield(f)
}

// lateAudioSynth stands in for a WebSocket provider without word timestamps,
// whose audio arrives later on a receive loop of its own. The spoken text is
// queued as soon as RunTTS returns, ahead of its audio.
type lateAudioSynth struct {
	speaks bool
	host   AudioContextHost
	wg     sync.WaitGroup
}

func (s *lateAudioSynth) SampleRate() int { return alignSampleRate }

func (s *lateAudioSynth) SetAudioContextHost(h AudioContextHost) { s.host = h }

func (s *lateAudioSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	if !s.speaks {
		return nil
	}
	s.wg.Go(func() {
		time.Sleep(noAudioStopTimeout / 2)
		f := frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	})
	return nil
}

func newNoAudioBase(syn Synthesizer) *Base {
	b := New("NoAudioTTS", syn)
	b.stopFrameTimeout = noAudioStopTimeout
	return b
}

// speakEach speaks each text in its own context, waiting for each to complete.
func speakEach(texts ...string) []frames.Frame {
	var out []frames.Frame
	for _, text := range texts {
		out = append(out, frames.NewTTSSpeakFrame(text), sleepFor(3*noAudioStopTimeout))
	}
	return out
}

// contextText is the spoken text that reaches the conversation.
func contextText(down []frames.Frame) []string {
	var out []string
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok && tf.AppendToContext {
			out = append(out, tf.Text)
		}
	}
	return out
}

func hasTTSText(down []frames.Frame) bool {
	return slices.ContainsFunc(down, func(f frames.Frame) bool {
		_, ok := f.(*frames.TTSTextFrame)
		return ok
	})
}

func TestASilentContextAddsNoTextToTheContext(t *testing.T) {
	down := runBase(t, newNoAudioBase(&utteranceSynth{}), speakEach("Hello there.")...)

	if hasTTSText(down) {
		t.Errorf("text reached the context: %q", contextText(down))
	}
}

func TestASpokenContextAddsItsTextToTheContext(t *testing.T) {
	syn := &utteranceSynth{speakOn: map[int]bool{1: true}}
	down := runBase(t, newNoAudioBase(syn), speakEach("Hello there.")...)

	if got := strings.TrimSpace(strings.Join(contextText(down), "")); got != "Hello there." {
		t.Errorf("context text = %q, want %q", got, "Hello there.")
	}
}

func TestTextQueuedBeforeItsAudioIsPushedAheadOfIt(t *testing.T) {
	syn := &lateAudioSynth{speaks: true}
	down := runBase(t, newNoAudioBase(syn), speakEach("Hello there.")...)
	syn.wg.Wait()

	if got := contextText(down); !slices.Equal(got, []string{"Hello there."}) {
		t.Fatalf("context text = %q, want the sentence", got)
	}
	text := slices.IndexFunc(down, func(f frames.Frame) bool { _, ok := f.(*frames.TTSTextFrame); return ok })
	audio := slices.IndexFunc(down, func(f frames.Frame) bool { _, ok := f.(*frames.TTSAudioRawFrame); return ok })
	if audio < 0 || text > audio {
		t.Errorf("text at %d, audio at %d: want the text ahead of its audio", text, audio)
	}
}

// The first silent context writes the service off, so the second, which would
// have produced audio, is never synthesized.
func TestAnUnusableServiceAddsNoTextToTheContext(t *testing.T) {
	b := newNoAudioBase(&utteranceSynth{speakOn: map[int]bool{2: true}})
	b.SetZeroAudioContextLimit(1)

	down := runBase(t, b, speakEach("one", "two")...)

	if b.Usable() {
		t.Error("the service is still usable after a silent context")
	}
	if hasTTSText(down) {
		t.Errorf("text reached the context: %q", contextText(down))
	}
}
