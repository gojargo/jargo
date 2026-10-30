package tts

// Ported from upstream's TTS frame-ordering suite: speech marked uninterruptible
// is synthesized and played through an interruption, while the interruptible
// speech around it is dropped, and every frame it produces is marked
// uninterruptible so the transport keeps it too.

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// heldSynth holds every synthesis until release is closed, the way a provider
// that is slow to answer does. Inline, RunTTS blocks and then yields its audio;
// otherwise it returns at once and its receive loop appends the audio later.
type heldSynth struct {
	inline  bool
	release chan struct{}

	mu          sync.Mutex
	host        AudioContextHost
	interrupted []string
	requests    map[string]string
	wg          sync.WaitGroup
}

func newHeldSynth(inline bool) *heldSynth {
	return &heldSynth{inline: inline, release: make(chan struct{}), requests: map[string]string{}}
}

func (s *heldSynth) SampleRate() int { return alignSampleRate }

func (s *heldSynth) SetAudioContextHost(h AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *heldSynth) OnAudioContextInterrupted(_ context.Context, contextID string) {
	s.mu.Lock()
	s.interrupted = append(s.interrupted, contextID)
	s.mu.Unlock()
}

func (s *heldSynth) interruptedContexts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.interrupted)
}

func (s *heldSynth) contextOf(text string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests[text]
}

func (s *heldSynth) RunTTS(_ context.Context, text, contextID string, yield func(frames.Frame) error) error {
	s.mu.Lock()
	s.requests[text] = contextID
	host := s.host
	s.mu.Unlock()
	if s.inline {
		<-s.release
		return yield(frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1))
	}
	s.wg.Go(func() {
		<-s.release
		// A request made after the release still answers over the network, a
		// little after RunTTS returns. Answering at once could end the context
		// before the base has queued the request's text behind its audio.
		time.Sleep(10 * time.Millisecond)
		s.deliver(host, contextID, nil)
	})
	return nil
}

// deliver appends a context's words and audio and ends it, unless the context is
// gone already.
func (s *heldSynth) deliver(host AudioContextHost, contextID string, words []uctx.WordTiming) {
	if !host.AudioContextAvailable(contextID) {
		return
	}
	if len(words) > 0 {
		host.AddWordTimestamps(contextID, words, WordTimingOptions{})
	}
	host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1))
	host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
	host.RemoveAudioContext(contextID)
}

// heldWordSynth is a heldSynth that also times the words it speaks, reporting
// each request as one word.
type heldWordSynth struct{ *heldSynth }

func (s heldWordSynth) RunTTSTimed(
	_ context.Context, text, contextID string,
	_ func(frames.Frame) error,
	_ func([]uctx.WordTiming, WordTimingOptions) error,
) error {
	s.mu.Lock()
	s.requests[text] = contextID
	host := s.host
	s.mu.Unlock()
	s.wg.Go(func() {
		<-s.release
		s.deliver(host, contextID, []uctx.WordTiming{{Word: text, Offset: 0}})
	})
	return nil
}

func uninterruptibleSpeak(text string, appendToContext bool) *frames.TTSSpeakFrame {
	f := frames.NewTTSSpeakFrame(text)
	f.AppendToContext = appendToContext
	f.SetInterruptible(false)
	return f
}

// interruptibility lists, for each frame of type T, whether it is
// interruptible.
func interruptibility[T frames.Frame](down []frames.Frame) []bool {
	var out []bool
	for _, f := range down {
		if _, ok := f.(T); ok {
			out = append(out, frames.Interruptible(f))
		}
	}
	return out
}

// TestUninterruptibleSpeechSurvivesBeforeAudio covers uninterruptible speech
// interrupted before any of its audio arrived. Its synthesis finishes and plays,
// marked uninterruptible throughout, the provider is never told it was cut off,
// and the speech after the interruption is interruptible again.
func TestUninterruptibleSpeechSurvivesBeforeAudio(t *testing.T) {
	for _, delivery := range []string{"http", "websocket", "words", "tokens"} {
		t.Run(delivery, func(t *testing.T) {
			held := newHeldSynth(delivery == "http")
			var syn Synthesizer = held
			if delivery == "words" || delivery == "tokens" {
				syn = heldWordSynth{held}
			}
			base := New("HeldTTS", syn)
			if delivery == "tokens" {
				base.SetTextAggregator(ttstext.NewTokenAggregator())
			}

			down := runBase(t, base,
				uninterruptibleSpeak("Mm-hmm.", false),
				sleepFor(50*time.Millisecond),
				frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
				frames.NewInterruptionFrame(),
				sleepFor(100*time.Millisecond),
				releaseFrame(held.release),
				sleepFor(100*time.Millisecond),
				frames.NewTTSSpeakFrame("Next reply."),
				sleepFor(200*time.Millisecond),
			)
			held.wg.Wait()

			var texts []string
			for _, f := range down {
				if tf, ok := f.(*frames.TTSTextFrame); ok {
					texts = append(texts, tf.Text)
				}
			}
			if !slices.Equal(texts, []string{"Mm-hmm.", "Next reply."}) {
				t.Fatalf("spoken %q, want both utterances: %s", texts, frameNames(down))
			}
			want := []bool{false, true}
			if got := interruptibility[*frames.TTSTextFrame](down); !slices.Equal(got, want) {
				t.Fatalf("text frames interruptible %v, want %v", got, want)
			}
			if got := interruptibility[*frames.TTSAudioRawFrame](down); !slices.Equal(got, want) {
				t.Fatalf("audio frames interruptible %v, want %v", got, want)
			}
			if got := interruptibility[*frames.TTSStartedFrame](down); !slices.Equal(got, want) {
				t.Fatalf("started frames interruptible %v, want %v", got, want)
			}
			if got := interruptibility[*frames.TTSStoppedFrame](down); !slices.Equal(got, want) {
				t.Fatalf("stopped frames interruptible %v, want %v", got, want)
			}
			if got := held.interruptedContexts(); len(got) != 0 {
				t.Fatalf("the provider was told %v were interrupted", got)
			}
		})
	}
}

// TestQueuedUninterruptibleSynthesisSurvivesAnInterruptedContext covers
// uninterruptible speech queued behind interruptible speech when the user cuts
// in. The interruptible context is dropped and reported to the provider; the
// uninterruptible one keeps its place and plays.
func TestQueuedUninterruptibleSynthesisSurvivesAnInterruptedContext(t *testing.T) {
	held := newHeldSynth(false)
	base := New("QueuedTTS", held)

	down := runBase(t, base,
		frames.NewTTSSpeakFrame("Drop this."),
		uninterruptibleSpeak("Keep this.", true),
		sleepFor(50*time.Millisecond),
		frames.NewInterruptionFrame(),
		sleepFor(100*time.Millisecond),
		releaseFrame(held.release),
		sleepFor(100*time.Millisecond),
	)
	held.wg.Wait()

	if got := held.interruptedContexts(); !slices.Equal(got, []string{held.contextOf("Drop this.")}) {
		t.Fatalf("the provider was told %v were interrupted, want only the first utterance", got)
	}
	audio := audioOf(down)
	if len(audio) != 1 {
		t.Fatalf("%d audio frames, want 1: %s", len(audio), frameNames(down))
	}
	if audio[0].ContextID != held.contextOf("Keep this.") || frames.Interruptible(audio[0]) {
		t.Fatalf("audio on %q (interruptible %v), want the uninterruptible utterance's",
			audio[0].ContextID, frames.Interruptible(audio[0]))
	}
	kept := false
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok && tf.Text == "Keep this." {
			kept = true
		}
	}
	if !kept {
		t.Fatalf("the uninterruptible utterance was not recorded: %s", frameNames(down))
	}
}

// laterWordsSynth delivers each request's words and audio on its own receive
// loop shortly after the request, then ends the context.
type laterWordsSynth struct {
	mu    sync.Mutex
	host  AudioContextHost
	calls [][]uctx.WordTiming
	wg    sync.WaitGroup
}

func (s *laterWordsSynth) SampleRate() int { return alignSampleRate }

func (s *laterWordsSynth) SetAudioContextHost(h AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *laterWordsSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *laterWordsSynth) RunTTSTimed(
	_ context.Context, _, contextID string,
	_ func(frames.Frame) error,
	_ func([]uctx.WordTiming, WordTimingOptions) error,
) error {
	s.mu.Lock()
	host := s.host
	var words []uctx.WordTiming
	if len(s.calls) > 0 {
		words, s.calls = s.calls[0], s.calls[1:]
	}
	s.mu.Unlock()
	s.wg.Go(func() {
		time.Sleep(10 * time.Millisecond)
		host.AddWordTimestamps(contextID, words, WordTimingOptions{})
		host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1))
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

// TestWordsAfterAnInterruptionFollowUninterruptibleWordsStillToPlay covers an
// interruption landing after uninterruptible speech has been synthesized but
// before its last word is heard. Those words outlive the interruption, so the
// words spoken next are timed after them rather than over them.
func TestWordsAfterAnInterruptionFollowUninterruptibleWordsStillToPlay(t *testing.T) {
	synth := &laterWordsSynth{calls: [][]uctx.WordTiming{
		{{Word: "Keep", Offset: 0}, {Word: "this.", Offset: 5}},
		{{Word: "Next.", Offset: 0}},
	}}
	base := New("LaterTTS", synth)

	next := frames.NewTTSSpeakFrame("Next.")
	next.AppendToContext = false
	down := runBase(t, base,
		uninterruptibleSpeak("Keep this.", false),
		// Synthesis finishes long before "this." plays.
		sleepFor(100*time.Millisecond),
		frames.NewInterruptionFrame(),
		sleepFor(50*time.Millisecond),
		next,
		sleepFor(100*time.Millisecond),
	)
	synth.wg.Wait()

	var words []*frames.TTSTextFrame
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			words = append(words, tf)
		}
	}
	if len(words) != 3 || words[0].Text != "Keep" || words[1].Text != "this." || words[2].Text != "Next." {
		t.Fatalf("words = %s, want Keep, this., Next.", frameNames(down))
	}
	this, _ := words[1].PTS()
	after, _ := words[2].PTS()
	if after < this {
		t.Fatalf("%q is timed at %d, before %q at %d which is still to play", "Next.", after, "this.", this)
	}
}

// silentSynth opens its context and never delivers anything, so the drain loop
// sits waiting on it while frames queue up behind.
type silentSynth struct{}

func (silentSynth) SampleRate() int                      { return alignSampleRate }
func (silentSynth) SetAudioContextHost(AudioContextHost) {}
func (silentSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

// markerFrame is a control frame queued behind speech to see whether it
// survives an interruption.
type markerFrame struct {
	frames.BaseControlFrame
	label string
}

func newMarkerFrame(label string, interruptible bool) *markerFrame {
	f := &markerFrame{BaseControlFrame: frames.NewBaseControlFrame("MarkerFrame"), label: label}
	f.SetInterruptible(interruptible)
	return f
}

// TestSerializationQueuePreservesUninterruptibleFramesOnInterruption covers
// frames queued behind speech that is still waiting for its audio. An
// interruption drops the interruptible one and keeps the uninterruptible one,
// which the restarted drain goroutine then pushes on.
func TestSerializationQueuePreservesUninterruptibleFramesOnInterruption(t *testing.T) {
	speak := frames.NewTTSSpeakFrame("hello")
	speak.AppendToContext = false

	down := runBase(t, New("SilentTTS", silentSynth{}),
		speak,
		sleepFor(50*time.Millisecond),
		newMarkerFrame("will_be_dropped", true),
		newMarkerFrame("must_survive", false),
		sleepFor(50*time.Millisecond),
		frames.NewInterruptionFrame(),
		sleepFor(100*time.Millisecond),
	)

	var labels []string
	for _, f := range down {
		if m, ok := f.(*markerFrame); ok {
			labels = append(labels, m.label)
		}
	}
	if !slices.Equal(labels, []string{"must_survive"}) {
		t.Fatalf("markers reaching the output = %q, want only the uninterruptible one", labels)
	}
}

// releaseFrame is not sent: runBase closes ch when it reaches it, releasing a
// held synthesis at that point in the sequence.
func releaseFrame(ch chan struct{}) *sleepFrame {
	return &sleepFrame{release: ch}
}
