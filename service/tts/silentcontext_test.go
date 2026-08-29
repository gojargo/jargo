package tts_test

// Tests for writing off a TTS service that stops producing audio. A provider can
// accept every request and return no audio at all, an unknown voice id say,
// without reporting an error. The Base reports every context that completes in
// silence and, past a configurable limit, reports itself unable to do its job so
// the pipeline worker and any switcher can act on it.
//
// Ported from upstream's tests for the same behavior.

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// silenceDelivery is how long the fake provider takes to answer a request. It is
// short enough to keep the tests quick and long enough that the context is still
// open when the turn's text runs out, which is when a pause would be taken.
const silenceDelivery = 20 * time.Millisecond

// silentSynth stands in for a provider that answers on a receive loop of its own
// and returns no audio for most requests. It closes each context itself, which is
// what completes one with nothing in it: without that the context would wait out
// the stop-frame timeout instead.
type silentSynth struct {
	rate int
	host tts.AudioContextHost

	mu sync.Mutex
	// speakOn holds the 1-based positions of the utterances that do produce
	// audio. Every other one is answered with silence.
	speakOn map[int]bool
	n       int
	wg      sync.WaitGroup
}

func newSilentSynth(speakOn ...int) *silentSynth {
	s := &silentSynth{rate: 24000, speakOn: map[int]bool{}}
	for _, n := range speakOn {
		s.speakOn[n] = true
	}
	return s
}

func (s *silentSynth) SampleRate() int { return s.rate }

func (s *silentSynth) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

func (s *silentSynth) RunTTS(
	_ context.Context, _, contextID string, _ func(frames.Frame) error,
) error {
	s.mu.Lock()
	s.n++
	speak := s.speakOn[s.n]
	s.mu.Unlock()

	s.wg.Go(func() {
		time.Sleep(silenceDelivery)
		if speak {
			s.host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(
				[]byte{1, 2, 3, 4}, s.rate, 1))
		}
		s.host.RemoveAudioContext(contextID)
	})
	return nil
}

// silenceHarness runs a Base over a silentSynth and collects the errors it
// reports.
type silenceHarness struct {
	task *pipeline.Worker
	base *tts.Base
	syn  *silentSynth

	mu   sync.Mutex
	errs []*frames.ErrorFrame

	runDone chan error
}

func newSilenceHarness(t *testing.T, limit int, syn *silentSynth) *silenceHarness {
	t.Helper()

	base := tts.New("SilentTTS", syn)
	base.SetZeroAudioContextLimit(limit)

	h := &silenceHarness{base: base, syn: syn, runDone: make(chan error, 1)}
	h.task = pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&h.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			h.mu.Lock()
			h.errs = append(h.errs, ef)
			h.mu.Unlock()
		}
	})

	go func() { h.runDone <- h.task.Run(context.Background()) }()
	t.Cleanup(func() {
		h.task.Cancel(context.Background(), "")
		select {
		case <-h.runDone:
		case <-time.After(3 * time.Second):
			t.Error("task did not stop")
		}
		h.syn.wg.Wait()
	})
	return h
}

// speak sends each text as its own turn and waits for the context it opened to
// complete.
func (h *silenceHarness) speak(texts ...string) {
	for _, text := range texts {
		h.task.QueueFrame(frames.NewTTSSpeakFrame(text))
		time.Sleep(silenceDelivery * 5)
	}
}

func (h *silenceHarness) errors() []*frames.ErrorFrame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]*frames.ErrorFrame(nil), h.errs...)
}

// TestTheFirstSilentContextIsReported covers a turn that produces no speech.
// Nothing else marks the end of one that never played audio, so this error is
// all anything waiting on the bot to speak has to go on.
func TestTheFirstSilentContextIsReported(t *testing.T) {
	h := newSilenceHarness(t, 3, newSilentSynth())
	h.speak("one")

	if errs := h.errors(); len(errs) != 1 {
		t.Fatalf("errors = %d, want 1: %v", len(errs), errs)
	}
	if !h.base.Usable() {
		t.Error("one silent context wrote the service off, want it still usable")
	}
}

// TestSilenceUnderTheLimitLeavesTheServiceUsable checks each silent context is
// reported as it happens, so a first turn that never speaks is something
// application code can act on right away.
func TestSilenceUnderTheLimitLeavesTheServiceUsable(t *testing.T) {
	h := newSilenceHarness(t, 3, newSilentSynth())
	h.speak("one", "two")

	if !h.base.Usable() {
		t.Error("two silent contexts wrote the service off, want it still usable")
	}
	errs := h.errors()
	if len(errs) != 2 {
		t.Fatalf("errors = %d, want 2: %v", len(errs), errs)
	}
	for i, ef := range errs {
		if ef.Source == nil || ef.Source.Name() != h.base.Name() {
			t.Errorf("error %d came from %v, want the service", i, ef.Source)
		}
		if !ef.Source.Usable() {
			t.Errorf("error %d reported the service unusable", i)
		}
	}
}

// TestConsecutiveSilentContextsWriteOffTheService checks the context that
// reaches the limit reports the permanent error in place of the recoverable one,
// so each silent context is reported once.
func TestConsecutiveSilentContextsWriteOffTheService(t *testing.T) {
	h := newSilenceHarness(t, 2, newSilentSynth())
	h.speak("one", "two")

	if h.base.Usable() {
		t.Error("the service is still usable after reaching the limit")
	}
	errs := h.errors()
	if len(errs) != 2 {
		t.Fatalf("errors = %d, want 2: %v", len(errs), errs)
	}
	// Already written off by the time the last error is seen, which is what
	// tells application code the error is not a transient one.
	if errs[len(errs)-1].Source.Usable() {
		t.Error("the last error reported the service as still usable")
	}
}

// TestAudioResetsTheSilentCount puts silence either side of an utterance that
// does produce audio. Without the reset the two silent ones together would
// reach the limit.
func TestAudioResetsTheSilentCount(t *testing.T) {
	h := newSilenceHarness(t, 2, newSilentSynth(2))
	h.speak("silent", "spoken", "silent again")

	if !h.base.Usable() {
		t.Error("the run of silence was not reset by the utterance that spoke")
	}
	if errs := h.errors(); len(errs) != 2 {
		t.Fatalf("errors = %d, want 2, one per silent context: %v", len(errs), errs)
	}
}

// TestZeroLimitReportsSilenceWithoutWritingTheServiceOff checks silence is
// reported however long it goes on, but never costs the service its usability.
func TestZeroLimitReportsSilenceWithoutWritingTheServiceOff(t *testing.T) {
	h := newSilenceHarness(t, 0, newSilentSynth())
	h.speak("one", "two", "three", "four")

	if !h.base.Usable() {
		t.Error("a zero limit wrote the service off")
	}
	if errs := h.errors(); len(errs) != 4 {
		t.Fatalf("errors = %d, want 4: %v", len(errs), errs)
	}
}

// TestTheServiceIsWrittenOffOnce checks an unusable service is no longer given
// work, so its silent contexts say nothing new and are not reported again.
func TestTheServiceIsWrittenOffOnce(t *testing.T) {
	h := newSilenceHarness(t, 1, newSilentSynth())
	h.speak("one", "two", "three")

	if errs := h.errors(); len(errs) != 1 {
		t.Fatalf("errors = %d, want 1: %v", len(errs), errs)
	}
}

// TestBecomingUsableAgainClearsTheSilentCount checks the count starts over once
// whatever silenced the service has been dealt with.
func TestBecomingUsableAgainClearsTheSilentCount(t *testing.T) {
	h := newSilenceHarness(t, 2, newSilentSynth())
	h.speak("one")

	if errs := h.errors(); len(errs) != 1 {
		t.Fatalf("errors = %d, want 1: %v", len(errs), errs)
	}

	h.base.SetUsable(context.Background(), true)

	// With the count cleared, the next silent context is the first of a fresh
	// run and cannot reach a limit of two on its own.
	h.speak("two")
	if !h.base.Usable() {
		t.Error("the run of silence carried over the service being brought back")
	}
}

// TestAnInterruptedContextIsNotCounted covers a context cut off before it could
// produce anything. It is abandoned rather than completed, so it says nothing
// about whether the provider speaks.
func TestAnInterruptedContextIsNotCounted(t *testing.T) {
	h := newSilenceHarness(t, 1, newSilentSynth())

	h.task.QueueFrame(frames.NewTTSSpeakFrame("interrupted"))
	time.Sleep(silenceDelivery / 2)
	h.task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(silenceDelivery * 5)

	if !h.base.Usable() {
		t.Error("an interrupted context was counted against the service")
	}
	if errs := h.errors(); len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
}

// TestASilentContextLiftsThePauseItTook covers a context known to have played
// nothing releasing the pause taken for it. The pause is taken while the context
// is still open and might yet produce audio; nothing else would resume it once
// the context completes in silence.
func TestASilentContextLiftsThePauseItTook(t *testing.T) {
	syn := newSilentSynth()
	base := tts.New("SilentPausingTTS", syn)
	base.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	base.SetZeroAudioContextLimit(0)

	var mu sync.Mutex
	var seen []string
	task := pipeline.NewWorker(pipeline.New(base, newMarker(&mu, &seen)), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.Cancel(context.Background(), "")
		<-runDone
		syn.wg.Wait()
	}()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.QueueFrame(&markerFrame{
		BaseDataFrame: frames.NewBaseDataFrame("MarkerFrame"),
		label:         "after the silence",
	})

	waitForMarker(t, &mu, &seen, "frame handling stayed paused after a context completed with no audio")
}

// waitForMarker waits for a marker to reach the end of the pipeline, which is
// what says frame handling is running rather than paused.
func waitForMarker(t *testing.T, mu *sync.Mutex, seen *[]string, msg string) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		mu.Lock()
		got := len(*seen)
		mu.Unlock()
		if got > 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatal(msg)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// markerFrame marks how far frame handling has got. It is a data frame the
// service knows nothing about, so it travels through untouched rather than being
// taken as text to speak.
type markerFrame struct {
	frames.BaseDataFrame
	label string
}

// marker records the markers that get past the service, which is how a test
// tells frame handling is running rather than paused.
type marker struct {
	*processor.Base
	mu   *sync.Mutex
	seen *[]string
}

func newMarker(mu *sync.Mutex, seen *[]string) *marker {
	m := &marker{mu: mu, seen: seen}
	m.Base = processor.New("Marker", m)
	return m
}

func (m *marker) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := m.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if mf, ok := f.(*markerFrame); ok {
		m.mu.Lock()
		*m.seen = append(*m.seen, mf.label)
		m.mu.Unlock()
	}
	return m.PushFrame(ctx, f, dir)
}
