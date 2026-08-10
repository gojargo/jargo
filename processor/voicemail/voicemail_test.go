package voicemail_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/voicemail"
)

// harness runs a voicemail processor in a pipeline and records the text that
// reaches the far end.
type harness struct {
	task *pipeline.Task
	done chan error
	stop sync.Once

	mu   sync.Mutex
	text strings.Builder
}

func run(t *testing.T, cfg voicemail.Config) *harness {
	t.Helper()
	h := &harness{done: make(chan error, 1)}
	p := voicemail.New(cfg)
	h.task = pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if tf, ok := f.(*frames.LLMTextFrame); ok {
				h.mu.Lock()
				h.text.WriteString(tf.Text)
				h.mu.Unlock()
			}
		},
	})
	go func() { h.done <- h.task.Run(context.Background()) }()
	t.Cleanup(func() { h.shutdown(t) })
	return h
}

// shutdown ends the pipeline and waits for Run to return. It is idempotent so a
// test may drain the pipeline itself and still be cleaned up safely.
func (h *harness) shutdown(t *testing.T) {
	t.Helper()
	h.stop.Do(func() {
		h.task.StopWhenDone()
		select {
		case <-h.done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not finish")
		}
	})
}

// say streams text as LLM deltas and closes the response, the way an LLM
// service drives this processor.
func (h *harness) say(chunks ...string) {
	for _, c := range chunks {
		h.task.QueueFrame(frames.NewLLMTextFrame(c))
	}
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
}

// spoken waits for the forwarded text to settle and returns it.
func (h *harness) spoken(t *testing.T) string {
	t.Helper()
	h.shutdown(t)
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.text.String()
}

// signal is a one-shot callback recorder.
type signal struct {
	ch chan struct{}
}

func newSignal() *signal { return &signal{ch: make(chan struct{}, 8)} }

func (s *signal) fire() { s.ch <- struct{}{} }

// wait reports whether the callback fired within d.
func (s *signal) wait(d time.Duration) bool {
	select {
	case <-s.ch:
		return true
	case <-time.After(d):
		return false
	}
}

// count drains and reports how many times the callback fired, after allowing d
// for stragglers.
func (s *signal) count(d time.Duration) int {
	time.Sleep(d)
	n := 0
	for {
		select {
		case <-s.ch:
			n++
		default:
			return n
		}
	}
}

func TestDetectsVoicemail(t *testing.T) {
	vm, human := newSignal(), newSignal()
	h := run(t, voicemail.Config{
		OnVoicemailDetected:    vm.fire,
		OnConversationDetected: human.fire,
	})

	h.say("<voicemail></voicemail>")
	if !vm.wait(2 * time.Second) {
		t.Fatal("voicemail callback never fired")
	}
	if human.wait(100 * time.Millisecond) {
		t.Error("the conversation callback must not fire for a voicemail verdict")
	}
}

// TestDetectsHuman covers both tags that mean a person answered.
func TestDetectsHuman(t *testing.T) {
	for _, tag := range []string{"human", "conversation"} {
		t.Run(tag, func(t *testing.T) {
			vm, human := newSignal(), newSignal()
			h := run(t, voicemail.Config{
				OnVoicemailDetected:    vm.fire,
				OnConversationDetected: human.fire,
			})

			h.say("<" + tag + "></" + tag + ">")
			if !human.wait(2 * time.Second) {
				t.Fatal("conversation callback never fired")
			}
			if vm.wait(100 * time.Millisecond) {
				t.Error("the voicemail callback must not fire for a human verdict")
			}
		})
	}
}

// TestStripsDecisionTags checks the classification markers never reach the TTS.
func TestStripsDecisionTags(t *testing.T) {
	h := run(t, voicemail.Config{OnConversationDetected: func() {}})
	h.say("Hi there.", " <human></human>", " How can I help?")

	got := h.spoken(t)
	if strings.Contains(got, "<human>") || strings.Contains(got, "</human>") {
		t.Errorf("spoken text = %q, want the decision tags stripped", got)
	}
	if !strings.Contains(got, "Hi there.") || !strings.Contains(got, "How can I help?") {
		t.Errorf("spoken text = %q, want the surrounding speech kept", got)
	}
}

// TestTagSplitAcrossDeltas checks a tag broken across LLM tokens is still
// recognized — the normal case, since the model streams a few characters at a
// time.
func TestTagSplitAcrossDeltas(t *testing.T) {
	vm := newSignal()
	h := run(t, voicemail.Config{OnVoicemailDetected: vm.fire})

	h.say("Please leave a message", " <voice", "mail></voice", "mail>")
	if !vm.wait(2 * time.Second) {
		t.Fatal("a tag split across deltas was not detected")
	}
}

// TestDecidesOnce checks the verdict is final: a second tag in the same call
// must not fire another callback, since the app has already acted on the first.
func TestDecidesOnce(t *testing.T) {
	vm, human := newSignal(), newSignal()
	h := run(t, voicemail.Config{
		OnVoicemailDetected:    vm.fire,
		OnConversationDetected: human.fire,
	})

	h.say("<voicemail></voicemail>", "<voicemail></voicemail>", "<human></human>")

	if n := vm.count(500 * time.Millisecond); n != 1 {
		t.Errorf("voicemail callback fired %d times, want exactly 1", n)
	}
	if n := human.count(0); n != 0 {
		t.Errorf("conversation callback fired %d times after a voicemail verdict, want 0", n)
	}
}

// TestVoicemailDelay checks the callback is held back so the answering machine's
// greeting can finish before the bot starts leaving a message.
func TestVoicemailDelay(t *testing.T) {
	vm := newSignal()
	h := run(t, voicemail.Config{
		OnVoicemailDetected: vm.fire,
		VoicemailDelay:      300 * time.Millisecond,
	})

	h.say("<voicemail></voicemail>")
	if vm.wait(100 * time.Millisecond) {
		t.Error("the callback fired before the configured delay elapsed")
	}
	if !vm.wait(3 * time.Second) {
		t.Fatal("the delayed callback never fired")
	}
}

// TestFlushesIncompleteTag checks that text held back as a possible tag is
// released when the response ends, rather than being swallowed.
func TestFlushesIncompleteTag(t *testing.T) {
	h := run(t, voicemail.Config{})
	h.say("hello <voicem")

	if got := h.spoken(t); !strings.Contains(got, "<voicem") {
		t.Errorf("spoken text = %q, want the incomplete tag flushed rather than dropped", got)
	}
}

// TestNilCallbacks checks a config with no callbacks is safe — the processor is
// usable purely as a tag stripper.
func TestNilCallbacks(t *testing.T) {
	h := run(t, voicemail.Config{})
	h.say("<voicemail></voicemail>ok")

	if got := h.spoken(t); strings.Contains(got, "<voicemail>") {
		t.Errorf("spoken text = %q, want the tag stripped", got)
	}
}

// TestNonTextFramesPassThrough checks unrelated frames are forwarded untouched.
func TestNonTextFramesPassThrough(t *testing.T) {
	got := make(chan struct{}, 1)
	p := voicemail.New(voicemail.Config{})
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TTSSpeakFrame); ok {
				select {
				case got <- struct{}{}:
				default:
				}
			}
		},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("hello"))
	select {
	case <-got:
	case <-time.After(3 * time.Second):
		t.Fatal("an unrelated frame was not forwarded")
	}
	task.StopWhenDone()
	<-done
}

// TestCleanupStopsPendingTimer checks tearing the pipeline down cancels a
// pending delayed callback, so a hung-up call cannot fire it later.
func TestCleanupStopsPendingTimer(t *testing.T) {
	vm := newSignal()
	p := voicemail.New(voicemail.Config{
		OnVoicemailDetected: vm.fire,
		VoicemailDelay:      time.Second,
	})
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMTextFrame("<voicemail></voicemail>"))
	task.StopWhenDone()
	<-done

	if n := vm.count(1500 * time.Millisecond); n != 0 {
		t.Errorf("the delayed callback fired %d times after cleanup, want 0", n)
	}
}
