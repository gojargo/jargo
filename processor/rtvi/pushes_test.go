package rtvi_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for which pushes the observer handles. Each drives the observer
// directly, as the pipeline would, with the pushes a frame makes on its way.

// pushHarness is an observer whose messages are collected as they reach the end
// of a running pipeline.
type pushHarness struct {
	observer *rtvi.Observer
	source   processor.Processor
	mu       sync.Mutex
	msgs     []rtvi.Message
}

func newPushHarness(t *testing.T, params rtvi.ObserverParams) *pushHarness {
	t.Helper()
	proc := rtvi.NewProcessor()
	h := &pushHarness{observer: rtvi.NewObserverWithParams(proc, params), source: newPlayback()}
	off := false
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{
		EnableRTVI:              &off,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				h.mu.Lock()
				h.msgs = append(h.msgs, msg)
				h.mu.Unlock()
			}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()
	t.Cleanup(func() {
		task.StopWhenDone()
		<-done
	})
	// Let the pipeline start before anything is sent through it.
	time.Sleep(50 * time.Millisecond)
	return h
}

// push reports one push of f, from source when it is given.
func (h *pushHarness) push(f frames.Frame, firstPush bool, source processor.Processor) {
	if source == nil {
		source = plainSource
	}
	h.observer.OnPushFrame(processor.FramePushed{
		Source:      source,
		Destination: source,
		Frame:       f,
		Direction:   processor.Downstream,
		FirstPush:   firstPush,
	})
}

// types is the types of the messages sent so far, once they have settled.
func (h *pushHarness) types() []string {
	time.Sleep(100 * time.Millisecond)
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]string, 0, len(h.msgs))
	for _, m := range h.msgs {
		out = append(out, m.Type)
	}
	return out
}

// plainSource is a processor that is not a transport's output end.
//
//nolint:gochecknoglobals // a stand-in source shared by the tests
var plainSource = processor.New("Plain", nil)

func TestObserverSkipsAudioUnlessAudioLevelsAreReported(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	h.push(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1), true, nil)
	h.push(frames.NewTTSAudioRawFrame(make([]byte, 320), 16000, 1), true, nil)
	if got := h.types(); len(got) != 0 {
		t.Fatalf("audio was reported with audio levels off: %v", got)
	}

	params := rtvi.DefaultObserverParams()
	params.UserAudioLevelEnabled = true
	h = newPushHarness(t, params)
	h.push(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1), true, nil)
	if got := h.types(); len(got) != 1 || got[0] != rtvi.TypeUserAudioLevel {
		t.Fatalf("got %v, want one user audio level", got)
	}
}

func TestObserverHandlesAFrameOnItsFirstPushOnly(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	f := frames.NewUserStartedSpeakingFrame()
	h.push(f, true, nil)
	h.push(f, false, nil)
	if got := h.types(); len(got) != 1 || got[0] != rtvi.TypeUserStartedSpeaking {
		t.Fatalf("got %v, want one user-started-speaking", got)
	}
}

func TestObserverNeverHandlesAFrameDisabledOnItsFirstPush(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	f := frames.NewVADUserStartedSpeakingFrame(0, time.Now())
	h.push(f, true, nil)
	on := true
	h.push(rtvi.NewConfigureObserverFrame(nil, &on, nil), true, nil)
	h.push(f, false, nil)
	if got := h.types(); len(got) != 0 {
		t.Fatalf("the frame was handled on a later push: %v", got)
	}
}

func TestObserverHandlesAggregatedTextOnceItHasGoneThroughTheTransport(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	f := frames.NewAggregatedTextFrame("hello", frames.AggregationSentence)

	h.push(f, true, nil)
	h.push(frames.NewBotStartedSpeakingFrame(), true, nil)
	for _, typ := range h.types() {
		if typ == rtvi.TypeBotOutput {
			t.Fatal("the text was reported before it had gone through the transport")
		}
	}

	h.push(f, false, h.source)
	got := h.types()
	n := 0
	for _, typ := range got {
		if typ == rtvi.TypeBotOutput {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("got %v, want the text reported once it had gone through the transport", got)
	}
}
