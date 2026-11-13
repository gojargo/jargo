package stt

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// muteWatcher records whether an STTMuteFrame got past the service.
type muteWatcher struct {
	mu     sync.Mutex
	passed bool
}

func (w *muteWatcher) observe(_ context.Context, f frames.Frame) {
	if _, ok := f.(*frames.STTMuteFrame); ok {
		w.mu.Lock()
		w.passed = true
		w.mu.Unlock()
	}
}

func (w *muteWatcher) sawMute() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.passed
}

// runWatched runs svc with every frame reaching the end of the pipeline handed
// to observe, until the returned stop is called.
func runWatched(
	t *testing.T, p *pipeline.Pipeline, observe func(context.Context, frames.Frame),
) (task *pipeline.Worker, stop func()) {
	t.Helper()
	task = pipeline.NewWorker(p, pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, observe)
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

// Audio reaching a muted service is not sent to the provider, so it is neither
// transcribed nor billed, and the audio after the unmute is sent again. The mute
// is consumed by the service.
func TestStreamServiceSendsNothingWhileMuted(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	svc := NewStream("FakeSTT", &kaConnector{stream: st}, 16000)
	w := &muteWatcher{}
	task, stop := runWatched(t, pipeline.New(svc), w.observe)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewSTTMuteFrame(true))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	waitFor(t, "the mute to apply", func() bool { return svc.Muted() })
	task.QueueFrame(frames.NewSTTMuteFrame(false))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	waitFor(t, "the audio after the unmute", func() bool { return len(st.sends()) > 0 })

	sends := st.sends()
	if len(sends) != 1 || len(sends[0]) != 640 {
		t.Errorf("sent chunks of %v bytes, want only the 640 after the unmute", chunkSizes(sends))
	}
	if svc.Muted() {
		t.Error("service still muted after the unmute")
	}
	if w.sawMute() {
		t.Error("the STTMuteFrame went past the service, want it consumed")
	}
}

// Audio dropped by the mute is not activity on the session, so a provider that
// asks for a keepalive still gets one while the service is muted.
func TestStreamServiceKeepsAMutedSessionAlive(t *testing.T) {
	t.Parallel()

	st := &kaProtocolStream{}
	conn := &kaConnector{stream: st, wants: true, opts: KeepaliveOptions{
		Timeout:  60 * time.Millisecond,
		Interval: 5 * time.Millisecond,
	}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewSTTMuteFrame(true))
	for range 15 {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 64), 16000, 1))
		time.Sleep(20 * time.Millisecond)
	}

	if got := st.keepalives(); got == 0 {
		t.Error("no keepalive while muted: the dropped audio was counted as activity")
	}
	if got := len(st.sends()); got != 0 {
		t.Errorf("submitted %d chunks while muted, want 0", got)
	}
}

// The audio dropped while muted leaves a gap, so the audio after the unmute is
// resampled as the start of a fresh stream rather than carrying on from the end
// of the audio before the mute.
func TestMutingStartsTheResampledAudioAfresh(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	svc := NewStream("FakeSTT", &kaConnector{stream: st}, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	before, after := tone(24000, 100*time.Millisecond), tone(24000, 100*time.Millisecond)
	task.QueueFrame(frames.NewInputAudioRawFrame(before, 24000, 1))
	task.QueueFrame(frames.NewSTTMuteFrame(true))
	task.QueueFrame(frames.NewSTTMuteFrame(false))
	task.QueueFrame(frames.NewInputAudioRawFrame(after, 24000, 1))
	waitFor(t, "the audio after the unmute", func() bool { return len(st.sends()) == 2 })

	if got, want := st.sends()[1], resampled(t, after, 24000, 16000); !bytes.Equal(got, want) {
		t.Error("the audio after the unmute carries on from the audio before the mute, want a fresh stream")
	}
}

// A segmented service records the mute and consumes the frame.
func TestSegmentServiceConsumesTheMute(t *testing.T) {
	t.Parallel()

	svc := NewSegment("FakeSTT", &settingsTranscriber{}, 16000)
	w := &muteWatcher{}
	task, stop := runWatched(t, pipeline.New(svc), w.observe)
	defer stop()

	task.QueueFrame(frames.NewSTTMuteFrame(true))
	waitFor(t, "the mute to apply", func() bool { return svc.Muted() })
	task.QueueFrame(frames.NewSTTMuteFrame(false))
	waitFor(t, "the unmute to apply", func() bool { return !svc.Muted() })

	if w.sawMute() {
		t.Error("the STTMuteFrame went past the service, want it consumed")
	}
}

func chunkSizes(chunks [][]byte) []int {
	sizes := make([]int, 0, len(chunks))
	for _, c := range chunks {
		sizes = append(sizes, len(c))
	}
	return sizes
}
