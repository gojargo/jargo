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

const outSampleRate = 16000

// outSynth speaks one short burst of audio per request.
type outSynth struct{}

func (outSynth) SampleRate() int { return outSampleRate }

func (outSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	return yield(frames.NewTTSAudioRawFrame(make([]byte, 320), outSampleRate, 1))
}

// speakAndCollect speaks one line through svc and returns the frames that
// reached the end of the pipeline, in order.
func speakAndCollect(t *testing.T, svc *tts.Base) []frames.Frame {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []frames.Frame
	)
	stopped := make(chan struct{})
	var once sync.Once

	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			seen = append(seen, f)
			mu.Unlock()
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				once.Do(func() { close(stopped) })
			}
		},
	})

	done := make(chan struct{})
	go func() { _ = task.Run(context.Background()); close(done) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("hello there"))
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("the utterance never finished")
	}
	task.Cancel()
	<-done

	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), seen...)
}

// TestSilenceAfterStopPadsTheUtterance checks the padding is sent, and sent
// before the frame that says the utterance stopped: a transport that stops
// sending the moment audio runs out would otherwise clip the last word.
func TestSilenceAfterStopPadsTheUtterance(t *testing.T) {
	svc := tts.New("PlainTTS", outSynth{})
	svc.SetSilenceAfterStop(tts.SilenceOptions{Enabled: true, Duration: 100 * time.Millisecond})

	got := speakAndCollect(t, svc)

	// 100ms of 16 kHz 16-bit mono.
	wantBytes := int(0.1*outSampleRate) * 2
	var padding *frames.TTSAudioRawFrame
	stoppedAt := -1
	paddingAt := -1
	for i, f := range got {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			if len(fr.Audio) == wantBytes {
				padding, paddingAt = fr, i
			}
		case *frames.TTSStoppedFrame:
			if stoppedAt < 0 {
				stoppedAt = i
			}
		}
	}

	if padding == nil {
		t.Fatalf("no padding of %d bytes was sent", wantBytes)
	}
	if padding.SampleRate != outSampleRate {
		t.Errorf("padding sample rate = %d, want %d", padding.SampleRate, outSampleRate)
	}
	if stoppedAt >= 0 && paddingAt > stoppedAt {
		t.Error("the padding arrived after the utterance was called finished, want it before")
	}
}

// TestNoSilenceByDefault checks a service pads nothing unless asked.
func TestNoSilenceByDefault(t *testing.T) {
	svc := tts.New("PlainTTS", outSynth{})

	for _, f := range speakAndCollect(t, svc) {
		if fr, ok := f.(*frames.TTSAudioRawFrame); ok && len(fr.Audio) != 320 {
			t.Errorf("unexpected audio of %d bytes, want only what was spoken", len(fr.Audio))
		}
	}
}

// TestDestinationAddressesTheAudio checks the frames a service produces name the
// transport stream they are meant for, so a transport carrying several can tell
// them apart.
func TestDestinationAddressesTheAudio(t *testing.T) {
	svc := tts.New("PlainTTS", outSynth{})
	svc.SetDestination("caller-a")

	var addressed int
	for _, f := range speakAndCollect(t, svc) {
		switch f.(type) {
		case *frames.TTSStartedFrame, *frames.TTSStoppedFrame,
			*frames.TTSAudioRawFrame, *frames.TTSTextFrame:
			if got := f.Base().TransportDestination(); got != "caller-a" {
				t.Errorf("%T destination = %q, want caller-a", f, got)
				continue
			}
			addressed++
		}
	}
	if addressed == 0 {
		t.Error("no frame was addressed to the destination")
	}
}
