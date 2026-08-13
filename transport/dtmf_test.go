package transport_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/dtmf"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
)

// startFakeOutput runs o in a task and returns the task plus a stop.
func startFakeOutput(t *testing.T, o *fakeOutput) (*pipeline.Worker, func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("the output task did not shut down")
		}
	}
}

// collectPCM drains writes until it has at least n bytes or gives up.
func collectPCM(t *testing.T, o *fakeOutput, n int) []byte {
	t.Helper()
	var pcm []byte
	deadline := time.After(3 * time.Second)
	for len(pcm) < n {
		select {
		case chunk := <-o.writes:
			pcm = append(pcm, chunk...)
		case <-deadline:
			t.Fatalf("only %d of %d bytes were written", len(pcm), n)
		}
	}
	return pcm
}

// TestOutputDTMFIsSoundedAsAudio checks a keypress reaches a transport that
// cannot signal one natively as the tone for that key. Without this a bot cannot
// answer an IVR at all: the frame was defined and nothing ever played it.
func TestOutputDTMFIsSoundedAsAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	o := newFakeOutput(params)
	task, stop := startFakeOutput(t, o)
	defer stop()

	task.QueueFrame(frames.NewOutputDTMFFrame(frames.KeypadSeven))

	want, err := dtmf.Tone(frames.KeypadSeven, params.AudioOutSampleRate)
	if err != nil {
		t.Fatalf("Tone: %v", err)
	}
	got := collectPCM(t, o, len(want))
	if len(got) < len(want) {
		t.Fatalf("wrote %d bytes, want at least the %d of one tone", len(got), len(want))
	}
}

// TestOutputDTMFSequenceKeepsItsOrder checks a run of keys is sounded in the
// order it was pressed, since a number entered out of order is a different
// number.
func TestOutputDTMFSequenceKeepsItsOrder(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	o := newFakeOutput(params)
	task, stop := startFakeOutput(t, o)
	defer stop()

	keys := []frames.KeypadEntry{frames.KeypadOne, frames.KeypadTwo, frames.KeypadPound}
	task.QueueFrame(frames.NewOutputDTMFSequenceFrame(keys))

	one, err := dtmf.Tone(frames.KeypadOne, params.AudioOutSampleRate)
	if err != nil {
		t.Fatalf("Tone: %v", err)
	}
	got := collectPCM(t, o, len(one)*len(keys))
	if len(got) < len(one)*len(keys) {
		t.Fatalf("wrote %d bytes, want the %d of three tones", len(got), len(one)*len(keys))
	}
}

// nativeDTMFOutput reports that it signals keypresses itself, so the base must
// hand them over rather than sounding them.
type nativeDTMFOutput struct {
	*transport.BaseOutput
	keys chan []frames.KeypadEntry
	pcm  chan []byte
}

func newNativeDTMFOutput(p transport.Params) *nativeDTMFOutput {
	o := &nativeDTMFOutput{
		keys: make(chan []frames.KeypadEntry, 8),
		pcm:  make(chan []byte, 64),
	}
	o.BaseOutput = transport.NewBaseOutput("NativeDTMFOutput", p, o)
	return o
}

func (o *nativeDTMFOutput) SupportsNativeDTMF() bool { return true }

func (o *nativeDTMFOutput) WriteDTMFNative(_ context.Context, f frames.DTMFOutput) error {
	select {
	case o.keys <- f.Keys():
	default:
	}
	return nil
}

func (o *nativeDTMFOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	cp := append([]byte(nil), f.AudioData().Audio...)
	select {
	case o.pcm <- cp:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// TestNativeDTMFBypassesTheAudioPath checks a transport whose protocol carries
// keypresses is handed them, and does not also hear them as audio. Sounding the
// tone as well would put it into anything recording the call.
func TestNativeDTMFBypassesTheAudioPath(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	o := newNativeDTMFOutput(params)
	task, stop := startFakeOutput2(t, o)
	defer stop()

	task.QueueFrame(frames.NewOutputDTMFFrame(frames.KeypadFour))

	select {
	case keys := <-o.keys:
		if frames.KeypadString(keys) != "4" {
			t.Errorf("keys = %q, want 4", frames.KeypadString(keys))
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never handed the keypress")
	}

	select {
	case pcm := <-o.pcm:
		t.Errorf("wrote %d bytes of audio for a natively signaled keypress", len(pcm))
	case <-time.After(300 * time.Millisecond):
	}
}

// TestUrgentDTMFGoesOutAtOnce checks the urgent frame is sent without waiting
// behind the audio already queued, which is what a keypress answering a prompt
// that is still playing needs.
func TestUrgentDTMFGoesOutAtOnce(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	o := newNativeDTMFOutput(params)
	task, stop := startFakeOutput2(t, o)
	defer stop()

	task.QueueFrame(frames.NewOutputDTMFUrgentFrame(frames.KeypadStar))

	select {
	case keys := <-o.keys:
		if frames.KeypadString(keys) != "*" {
			t.Errorf("keys = %q, want *", frames.KeypadString(keys))
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the urgent keypress never reached the transport")
	}
}

// startFakeOutput2 runs a native-DTMF output in a task.
func startFakeOutput2(t *testing.T, o *nativeDTMFOutput) (*pipeline.Worker, func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("the output task did not shut down")
		}
	}
}
