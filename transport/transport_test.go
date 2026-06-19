package transport_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
)

// fakeInput is an input transport whose StartReading emits a fixed number of
// audio frames.
type fakeInput struct {
	*transport.BaseInput
	frames int
}

func newFakeInput(p transport.Params, n int) *fakeInput {
	in := &fakeInput{frames: n}
	in.BaseInput = transport.NewBaseInput("FakeInput", p, in)
	return in
}

func (in *fakeInput) StartReading(ctx context.Context) error {
	go func() {
		for range in.frames {
			f := frames.NewInputAudioRawFrame(make([]byte, 1920), 48000, 1)
			in.PushAudioFrame(ctx, f)
		}
	}()
	return nil
}

func TestBaseInputPushesAudioDownstream(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000

	var got atomic.Int32
	done := make(chan struct{})
	taskParams := pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.InputAudioRawFrame); ok {
				if got.Add(1) == 3 {
					close(done)
				}
			}
		},
	}

	in := newFakeInput(params, 3)
	task := pipeline.NewTask(pipeline.New(in), taskParams)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatalf("only %d of 3 audio frames reached downstream", got.Load())
	}

	task.StopWhenDone()
	<-runDone
}

// fakeOutput is an output transport that records the audio chunks it is asked
// to send.
type fakeOutput struct {
	*transport.BaseOutput
	writes chan []byte
}

func newFakeOutput(p transport.Params) *fakeOutput {
	o := &fakeOutput{writes: make(chan []byte, 64)}
	o.BaseOutput = transport.NewBaseOutput("FakeOutput", p, o)
	return o
}

func (o *fakeOutput) WriteAudio(_ context.Context, pcm []byte) error {
	cp := make([]byte, len(pcm))
	copy(cp, pcm)
	o.writes <- cp
	return nil
}

func TestBaseOutputChunksAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // chunk size = 480 samples/10ms * 2 * 2 = 1920 bytes

	o := newFakeOutput(params)
	task := pipeline.NewTask(pipeline.New(o), pipeline.TaskParams{})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Two chunks worth of audio in a single frame.
	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 3840), 48000, 1))

	for i := range 2 {
		select {
		case chunk := <-o.writes:
			if len(chunk) != 1920 {
				t.Fatalf("chunk %d = %d bytes, want 1920", i, len(chunk))
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("timed out waiting for chunk %d", i)
		}
	}

	task.StopWhenDone()
	<-runDone
}
