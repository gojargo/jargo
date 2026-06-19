package transport

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// BaseInput is the head of a pipeline: it turns received audio into
// InputAudioRawFrames and pushes them downstream. A concrete transport embeds
// it and implements InputDriver to supply the media reading; the driver calls
// PushAudioFrame for each chunk of audio it receives.
type BaseInput struct {
	*processor.Base
	params Params
	self   InputDriver

	sampleRate int

	mu          sync.Mutex
	paused      bool
	audioIn     chan *frames.InputAudioRawFrame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
}

// NewBaseInput builds a BaseInput. self is the embedding transport, used to
// dispatch StartReading/StopReading and to process frames.
func NewBaseInput(name string, params Params, self InputDriver) *BaseInput {
	bi := &BaseInput{params: params, self: self}
	bi.Base = processor.New(name, self)
	return bi
}

// SampleRate is the input sample rate in Hz, set when the transport starts.
func (bi *BaseInput) SampleRate() int { return bi.sampleRate }

// Params returns the transport parameters.
func (bi *BaseInput) Params() Params { return bi.params }

// StartReading is the default no-op; a concrete transport overrides it.
func (bi *BaseInput) StartReading(context.Context) error { return nil }

// StopReading is the default no-op; a concrete transport overrides it.
func (bi *BaseInput) StopReading(context.Context) error { return nil }

// PushAudioFrame queues a received audio frame to be pushed downstream. The
// driver calls it for each chunk of audio it reads from the transport.
func (bi *BaseInput) PushAudioFrame(ctx context.Context, f *frames.InputAudioRawFrame) {
	if !bi.params.AudioInEnabled {
		return
	}
	bi.mu.Lock()
	paused, ch := bi.paused, bi.audioIn
	bi.mu.Unlock()
	if paused || ch == nil {
		return
	}
	sendAudio(ctx, ch, f)
}

// PushTransportMessage pushes a message received from the client downstream as
// an InputTransportMessageFrame. A concrete transport calls it when an
// application message arrives (for example off a data channel).
func (bi *BaseInput) PushTransportMessage(ctx context.Context, raw []byte) {
	_ = bi.PushFrame(ctx, frames.NewInputTransportMessageFrame(raw), processor.Downstream)
}

// ProcessFrame handles the transport lifecycle and forwards frames.
func (bi *BaseInput) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := bi.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Push StartFrame before starting so every processor downstream is
		// initialized before audio begins to flow.
		if err := bi.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return bi.startStreaming(ctx, fr)
	case *frames.EndFrame:
		if err := bi.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		bi.stopStreaming(ctx)
		return nil
	case *frames.CancelFrame:
		bi.stopStreaming(ctx)
		return bi.PushFrame(ctx, f, dir)
	default:
		return bi.PushFrame(ctx, f, dir)
	}
}

// Cleanup stops the audio goroutine and the processor.
func (bi *BaseInput) Cleanup(ctx context.Context) error {
	bi.stopStreaming(ctx)
	return bi.Base.Cleanup(ctx)
}

func (bi *BaseInput) startStreaming(ctx context.Context, f *frames.StartFrame) error {
	bi.sampleRate = pick(bi.params.AudioInSampleRate, f.AudioInSampleRate)

	bi.mu.Lock()
	bi.paused = false
	bi.audioCtx, bi.audioCancel = context.WithCancel(ctx)
	bi.audioIn = make(chan *frames.InputAudioRawFrame, audioFrameChanCap)
	audioCtx := bi.audioCtx
	bi.mu.Unlock()

	bi.audioWG.Add(1)
	go bi.audioLoop(audioCtx)

	return bi.self.StartReading(audioCtx)
}

func (bi *BaseInput) stopStreaming(ctx context.Context) {
	_ = bi.self.StopReading(ctx)

	bi.mu.Lock()
	cancel := bi.audioCancel
	bi.audioCancel = nil
	bi.mu.Unlock()

	if cancel != nil {
		cancel()
		bi.audioWG.Wait()
	}
}

// audioLoop drains received audio frames and pushes them downstream.
func (bi *BaseInput) audioLoop(ctx context.Context) {
	defer bi.audioWG.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case f := <-bi.audioIn:
			if bi.params.AudioInPassthrough {
				_ = bi.PushFrame(ctx, f, processor.Downstream)
			}
		}
	}
}

func pick(a, b int) int {
	if a != 0 {
		return a
	}
	return b
}
