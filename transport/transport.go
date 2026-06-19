// Package transport defines the boundary between a pipeline and the outside
// world. A Transport exposes an input processor that turns received media into
// frames and an output processor that turns frames into sent media.
//
// This package holds the transport-agnostic base processors; a concrete
// transport (for example the Pion WebRTC transport in transport/pionrtc) embeds
// them and supplies the media I/O.
package transport

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Params configures a transport's audio input and output.
type Params struct {
	// AudioInEnabled enables receiving audio.
	AudioInEnabled bool
	// AudioInSampleRate is the input sample rate in Hz; 0 uses the StartFrame's.
	AudioInSampleRate int
	// AudioInChannels is the number of input channels.
	AudioInChannels int
	// AudioInPassthrough pushes received audio frames downstream.
	AudioInPassthrough bool

	// AudioOutEnabled enables sending audio.
	AudioOutEnabled bool
	// AudioOutSampleRate is the output sample rate in Hz; 0 uses the StartFrame's.
	AudioOutSampleRate int
	// AudioOutChannels is the number of output channels.
	AudioOutChannels int
	// AudioOutBitrate is the output bitrate in bits per second; 0 uses the
	// codec default.
	AudioOutBitrate int
	// AudioOut10msChunks is how many 10 ms chunks of audio are written at a
	// time. With WebRTC Opus this is 2, so audio is written in 20 ms frames.
	AudioOut10msChunks int
}

// DefaultParams returns Params with audio input and output enabled and the
// defaults a WebRTC transport uses: mono, input passthrough on, 20 ms output
// chunks.
func DefaultParams() Params {
	return Params{
		AudioInEnabled:     true,
		AudioInChannels:    1,
		AudioInPassthrough: true,
		AudioOutEnabled:    true,
		AudioOutChannels:   1,
		AudioOut10msChunks: 2,
	}
}

// Transport is a source and sink of media for a pipeline. Input and Output
// return the processors that sit at the head and tail of the pipeline.
type Transport interface {
	// Input returns the processor that emits frames from received media.
	Input() processor.Processor
	// Output returns the processor that sends frames as media.
	Output() processor.Processor
}

// InputDriver is implemented by a concrete input transport so the base can
// start and stop the transport-specific media reading. A driver produces audio
// by calling BaseInput.PushAudioFrame.
type InputDriver interface {
	processor.Processor
	// StartReading begins reading media from the transport. It runs under ctx,
	// which is canceled when the transport stops.
	StartReading(ctx context.Context) error
	// StopReading stops reading media from the transport.
	StopReading(ctx context.Context) error
}

// OutputDriver is implemented by a concrete output transport so the base can
// hand it audio chunks and messages to send.
type OutputDriver interface {
	processor.Processor
	// WriteAudio sends one chunk of interleaved S16LE PCM over the transport.
	WriteAudio(ctx context.Context, pcm []byte) error
	// SendMessage sends an application message to the client (for example over
	// a data channel).
	SendMessage(ctx context.Context, data []byte) error
}

// audioFrameChanCap bounds the buffered audio channels between the media
// goroutines and the pipeline.
const audioFrameChanCap = 256

// sendAudio enqueues f on ch, dropping it if ctx is done.
func sendAudio[T frames.Frame](ctx context.Context, ch chan T, f T) {
	select {
	case ch <- f:
	case <-ctx.Done():
	}
}
