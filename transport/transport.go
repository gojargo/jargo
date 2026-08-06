// Package transport defines the boundary between a pipeline and the outside
// world. A Transport exposes an input processor that turns received media into
// frames and an output processor that turns frames into sent media.
//
// This package holds the transport-agnostic base processors; a concrete
// transport (for example the WebRTC transport in transport/rtc) embeds
// them and supplies the media I/O.
package transport

import (
	"context"

	"github.com/gojargo/jargo/audio"
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
	// AudioInFilter, when set, transforms received audio (for example noise
	// reduction) before it is pushed downstream to VAD, STT and turn detection.
	AudioInFilter audio.Filter

	// AudioOutEnabled enables sending audio.
	AudioOutEnabled bool
	// AudioOutSampleRate is the output sample rate in Hz; 0 uses the StartFrame's.
	AudioOutSampleRate int
	// AudioOutChannels is the number of output channels.
	AudioOutChannels int
	// AudioOutBitrate is the output bitrate in bits per second; 0 uses the
	// codec default.
	AudioOutBitrate int
	// AudioOutFEC enables Opus inband forward error correction on the outgoing
	// stream, letting receivers rebuild dropped packets. Recommended whenever
	// clients may be on lossy links.
	AudioOutFEC bool
	// AudioOutExpectedPacketLoss is the loss percentage (0-100) the encoder
	// sizes its FEC redundancy for. Ignored unless AudioOutFEC is set; 0 leaves
	// FEC enabled but carrying no redundancy, so set it alongside.
	AudioOutExpectedPacketLoss int
	// AudioOut10msChunks is how many 10 ms chunks of audio are written at a
	// time. With WebRTC Opus this is 2, so audio is written in 20 ms frames.
	AudioOut10msChunks int
	// AudioOutMixer, when set, mixes auxiliary audio (for example background
	// music) into the outgoing audio before it is sent. It serves the default
	// destination only; use AudioOutMixers to mix into a named one.
	AudioOutMixer audio.Mixer
	// AudioOutMixers maps a destination name to the mixer serving it, for a
	// transport that sends several outgoing streams and wants different
	// auxiliary audio (or none) on each. It takes precedence over AudioOutMixer.
	AudioOutMixers map[string]audio.Mixer
	// AudioOutEndSilenceSecs is how many seconds of silence are sent after the
	// last of the audio when the pipeline ends, so the closing words are not
	// clipped by whatever closes on top of them. 0 sends none.
	AudioOutEndSilenceSecs int
	// AudioOutDestinations names the additional outgoing audio streams the
	// transport serves, beyond the default unnamed one. A frame is routed to the
	// destination it carries, so each stream keeps its own buffer, mixer and
	// speaking state. Leave it empty for a transport with a single stream.
	AudioOutDestinations []string
}

// DefaultParams returns Params with audio input and output enabled and the
// defaults a WebRTC transport uses: mono, input passthrough on, 20 ms output
// chunks.
func DefaultParams() Params {
	return Params{
		AudioInEnabled:         true,
		AudioInChannels:        1,
		AudioInPassthrough:     true,
		AudioOutEnabled:        true,
		AudioOutChannels:       1,
		AudioOut10msChunks:     2,
		AudioOutEndSilenceSecs: 2,
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
	// StartAudioStreaming begins streaming audio from the transport's source,
	// for a transport that does not start streaming as soon as it connects. It
	// is driven by an InputTransportStartAudioStreamingFrame; the default does
	// nothing.
	StartAudioStreaming(ctx context.Context) error
}

// OutputDriver is implemented by a concrete output transport so the base can
// hand it audio chunks and messages to send.
type OutputDriver interface {
	processor.Processor
	// WriteAudio sends one chunk of audio over the transport. The frame carries
	// the interleaved S16LE PCM in AudioData, and the outgoing stream it belongs
	// to in TransportDestination, so a transport serving several streams can
	// send it on the right one.
	//
	// It must return once ctx is done, however far it had got. Stopping the
	// output cancels ctx and then waits for the send loop to finish, and the
	// loop is inside this call whenever it is sending, so a write that blocks
	// past cancellation holds the pipeline open for good. A transport that
	// paces its sends, or waits for room downstream, waits on ctx.Done()
	// alongside whatever else it waits for.
	//
	// It reports whether the chunk was sent. A transport with nowhere to put it
	// returns false and no error: its track is not live yet, its stream has
	// closed, its serializer produced nothing for this frame. Nothing failed,
	// but the audio will not be heard, so the base does not forward the frame
	// downstream as though it had been. An error is a genuine failure, which
	// the base logs; audio that errored counts as unsent either way.
	WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (sent bool, err error)
	// SendMessage sends an application message to the client (for example over
	// a data channel).
	SendMessage(ctx context.Context, data []byte) error
	// WriteTransportFrame handles a queued frame that carries no audio, once the
	// audio ahead of it has been sent. A transport overrides it to act on frame
	// types it carries itself; the default does nothing.
	WriteTransportFrame(ctx context.Context, f frames.Frame) error
	// StartWriting opens the transport's outgoing media path, so that nothing is
	// sent before it can carry audio. The base calls it while starting, and only
	// once it returns does it start the senders and report the transport ready.
	// A transport with nothing to open leaves the default, which does nothing.
	StartWriting(ctx context.Context) error
	// RegisterAudioDestination opens the outgoing stream a destination names,
	// for a transport that serves more than one. It is called for each entry in
	// Params.AudioOutDestinations when the transport starts, and a destination
	// that fails to open is skipped rather than being sent to. The default does
	// nothing.
	RegisterAudioDestination(ctx context.Context, destination string) error
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
