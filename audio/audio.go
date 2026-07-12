// Package audio defines the interfaces for pluggable audio processing a
// transport applies to the media it sends and receives. A Filter transforms
// incoming audio — for example noise reduction — before it enters the pipeline;
// a Mixer blends auxiliary audio — for example background music — into the
// outgoing audio. Concrete implementations live in the audio subpackages
// (rnnoise, noise, mixer).
package audio

import "context"

// Filter transforms input audio before it enters the pipeline. Start is called
// with the input sample rate when the transport starts, Filter is called for
// each received chunk of 16-bit mono PCM, and Stop is called when the transport
// stops. Filter may buffer internally and return fewer (or more) samples than it
// was given; it returns an empty slice when it has nothing to emit yet.
type Filter interface {
	Start(ctx context.Context, sampleRate int) error
	Stop(ctx context.Context) error
	Filter(ctx context.Context, pcm []byte) ([]byte, error)
}

// Mixer mixes auxiliary audio — for example background music or sound effects —
// into a transport's outgoing audio. Start is called with the output sample rate
// when the transport starts; Mix blends the mixer's current audio into an
// outgoing chunk of bot audio and returns the result; Control applies a runtime
// settings update from a MixerControlFrame; Stop is called when the transport
// stops.
type Mixer interface {
	Start(ctx context.Context, sampleRate int) error
	Stop(ctx context.Context) error
	Mix(ctx context.Context, pcm []byte) ([]byte, error)
	Control(ctx context.Context, settings map[string]any) error
}
