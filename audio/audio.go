// Package audio defines the interfaces for pluggable audio processing a
// transport applies to the media it sends and receives. A Filter transforms
// incoming audio (noise reduction, say) before it enters the pipeline; a Mixer
// blends auxiliary audio (background music, say) into the outgoing audio.
// Concrete implementations live in the audio subpackages (rnnoise, noise,
// mixer).
package audio

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// Filter transforms input audio before it enters the pipeline. Start is called
// with the input sample rate when the transport starts, Filter is called for
// each received chunk of 16-bit mono PCM, and Stop is called when the transport
// stops. Filter may buffer internally and return fewer (or more) samples than it
// was given; it returns an empty slice when it has nothing to emit yet.
// ProcessFrame applies a runtime control frame, so the filter can be retuned or
// switched off without being torn down.
//
// The pcm a filter is handed is read-only. The chunk it came in belongs to a
// frame, and a frame that has been pushed is being read elsewhere, so a filter
// that wrote into it would be writing under that reader. Return the filtered
// audio in a slice of the filter's own, or return the one it was given
// unchanged.
type Filter interface {
	Start(ctx context.Context, sampleRate int) error
	Stop(ctx context.Context) error
	Filter(ctx context.Context, pcm []byte) ([]byte, error)
	ProcessFrame(ctx context.Context, f frames.FilterControlFrame) error
}

// Mixer mixes auxiliary audio (background music or sound effects, say) into a
// transport's outgoing audio. Start is called with the output sample rate
// when the transport starts; Mix blends the mixer's current audio into an
// outgoing chunk of bot audio and returns the result; ProcessFrame applies a
// runtime control frame; Stop is called when the transport stops.
type Mixer interface {
	Start(ctx context.Context, sampleRate int) error
	Stop(ctx context.Context) error
	Mix(ctx context.Context, pcm []byte) ([]byte, error)
	ProcessFrame(ctx context.Context, f frames.MixerControlFrame) error
}
