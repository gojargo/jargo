// Package resample converts interleaved S16LE PCM audio between sample rates,
// preserving the channel count. It reconciles service sample rates (for example
// 24 kHz TTS audio) with the 48 kHz the WebRTC/Opus output path needs.
//
// There are two ways to convert, and which one is right depends on whether the
// audio keeps coming:
//
//   - A Resampler is for a stream. It is stateful, carrying filter state across
//     calls so a continuous stream converts cleanly across chunk boundaries.
//     Its output keeps pace with its input, so equal chunks in give equal
//     chunks out, the first one included; the filter delays the audio by a few
//     milliseconds, which Flush emits where the stream ends. Create one per
//     stream with New; it is not safe for concurrent use.
//   - Resample is for a buffer that is complete on its own: a sound effect, a
//     recorded utterance, one turn of audio handed to a model. It converts in a
//     single pass, taking the clip to be silent before and after, and its
//     output lines up with its input and keeps the clip's length.
//
// Both run one polyphase windowed-sinc filter (see sinc.go), flat to 90% of the
// lower rate's band, with about 90 dB of aliasing and image rejection. It is
// pure Go, so it cross-compiles and links into static binaries with no native
// dependency.
package resample

// Resample converts a complete buffer of interleaved S16LE PCM from inRate to
// outRate in a single pass. The output lines up with the input and holds the
// clip's length at the output rate, rounded to the nearest frame. Use it for
// audio that arrives whole; use a Resampler for audio that keeps coming.
//
// It returns the input unchanged when the rates match, and nil for an empty
// buffer.
func Resample(pcm []byte, inRate, outRate, channels int) ([]byte, error) {
	return ResampleQuality(pcm, inRate, outRate, channels, QualityVHQ)
}

// ResampleQuality is Resample; the quality has no effect.
//
// Deprecated: use Resample. Every conversion runs the same filter.
func ResampleQuality(pcm []byte, inRate, outRate, channels int, q Quality) ([]byte, error) {
	if channels < 1 {
		channels = 1
	}
	if inRate == outRate {
		return pcm, nil
	}
	return resampleAll(pcm, inRate, outRate, channels)
}
