// Package resample converts interleaved S16LE PCM audio between sample rates,
// preserving the channel count. It reconciles service sample rates (for example
// 24 kHz TTS audio) with the 48 kHz the WebRTC/Opus output path needs.
//
// There are two ways to convert, and which one is right depends on whether the
// audio keeps coming:
//
//   - A Resampler is for a stream. It is stateful, carrying filter state across
//     calls so a continuous stream converts cleanly across chunk boundaries, and
//     it holds back the filter delay at the end of every call because more audio
//     is expected. Create one per stream with New; it is not safe for concurrent
//     use, and Close must be called when finished.
//   - Resample is for a buffer that is complete on its own: a sound effect, a
//     recorded utterance, one turn of audio handed to a model. It converts in a
//     single pass and flushes the filter delay, so nothing is clipped off the
//     end. Running a complete buffer through a Resampler instead loses a
//     millisecond or two of its tail.
//
// The conversion is pure Go (see resampler.go), a no-cgo converter from
// github.com/gojargo/go-resample. It cross-compiles and links into static
// binaries with no native dependency.
package resample

// Resample converts a complete buffer of interleaved S16LE PCM from inRate to
// outRate at the default quality, in a single pass. Unlike Resampler.Process it
// flushes the converter's filter delay, so the returned audio holds the whole
// signal rather than stopping a filter length short of it. Use it for audio that
// arrives whole; use a Resampler for audio that keeps coming.
//
// It returns the input unchanged when the rates match, and nil for an empty
// buffer.
func Resample(pcm []byte, inRate, outRate, channels int) ([]byte, error) {
	return ResampleQuality(pcm, inRate, outRate, channels, QualityVHQ)
}

// ResampleQuality is Resample at a chosen quality.
func ResampleQuality(pcm []byte, inRate, outRate, channels int, q Quality) ([]byte, error) {
	if channels < 1 {
		channels = 1
	}
	if inRate == outRate {
		return pcm, nil
	}
	return resampleAll(pcm, inRate, outRate, channels, q)
}
