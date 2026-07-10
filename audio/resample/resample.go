// Package resample converts interleaved S16LE PCM audio between sample rates,
// preserving the channel count. It reconciles service sample rates (for example
// 24 kHz TTS audio) with the 48 kHz the WebRTC/Opus output path needs.
//
// A Resampler is stateful: it carries filter state across calls so a continuous
// stream resamples cleanly across chunk boundaries. Create one per audio stream
// with New; it is not safe for concurrent use, and Close must be called when
// finished.
//
// Two builds are selected by the `libsoxr` build tag, both exposing the same
// New/Process/Close API:
//
//   - Default (pure Go, see resample_purego.go): a no-cgo converter from
//     github.com/gojargo/go-resample. It cross-compiles and links into static
//     binaries with no native dependency.
//   - `-tags libsoxr` (see resample_soxr.go): links libsoxr (the SoX Resampler)
//     via cgo for its high-quality polyphase conversion. Requires libsoxr at
//     build and run time (libsoxr-dev to build, libsoxr0 to run).
package resample
