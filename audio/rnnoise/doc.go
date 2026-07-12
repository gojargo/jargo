// Package rnnoise is an input audio filter that suppresses background noise with
// RNNoise, Xiph's recurrent-network denoiser. It satisfies transport.AudioFilter
// and is placed on a transport's audio input so VAD, STT and turn detection see
// cleaned audio. Select it like any other component by setting it as the
// transport's AudioInFilter; leaving that nil runs without noise reduction.
//
// RNNoise is a C library, but it is bound through ebitengine/purego rather than
// cgo: the package always builds (no build tag, CGO_ENABLED=0 friendly) and
// dlopens librnnoise at run time. New returns ErrNotAvailable when the library
// cannot be located, so a caller can fall back to running without it. Point at a
// non-standard install with the JARGO_RNNOISE_LIB environment variable.
//
// RNNoise operates on 48 kHz mono audio in 480-sample (10 ms) frames; the filter
// buffers partial frames and resamples to and from 48 kHz when the transport
// runs at another rate.
package rnnoise
