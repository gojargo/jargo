// Package rnnoise is an input audio filter that suppresses background noise with
// RNNoise, Xiph's recurrent-network denoiser. It satisfies transport.AudioFilter
// and is placed on a transport's audio input so VAD, STT and turn detection see
// cleaned audio.
//
// RNNoise is a C library linked through cgo, so the denoiser is only compiled in
// with the "rnnoise" build tag (which requires librnnoise). The default build
// uses a no-op passthrough, keeping the framework buildable without the native
// dependency:
//
//	go build -tags rnnoise ./...   # real denoiser (needs librnnoise)
//	go build ./...                 # no-op passthrough
//
// RNNoise operates on 48 kHz mono audio in 480-sample (10 ms) frames; the filter
// buffers partial frames and resamples to and from 48 kHz when the transport
// runs at another rate.
package rnnoise
