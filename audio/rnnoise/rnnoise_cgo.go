//go:build rnnoise

package rnnoise

/*
#cgo pkg-config: rnnoise
#include <rnnoise.h>
*/
import "C"

import (
	"context"
	"encoding/binary"
	"errors"
	"sync"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/transport"
)

// rnnoiseRate is the only sample rate RNNoise operates at.
const rnnoiseRate = 48000

// errCreate is returned when RNNoise fails to allocate its state.
//
//nolint:gochecknoglobals // sentinel error
var errCreate = errors.New("rnnoise: failed to create denoise state")

// New builds an RNNoise input audio filter.
func New() (transport.AudioFilter, error) {
	return &filter{}, nil
}

type filter struct {
	mu    sync.Mutex
	st    *C.DenoiseState
	frame int // samples RNNoise consumes per call (480)

	up   *resample.Resampler // input rate -> 48 kHz
	down *resample.Resampler // 48 kHz -> input rate

	buf    []byte    // buffered 48 kHz S16LE mono awaiting whole frames
	in     []C.float // scratch: one frame of input samples
	out    []C.float // scratch: one frame of denoised samples
	frameB int       // frame size in bytes (frame * 2)
}

// Start allocates the denoiser and, when the transport runs at another rate, the
// resamplers to and from 48 kHz.
func (f *filter) Start(_ context.Context, sampleRate int) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.st = C.rnnoise_create(nil)
	if f.st == nil {
		return errCreate
	}
	f.frame = int(C.rnnoise_get_frame_size())
	f.frameB = f.frame * 2
	f.in = make([]C.float, f.frame)
	f.out = make([]C.float, f.frame)
	f.buf = nil

	if sampleRate != rnnoiseRate {
		var err error
		if f.up, err = resample.New(sampleRate, rnnoiseRate, 1); err != nil {
			return err
		}
		if f.down, err = resample.New(rnnoiseRate, sampleRate, 1); err != nil {
			return err
		}
	}
	return nil
}

// Stop frees the denoiser and resamplers.
func (f *filter) Stop(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.st != nil {
		C.rnnoise_destroy(f.st)
		f.st = nil
	}
	if f.up != nil {
		f.up.Close()
		f.up = nil
	}
	if f.down != nil {
		f.down.Close()
		f.down = nil
	}
	f.buf = nil
	return nil
}

// Filter denoises pcm. It resamples to 48 kHz, processes every complete 480-sample
// frame (buffering any remainder for the next call), resamples back to the input
// rate, and returns the result — empty when no whole frame was available yet.
func (f *filter) Filter(_ context.Context, pcm []byte) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.st == nil {
		return pcm, nil // not started: passthrough
	}

	wide := pcm
	if f.up != nil {
		wide = f.up.Process(pcm)
	}
	f.buf = append(f.buf, wide...)

	nFrames := len(f.buf) / f.frameB
	if nFrames == 0 {
		return nil, nil
	}

	denoised := make([]byte, 0, nFrames*f.frameB)
	for i := range nFrames {
		chunk := f.buf[i*f.frameB : (i+1)*f.frameB]
		for j := range f.frame {
			f.in[j] = C.float(int16(binary.LittleEndian.Uint16(chunk[j*2:])))
		}
		C.rnnoise_process_frame(f.st, &f.out[0], &f.in[0])
		frameOut := make([]byte, f.frameB)
		for j := range f.frame {
			binary.LittleEndian.PutUint16(frameOut[j*2:], uint16(clampInt16(float32(f.out[j]))))
		}
		denoised = append(denoised, frameOut...)
	}

	// Keep the sub-frame remainder for the next call.
	rem := len(f.buf) - nFrames*f.frameB
	tail := make([]byte, rem)
	copy(tail, f.buf[nFrames*f.frameB:])
	f.buf = tail

	if f.down != nil {
		return f.down.Process(denoised), nil
	}
	return denoised, nil
}

// clampInt16 saturates a float sample to the 16-bit range.
func clampInt16(v float32) int16 {
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}
