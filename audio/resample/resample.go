// Package resample converts interleaved S16LE PCM audio between sample rates
// using libsoxr (the SoX Resampler), a high-quality polyphase resampler. It
// reconciles service sample rates (for example 24 kHz TTS audio) with the
// 48 kHz the WebRTC/Opus output path needs; the channel count is preserved.
//
// A Resampler is stateful: libsoxr carries filter state across calls so a
// continuous stream resamples cleanly across chunk boundaries. Create one per
// audio stream with New; it is not safe for concurrent use, and Close must be
// called to release the native resampler.
//
// This package uses cgo and requires libsoxr at build and run time. On
// Debian/Ubuntu install libsoxr-dev to build and libsoxr0 to run; the container
// image jargo ships in bundles both.
package resample

/*
#cgo pkg-config: soxr
#include <soxr.h>

// Thin wrappers so the spec constructors and their constants are resolved by the
// C compiler rather than relying on cgo macro/enum exposure.
static soxr_io_spec_t      jargo_io_int16(void)   { return soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I); }
static soxr_quality_spec_t jargo_quality_hq(void) { return soxr_quality_spec(SOXR_HQ, 0); }
static soxr_runtime_spec_t jargo_runtime_1t(void) { return soxr_runtime_spec(1); }
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// errSoxrCreate is returned when libsoxr fails to create a resampler.
//
//nolint:gochecknoglobals // sentinel error
var errSoxrCreate = errors.New("resample: soxr_create failed")

// Resampler converts a stream of interleaved S16LE PCM from one sample rate to
// another. Create one per audio stream with New; it is not safe for concurrent
// use. Call Close when finished to free the native resampler.
type Resampler struct {
	inRate   int
	outRate  int
	channels int
	soxr     C.soxr_t // nil when inRate == outRate (passthrough)
}

// New returns a Resampler from inRate to outRate for the given channel count. It
// returns an error if libsoxr cannot create the resampler. When inRate equals
// outRate the Resampler passes audio through unchanged and holds no native
// resources.
func New(inRate, outRate, channels int) (*Resampler, error) {
	if channels < 1 {
		channels = 1
	}
	r := &Resampler{inRate: inRate, outRate: outRate, channels: channels}
	if inRate == outRate {
		return r, nil
	}

	ioSpec := C.jargo_io_int16()
	qSpec := C.jargo_quality_hq()
	rtSpec := C.jargo_runtime_1t() // single-threaded: one resampler per stream

	var serr C.soxr_error_t
	r.soxr = C.soxr_create(
		C.double(inRate), C.double(outRate), C.uint(channels),
		&serr, &ioSpec, &qSpec, &rtSpec) //nolint:gocritic // dupSubExpr false positive in cgo-generated soxr_create
	if serr != nil {
		return nil, fmt.Errorf("%w %d->%d ch=%d: %s",
			errSoxrCreate, inRate, outRate, channels, C.GoString((*C.char)(unsafe.Pointer(serr))))
	}
	return r, nil
}

// Process resamples one buffer of interleaved S16LE PCM and returns the
// resampled audio. When the input and output rates match it returns the input
// unchanged; otherwise the returned slice is freshly allocated and owned by the
// caller. Because libsoxr has filter delay, the first calls of a stream emit
// slightly fewer frames than the rate ratio implies, which later calls make up.
func (r *Resampler) Process(in []byte) []byte {
	if r.soxr == nil {
		return in
	}
	bytesPerFrame := r.channels * 2
	inFrames := len(in) / bytesPerFrame
	if inFrames == 0 {
		return nil
	}

	// libsoxr emits about inFrames*outRate/inRate frames per call; the margin
	// covers rounding and the filter delay flushed in steady state, so a single
	// call always consumes all input.
	outFrames := inFrames*r.outRate/r.inRate + 64
	out := make([]byte, outFrames*bytesPerFrame)

	var idone, odone C.size_t
	serr := C.soxr_process(r.soxr,
		C.soxr_in_t(unsafe.Pointer(&in[0])), C.size_t(inFrames), &idone,
		C.soxr_out_t(unsafe.Pointer(&out[0])), C.size_t(outFrames), &odone)
	if serr != nil {
		return nil
	}
	return out[:int(odone)*bytesPerFrame]
}

// Close releases the native resampler. It is safe to call more than once and on
// a passthrough Resampler.
func (r *Resampler) Close() {
	if r.soxr != nil {
		C.soxr_delete(r.soxr)
		r.soxr = nil
	}
}
