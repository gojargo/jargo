//go:build libsoxr

package resample

/*
#cgo pkg-config: soxr
#include <soxr.h>

// Thin wrappers so the spec constructors and their constants are resolved by the
// C compiler rather than relying on cgo macro/enum exposure.
static soxr_io_spec_t      jargo_io_int16(void)   { return soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I); }
static soxr_quality_spec_t jargo_quality(unsigned long recipe) { return soxr_quality_spec(recipe, 0); }
static soxr_runtime_spec_t jargo_runtime_1t(void) { return soxr_runtime_spec(1); }

static unsigned long jargo_recipe_vhq(void) { return SOXR_VHQ; }
static unsigned long jargo_recipe_hq(void)  { return SOXR_HQ;  }
static unsigned long jargo_recipe_mq(void)  { return SOXR_MQ;  }
static unsigned long jargo_recipe_lq(void)  { return SOXR_LQ;  }
static unsigned long jargo_recipe_qq(void)  { return SOXR_QQ;  }
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

//nolint:gochecknoglobals // sentinel errors
var (
	// errSoxrCreate is returned when libsoxr fails to create a resampler.
	errSoxrCreate = errors.New("resample: soxr_create failed")
	// errSoxrOneshot is returned when a single-pass conversion fails.
	errSoxrOneshot = errors.New("resample: soxr_oneshot failed")
)

// recipeFor maps a Quality onto the libsoxr recipe of the same name. The five
// names are libsoxr's own, so this is a straight pass-through.
func recipeFor(q Quality) C.ulong {
	switch q {
	case QualityVHQ:
		return C.jargo_recipe_vhq()
	case QualityHQ:
		return C.jargo_recipe_hq()
	case QualityMQ:
		return C.jargo_recipe_mq()
	case QualityLQ:
		return C.jargo_recipe_lq()
	case QualityQQ:
		return C.jargo_recipe_qq()
	default:
		return C.jargo_recipe_vhq()
	}
}

// Resampler converts a stream of interleaved S16LE PCM from one sample rate to
// another using libsoxr (the SoX Resampler) via cgo. Built with `-tags libsoxr`;
// the default build is the pure-Go converter (see resample_purego.go). Create
// one per audio stream with New; it is not safe for concurrent use. Call Close
// when finished to free the native resampler.
type Resampler struct {
	inRate   int
	outRate  int
	channels int
	soxr     C.soxr_t // nil when inRate == outRate (passthrough)

	idle idleClock
}

// New returns a Resampler from inRate to outRate for the given channel count,
// at the default quality and idle window. It returns an error if libsoxr cannot
// create the resampler. When inRate equals outRate the Resampler passes audio
// through unchanged and holds no native resources.
func New(inRate, outRate, channels int) (*Resampler, error) {
	return NewWithConfig(inRate, outRate, channels, Config{})
}

// NewWithConfig returns a Resampler configured by cfg. See New.
func NewWithConfig(inRate, outRate, channels int, cfg Config) (*Resampler, error) {
	if channels < 1 {
		channels = 1
	}
	r := &Resampler{inRate: inRate, outRate: outRate, channels: channels, idle: newIdleClock(cfg)}
	if inRate == outRate {
		return r, nil
	}

	ioSpec := C.jargo_io_int16()
	qSpec := C.jargo_quality(recipeFor(cfg.Quality))
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

// Clear discards the filter history, so the next chunk is converted as the start
// of a fresh signal rather than the continuation of the last one. The rate and
// quality are unchanged.
func (r *Resampler) Clear() {
	if r.soxr != nil {
		C.soxr_clear(r.soxr)
	}
	r.idle.reset()
}

// Process resamples one buffer of interleaved S16LE PCM and returns the
// resampled audio. When the input and output rates match it returns the input
// unchanged; otherwise the returned slice is freshly allocated and owned by the
// caller. Because libsoxr has filter delay, the first calls of a stream emit
// slightly fewer frames than the rate ratio implies, which later calls make up.
// Use Resample instead for a buffer that is complete on its own.
//
// A gap longer than the configured idle window clears the filter history first,
// so the tail of what came before is not mixed into what comes next.
func (r *Resampler) Process(in []byte) []byte {
	if r.soxr == nil {
		return in
	}
	if r.idle.stale() {
		C.soxr_clear(r.soxr)
	}
	bytesPerFrame := r.channels * 2
	inFrames := len(in) / bytesPerFrame
	if inFrames == 0 {
		return nil
	}

	// soxr_process is not obliged to consume the whole buffer in one call, so
	// feed it until it has. Anything left behind would be audio silently dropped
	// from the middle of the stream. The room each pass asks for covers the
	// ratio's worth of output plus the delay libsoxr is currently holding, since
	// a call that catches up on an earlier delay emits more than the ratio
	// implies; one pass is normally enough and the loop is the safety net.
	var out []byte
	for consumed := 0; consumed < inFrames; {
		remaining := inFrames - consumed
		room := remaining*r.outRate/r.inRate + int(C.soxr_delay(r.soxr)) + 64
		buf := make([]byte, room*bytesPerFrame)

		var idone, odone C.size_t
		serr := C.soxr_process(r.soxr,
			C.soxr_in_t(unsafe.Pointer(&in[consumed*bytesPerFrame])), C.size_t(remaining), &idone,
			C.soxr_out_t(unsafe.Pointer(&buf[0])), C.size_t(room), &odone)
		if serr != nil {
			return nil
		}
		out = append(out, buf[:int(odone)*bytesPerFrame]...)
		if idone == 0 {
			// No progress, so looping again would spin.
			break
		}
		consumed += int(idone)
	}
	return out
}

// resampleAll converts a complete buffer in one pass, flushing the converter's
// filter delay so nothing is left behind. It backs Resample; see there.
func resampleAll(pcm []byte, inRate, outRate, channels int, q Quality) ([]byte, error) {
	bytesPerFrame := channels * 2
	inFrames := len(pcm) / bytesPerFrame
	if inFrames == 0 {
		return nil, nil
	}

	// The margin covers rounding and the flushed filter tail, which a one-shot
	// conversion emits on top of the ratio's worth of output.
	outFrames := inFrames*outRate/inRate + 64
	out := make([]byte, outFrames*bytesPerFrame)

	ioSpec := C.jargo_io_int16()
	qSpec := C.jargo_quality(recipeFor(q))
	rtSpec := C.jargo_runtime_1t()

	var idone, odone C.size_t
	serr := C.soxr_oneshot(
		C.double(inRate), C.double(outRate), C.uint(channels),
		C.soxr_in_t(unsafe.Pointer(&pcm[0])), C.size_t(inFrames), &idone,
		C.soxr_out_t(unsafe.Pointer(&out[0])), C.size_t(outFrames), &odone,
		&ioSpec, &qSpec, &rtSpec)
	if serr != nil {
		return nil, fmt.Errorf("%w %d->%d ch=%d: %s",
			errSoxrOneshot, inRate, outRate, channels,
			C.GoString((*C.char)(unsafe.Pointer(serr))))
	}
	if int(idone) != inFrames {
		return nil, fmt.Errorf("%w %d->%d ch=%d: consumed %d of %d frames",
			errSoxrOneshot, inRate, outRate, channels, int(idone), inFrames)
	}
	return out[:int(odone)*bytesPerFrame], nil
}

// Close releases the native resampler. It is safe to call more than once and on
// a passthrough Resampler.
func (r *Resampler) Close() {
	if r.soxr != nil {
		C.soxr_delete(r.soxr)
		r.soxr = nil
	}
}
