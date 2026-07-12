package rnnoise

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/transport"
)

// rnnoiseRate is the only sample rate RNNoise operates at.
const rnnoiseRate = 48000

// LibPathEnv points at the librnnoise shared library when it is not on the
// loader's default search path.
const LibPathEnv = "JARGO_RNNOISE_LIB"

// librnnoise is bound at run time through purego (no cgo). RNNoise exposes flat
// exported symbols, so each function binds by name.
//
//nolint:gochecknoglobals // librnnoise is loaded process-wide, guarded by loadOnce
var (
	loadOnce sync.Once
	errLoad  error

	rnnoiseCreate       func(model uintptr) uintptr
	rnnoiseDestroy      func(st uintptr)
	rnnoiseGetFrameSize func() int32
	rnnoiseProcessFrame func(st uintptr, out, in *float32) float32
)

//nolint:gochecknoglobals // sentinel errors
var (
	errCreate = errors.New("rnnoise: failed to create denoise state")
	// ErrNotAvailable is returned by New when librnnoise cannot be located, so
	// callers can fall back to running without noise reduction.
	ErrNotAvailable = errors.New("rnnoise: librnnoise not available")
)

// load dlopens librnnoise and binds its symbols exactly once.
func load() error {
	loadOnce.Do(func() {
		handle, err := openLib()
		if err != nil {
			errLoad = err
			return
		}
		purego.RegisterLibFunc(&rnnoiseCreate, handle, "rnnoise_create")
		purego.RegisterLibFunc(&rnnoiseDestroy, handle, "rnnoise_destroy")
		purego.RegisterLibFunc(&rnnoiseGetFrameSize, handle, "rnnoise_get_frame_size")
		purego.RegisterLibFunc(&rnnoiseProcessFrame, handle, "rnnoise_process_frame")
	})
	return errLoad
}

// openLib dlopens the library from LibPathEnv, or the platform's conventional
// names on the loader's default search path.
func openLib() (uintptr, error) {
	var candidates []string
	if p := os.Getenv(LibPathEnv); p != "" {
		candidates = []string{p}
	} else {
		switch runtime.GOOS {
		case "windows":
			candidates = []string{"rnnoise.dll", "librnnoise.dll"}
		case "darwin":
			candidates = []string{"librnnoise.0.dylib", "librnnoise.dylib"}
		default:
			candidates = []string{"librnnoise.so.0", "librnnoise.so"}
		}
	}
	var last error
	for _, name := range candidates {
		h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return h, nil
		}
		last = err
	}
	return 0, fmt.Errorf("%w (set %s): tried %v: %w", ErrNotAvailable, LibPathEnv, candidates, last)
}

// New builds an RNNoise input audio filter. It loads librnnoise via purego (no
// cgo) on first use and returns ErrNotAvailable if the library cannot be found,
// so the caller can choose to run without noise reduction.
func New() (transport.AudioFilter, error) {
	if err := load(); err != nil {
		return nil, err
	}
	return &filter{}, nil
}

type filter struct {
	mu    sync.Mutex
	st    uintptr // RNNoise DenoiseState*
	frame int     // samples RNNoise consumes per call (480)

	up   *resample.Resampler // input rate -> 48 kHz
	down *resample.Resampler // 48 kHz -> input rate

	buf    []byte    // buffered 48 kHz S16LE mono awaiting whole frames
	in     []float32 // scratch: one frame of input samples
	out    []float32 // scratch: one frame of denoised samples
	frameB int       // frame size in bytes (frame * 2)
}

// Start allocates the denoiser and, when the transport runs at another rate, the
// resamplers to and from 48 kHz.
func (f *filter) Start(_ context.Context, sampleRate int) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.st = rnnoiseCreate(0)
	if f.st == 0 {
		return errCreate
	}
	f.frame = int(rnnoiseGetFrameSize())
	f.frameB = f.frame * 2
	f.in = make([]float32, f.frame)
	f.out = make([]float32, f.frame)
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

	if f.st != 0 {
		rnnoiseDestroy(f.st)
		f.st = 0
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

	if f.st == 0 {
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
			f.in[j] = float32(int16(binary.LittleEndian.Uint16(chunk[j*2:])))
		}
		// The return value is a per-frame voice-activity probability we don't use.
		rnnoiseProcessFrame(f.st, &f.out[0], &f.in[0])
		runtime.KeepAlive(f.in)
		frameOut := make([]byte, f.frameB)
		for j := range f.frame {
			binary.LittleEndian.PutUint16(frameOut[j*2:], uint16(clampInt16(f.out[j])))
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
