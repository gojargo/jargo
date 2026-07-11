// Package onnxrt centralizes jargo's use of the ONNX runtime. It owns the
// single native boundary in jargo: the ONNX runtime shared library is located
// and initialized here once for the whole process, and every model session is
// created through New.
//
// The binding has two interchangeable backends, selected at build time:
//
//   - cgo build (the default): the C++ ONNX Runtime via yalue/onnxruntime_go.
//   - CGO_ENABLED=0 build: the same runtime via ebitengine/purego, calling the
//     ONNX Runtime C API with no cgo. See backend_purego.go.
//
// Either way the runtime is loaded at run time from its shared library
// (libonnxruntime.so / .dylib / onnxruntime.dll). Point jargo at it with the
// JARGO_ONNXRUNTIME_LIB environment variable:
//
//	export JARGO_ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so
//
// The library is not bundled; download a build matching your platform from
// https://github.com/microsoft/onnxruntime/releases. Models themselves are
// embedded in the vad/turn packages, so only the runtime needs locating.
package onnxrt

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
)

// LibPathEnv is the environment variable that points at the ONNX runtime
// shared library.
const LibPathEnv = "JARGO_ONNXRUNTIME_LIB"

// Tensor is a backend-neutral dense tensor: a shape plus flat data of exactly
// one element type. It is the currency of Run, so neither the models nor their
// callers depend on a particular binding's tensor type. Only the element types
// jargo's models use are represented; exactly one data field is non-nil.
type Tensor struct {
	Shape []int64
	F32   []float32
	I64   []int64
}

// Float32 builds a float32 tensor.
func Float32(shape []int64, data []float32) Tensor { return Tensor{Shape: shape, F32: data} }

// Int64 builds an int64 tensor.
func Int64(shape []int64, data []int64) Tensor { return Tensor{Shape: shape, I64: data} }

// Options tune a session. The zero value uses the runtime's defaults and
// matches the behavior before Options existed.
type Options struct {
	// IntraOpThreads caps ONNX Runtime's intra-op thread pool. 0 leaves the
	// runtime default (roughly one thread per core). When many small models run
	// concurrently — one session per stream — the default oversubscribes the
	// machine badly (streams × cores threads); 1 keeps each Run on its caller.
	IntraOpThreads int
}

// backendSession is one loaded model. It is implemented by the build-tagged
// backend (backend_cgo.go or backend_purego.go). run allocates and returns its
// outputs as copies that are safe to keep after run returns.
type backendSession interface {
	run(inputs []Tensor) ([]Tensor, error)
	close() error
}

// The build-tagged backend also provides these package functions:
//
//	func backendInit(libPath string) error
//	func newBackendSession(model []byte, inputNames, outputNames []string, opts Options) (backendSession, error)

//nolint:gochecknoglobals // process-wide one-time runtime initialization
var (
	initOnce sync.Once
	errInit  error
)

// Init locates and initializes the ONNX runtime exactly once for the process.
// It is safe to call repeatedly and from multiple goroutines; later calls
// return the result of the first. Every analyzer that loads a model calls it
// before creating a session.
func Init() error {
	initOnce.Do(func() {
		path, err := libraryPath()
		if err != nil {
			errInit = err
			return
		}
		if err := backendInit(path); err != nil {
			errInit = fmt.Errorf("onnxrt: initialize runtime (lib %q): %w", path, err)
		}
	})
	return errInit
}

// Session is a thin, concurrency-safe wrapper over a backend session. Inputs and
// outputs are named and supplied per call, which suits the recurrent Silero VAD
// model (state fed back in each call) and the single-shot Smart Turn model
// alike.
type Session struct {
	mu sync.Mutex
	b  backendSession
}

// New creates a session for the given embedded model bytes, declaring the
// input and output tensor names in the order Run expects them. It initializes
// the runtime if needed and uses the runtime's default options.
func New(model []byte, inputNames, outputNames []string) (*Session, error) {
	return NewWithOptions(model, inputNames, outputNames, Options{})
}

// NewWithOptions is New with explicit session Options.
func NewWithOptions(model []byte, inputNames, outputNames []string, opts Options) (*Session, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	b, err := newBackendSession(model, inputNames, outputNames, opts)
	if err != nil {
		return nil, err
	}
	return &Session{b: b}, nil
}

// Run executes the model. inputs must match the input names given to New, in
// order; outputs are returned in the order of the output names. Run is safe for
// concurrent use but serializes calls on a session, matching ONNX Runtime's
// single-stream model and the one-stream-per-analyzer design.
func (s *Session) Run(inputs []Tensor) ([]Tensor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.b == nil {
		return nil, errClosed
	}
	return s.b.run(inputs)
}

// Close releases the session.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.b == nil {
		return nil
	}
	err := s.b.close()
	s.b = nil
	return err
}

// libraryPath resolves the ONNX runtime shared-library path from the
// environment, falling back to the platform's conventional name so a
// system-installed runtime is found on the loader's default search path.
func libraryPath() (string, error) {
	if p := os.Getenv(LibPathEnv); p != "" {
		//nolint:gosec // operator-provided path to the ONNX runtime library
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("onnxrt: %s=%q: %w", LibPathEnv, p, err)
		}
		return p, nil
	}
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	default:
		return "libonnxruntime.so", nil
	}
}

// ErrNotConfigured is returned when the runtime library cannot be located, so
// callers (and tests) can distinguish "not set up" from a genuine failure.
var ErrNotConfigured = errors.New("onnxrt: ONNX runtime library not configured")

// errClosed is returned by Run after the session has been closed.
var errClosed = errors.New("onnxrt: run on closed session")

// Available reports whether the ONNX runtime can be initialized. It is intended
// for tests that should skip when no runtime is present.
func Available() bool {
	return Init() == nil
}

// Backend reports which binding was compiled in: "cgo (yalue)" for the default
// build, or "purego" for a CGO_ENABLED=0 build.
func Backend() string { return backendName }
