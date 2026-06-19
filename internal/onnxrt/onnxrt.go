// Package onnxrt centralizes jargo's use of the ONNX runtime. It owns the
// single cgo boundary in jargo: the ONNX runtime shared library is located and
// initialized here once for the whole process, and every model session is
// created through New.
//
// The runtime is the C++ ONNX Runtime, loaded at run time via its shared
// library (libonnxruntime.so / .dylib / onnxruntime.dll). Point jargo at it
// with the JARGO_ONNXRUNTIME_LIB environment variable:
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

	ort "github.com/yalue/onnxruntime_go"
)

// LibPathEnv is the environment variable that points at the ONNX runtime
// shared library.
const LibPathEnv = "JARGO_ONNXRUNTIME_LIB"

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
		if ort.IsInitialized() {
			return
		}
		path, err := libraryPath()
		if err != nil {
			errInit = err
			return
		}
		ort.SetSharedLibraryPath(path)
		if err := ort.InitializeEnvironment(); err != nil {
			errInit = fmt.Errorf("onnxrt: initialize environment (lib %q): %w", path, err)
		}
	})
	return errInit
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

// Session is a thin wrapper over a dynamic ONNX runtime session. Inputs and
// outputs are named and supplied per call, which suits the recurrent Silero VAD
// model (state fed back in each call) and the single-shot Smart Turn model
// alike.
type Session struct {
	mu      sync.Mutex
	s       *ort.DynamicAdvancedSession
	inputs  []string
	outputs []string
}

// New creates a session for the given embedded model bytes, declaring the
// input and output tensor names in the order Run expects them. It initializes
// the runtime if needed.
func New(model []byte, inputNames, outputNames []string) (*Session, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnxrt: session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	s, err := ort.NewDynamicAdvancedSessionWithONNXData(model, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("onnxrt: create session: %w", err)
	}
	return &Session{s: s, inputs: inputNames, outputs: outputNames}, nil
}

// Run executes the model. inputs must match the input names given to New, in
// order. Outputs are allocated by the runtime and returned as Values; the
// caller is responsible for calling Destroy on each. Run is safe for
// concurrent use but serializes calls, matching ONNX Runtime's single-stream
// model and the one-stream-per-analyzer design.
func (s *Session) Run(inputs []ort.Value) ([]ort.Value, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	outputs := make([]ort.Value, len(s.outputs))
	if err := s.s.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("onnxrt: run: %w", err)
	}
	return outputs, nil
}

// Close releases the session.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.s == nil {
		return nil
	}
	err := s.s.Destroy()
	s.s = nil
	return err
}

// ErrNotConfigured is returned by Available when the runtime library cannot be
// located, so callers (and tests) can distinguish "not set up" from a genuine
// failure.
var ErrNotConfigured = errors.New("onnxrt: ONNX runtime library not configured")

// Available reports whether the ONNX runtime can be initialized. It is intended
// for tests that should skip when no runtime is present.
func Available() bool {
	return Init() == nil
}
