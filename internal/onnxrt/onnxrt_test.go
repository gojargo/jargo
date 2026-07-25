package onnxrt

// These tests deliberately stop at the native boundary.
//
// Most of this package is FFI: purego symbol lookup into libonnxruntime, unsafe
// pointer arithmetic, and a cgo alternative behind a build tag. Exercising that
// needs the shared library present, which CI does not guarantee — a test that
// skips when it is missing buys nothing, and one that fakes the library tests
// the fake. So what is covered here is the logic that is genuinely independent
// of the runtime: locating the library, the session wrapper's lifecycle, and the
// tensor constructors. Loading and running a real model is exercised for real by
// the vad and turn packages when a runtime is installed.

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// errBackend is the failure the fake backend session reports.
//
//nolint:gochecknoglobals // sentinel error
var errBackend = errors.New("backend failed")

// fakeSession stands in for a loaded model so the Session wrapper's locking and
// lifecycle can be tested without a runtime.
type fakeSession struct {
	mu       sync.Mutex
	runs     int
	closes   int
	runErr   error
	closeErr error
	outputs  []Tensor
}

func (f *fakeSession) run(inputs []Tensor) ([]Tensor, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.runs++
	if f.runErr != nil {
		return nil, f.runErr
	}
	return append(f.outputs, inputs...), nil
}

func (f *fakeSession) close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closes++
	return f.closeErr
}

func (f *fakeSession) counts() (runs, closes int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.runs, f.closes
}

// TestLibraryPathFromEnv checks an operator-supplied path is used when it points
// at a real file, and rejected with a useful error when it does not — a typo in
// JARGO_ONNXRUNTIME_LIB must not silently fall back to the default name and load
// some other runtime.
func TestLibraryPathFromEnv(t *testing.T) {
	t.Run("existing file is used", func(t *testing.T) {
		lib := filepath.Join(t.TempDir(), "libonnxruntime.so")
		if err := os.WriteFile(lib, []byte("not really a library"), 0o600); err != nil {
			t.Fatalf("writing the fake library: %v", err)
		}
		t.Setenv(LibPathEnv, lib)

		got, err := libraryPath()
		if err != nil {
			t.Fatalf("libraryPath: %v", err)
		}
		if got != lib {
			t.Errorf("libraryPath() = %q, want %q", got, lib)
		}
	})

	t.Run("missing file is an error", func(t *testing.T) {
		missing := filepath.Join(t.TempDir(), "nope.so")
		t.Setenv(LibPathEnv, missing)

		got, err := libraryPath()
		if err == nil {
			t.Fatalf("libraryPath() = %q, want an error for a nonexistent path", got)
		}
		// The message must name the variable and the path, or an operator
		// cannot tell which of the two is wrong.
		if msg := err.Error(); !strings.Contains(msg, LibPathEnv) || !strings.Contains(msg, missing) {
			t.Errorf("error = %q, want it to name %s and the path", msg, LibPathEnv)
		}
	})
}

// TestLibraryPathDefault checks the platform's conventional library name is used
// when nothing is configured, so a system-installed runtime is found on the
// loader's default search path.
func TestLibraryPathDefault(t *testing.T) {
	t.Setenv(LibPathEnv, "")

	got, err := libraryPath()
	if err != nil {
		t.Fatalf("libraryPath: %v", err)
	}

	want := map[string]string{
		"windows": "onnxruntime.dll",
		"darwin":  "libonnxruntime.dylib",
	}[runtime.GOOS]
	if want == "" {
		want = "libonnxruntime.so"
	}
	if got != want {
		t.Errorf("libraryPath() = %q, want %q on %s", got, want, runtime.GOOS)
	}
	if filepath.IsAbs(got) {
		t.Errorf("libraryPath() = %q, want a bare name so the loader searches its default paths", got)
	}
}

// TestSessionRun checks the wrapper passes inputs through and returns the
// backend's outputs and errors unchanged.
func TestSessionRun(t *testing.T) {
	t.Run("delegates to the backend", func(t *testing.T) {
		fake := &fakeSession{outputs: []Tensor{Float32([]int64{1}, []float32{0.5})}}
		s := &Session{b: fake}

		out, err := s.Run([]Tensor{Int64([]int64{1}, []int64{7})})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(out) != 2 {
			t.Fatalf("outputs = %d, want the backend's output plus the echoed input", len(out))
		}
		if runs, _ := fake.counts(); runs != 1 {
			t.Errorf("backend runs = %d, want 1", runs)
		}
	})

	t.Run("propagates backend errors", func(t *testing.T) {
		s := &Session{b: &fakeSession{runErr: errBackend}}
		if _, err := s.Run(nil); !errors.Is(err, errBackend) {
			t.Errorf("Run error = %v, want errBackend", err)
		}
	})

	t.Run("after close", func(t *testing.T) {
		s := &Session{b: &fakeSession{}}
		if err := s.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if _, err := s.Run(nil); !errors.Is(err, errClosed) {
			t.Errorf("Run error = %v, want errClosed", err)
		}
	})
}

// TestSessionClose checks closing is idempotent and releases the backend exactly
// once — a second native close would be a double free.
func TestSessionClose(t *testing.T) {
	fake := &fakeSession{}
	s := &Session{b: fake}

	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v, want nil", err)
	}
	if _, closes := fake.counts(); closes != 1 {
		t.Errorf("backend closes = %d, want exactly 1", closes)
	}
}

// TestSessionCloseError checks a backend failure on close is reported, while the
// session still drops its reference so it cannot be closed twice.
func TestSessionCloseError(t *testing.T) {
	fake := &fakeSession{closeErr: errBackend}
	s := &Session{b: fake}

	if err := s.Close(); !errors.Is(err, errBackend) {
		t.Errorf("Close error = %v, want errBackend", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
	if _, closes := fake.counts(); closes != 1 {
		t.Errorf("backend closes = %d, want exactly 1", closes)
	}
}

// TestSessionConcurrentRun checks Run serializes on the session, which is the
// contract that lets one analyzer share a session across goroutines.
func TestSessionConcurrentRun(t *testing.T) {
	fake := &fakeSession{}
	s := &Session{b: fake}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 25 {
				if _, err := s.Run(nil); err != nil {
					t.Errorf("Run: %v", err)
					return
				}
			}
		})
	}
	wg.Wait()

	if runs, _ := fake.counts(); runs != 200 {
		t.Errorf("backend runs = %d, want 200", runs)
	}
}

// TestTensorConstructors checks each constructor fills exactly its own element
// type, since the backends switch on which field is non-nil.
func TestTensorConstructors(t *testing.T) {
	f := Float32([]int64{1, 2}, []float32{1, 2})
	if f.I64 != nil {
		t.Error("Float32 should leave I64 nil")
	}
	if len(f.F32) != 2 || len(f.Shape) != 2 {
		t.Errorf("Float32 = %+v", f)
	}

	i := Int64([]int64{2}, []int64{3, 4})
	if i.F32 != nil {
		t.Error("Int64 should leave F32 nil")
	}
	if len(i.I64) != 2 || len(i.Shape) != 1 {
		t.Errorf("Int64 = %+v", i)
	}
}

// TestBackendName checks the build reports which binding was compiled in, which
// is what a bug report needs to be actionable.
func TestBackendName(t *testing.T) {
	switch got := Backend(); got {
	case "cgo (yalue)", "purego":
	default:
		t.Errorf("Backend() = %q, want one of the two known bindings", got)
	}
}
