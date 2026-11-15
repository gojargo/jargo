// Package videocodec binds the native video codecs a WebRTC peer encodes and
// decodes with: libvpx for VP8, and OpenH264 for H.264. Both are loaded at run
// time through purego from the shared libraries a distribution installs, so the
// build needs no C toolchain.
//
// The frames the codecs exchange are I420 (planar YUV 4:2:0), converted to and
// from packed RGB by I420ToRGB and RGBToI420.
package videocodec

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/ebitengine/purego"
)

// The platforms whose libraries go by names of their own.
const (
	goosDarwin  = "darwin"
	goosWindows = "windows"
)

// Errors from loading the libraries.
var (
	// ErrVPXNotAvailable is returned when libvpx cannot be loaded, so a caller
	// can tell a missing install from a failure of the codec.
	ErrVPXNotAvailable = errors.New("videocodec: libvpx not available")
	// ErrOpenH264NotAvailable is returned when libopenh264 cannot be loaded.
	ErrOpenH264NotAvailable = errors.New("videocodec: libopenh264 not available")
)

// openShared opens the library at path, which must exist, or the first of its
// conventional names the loader finds. notAvailable is the error a failure
// wraps.
func openShared(path string, names []string, notAvailable error) (uintptr, error) {
	candidates := names
	if path != "" {
		if _, err := os.Stat(path); err != nil {
			return 0, fmt.Errorf("%w: %w", notAvailable, err)
		}
		candidates = []string{path}
	}
	var last error
	for _, name := range candidates {
		h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return h, nil
		}
		last = err
	}
	return 0, fmt.Errorf("%w: tried %v: %w", notAvailable, candidates, last)
}

// bind looks up each named symbol of the library and binds it to the function
// pointed at.
func bind(handle uintptr, symbols map[string]any, notAvailable error) error {
	for name, fn := range symbols {
		sym, err := purego.Dlsym(handle, name)
		if err != nil {
			return fmt.Errorf("%w: %s: %w", notAvailable, name, err)
		}
		purego.RegisterFunc(fn, sym)
	}
	return nil
}

// libraryCache keeps the libraries opened so far, by the path they were asked
// for, so every codec naming the same library binds it once.
type libraryCache[L any] struct {
	mu   sync.Mutex
	libs map[string]*L
}

// get returns the library loaded from path, opening it with open the first
// time.
func (c *libraryCache[L]) get(path string, open func(string) (*L, error)) (*L, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if lib, ok := c.libs[path]; ok {
		return lib, nil
	}
	lib, err := open(path)
	if err != nil {
		return nil, err
	}
	if c.libs == nil {
		c.libs = map[string]*L{}
	}
	c.libs[path] = lib
	return lib, nil
}
