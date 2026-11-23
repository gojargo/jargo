package piper

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The status codes libpiper returns.
const (
	codeOK      int32 = 0
	codeDone    int32 = 1
	codeGeneric int32 = -1
)

// Errors the library reports, and loading it.
var (
	// ErrLibraryNotAvailable is returned when libpiper cannot be loaded, so a
	// caller can tell a missing install from a failure of the voice.
	ErrLibraryNotAvailable = errors.New("piper: libpiper not available")
	// errLoadVoice reports a voice the library could not load.
	errLoadVoice = errors.New("piper: failed to load voice")
	// errSynthesize reports the library failing to synthesize.
	errSynthesize = errors.New("piper: synthesis failed")
	// errVoiceClosed reports a voice used after it was freed.
	errVoiceClosed = errors.New("piper: voice is not loaded")
)

// synthesizeOptions is the synthesis options as the library reads them
// (piper_synthesize_options).
type synthesizeOptions struct {
	speakerID   int32
	lengthScale float32
	noiseScale  float32
	noiseWScale float32
}

// audioChunk is one chunk of synthesized audio as the library fills it
// (piper_audio_chunk). The service reads only the samples, their rate and
// whether the chunk is the last; the phoneme fields are laid out so the
// library writes the struct where it expects to.
type audioChunk struct {
	samples       *float32
	numSamples    uintptr
	sampleRate    int32
	isLast        bool
	phonemes      unsafe.Pointer
	numPhonemes   uintptr
	phonemeIDs    unsafe.Pointer
	numPhonemeIDs uintptr
	alignments    unsafe.Pointer
	numAlignments uintptr
}

// library is libpiper, with the calls the service makes bound.
type library struct {
	create                   func(modelPath, configPath, espeakDataPath *byte) uintptr
	free                     func(synth uintptr)
	defaultSynthesizeOptions func(synth uintptr) synthesizeOptions
	synthesizeStart          func(synth uintptr, text *byte, options *synthesizeOptions) int32
	synthesizeNext           func(synth uintptr, chunk *audioChunk) int32

	// path is the library file named in the configuration, "" for one found
	// on the default search path.
	path string
}

// The libraries opened so far, by the path they were asked for, so every
// service naming the same library binds it once.
//
//nolint:gochecknoglobals // the loaded library is process-wide
var (
	librariesMu sync.Mutex
	libraries   = map[string]*library{}
)

// libraryNames is the names libpiper goes by on this platform, looked for on
// the loader's default search path when no path is configured.
func libraryNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"libpiper.dylib"}
	case "windows":
		return []string{"piper.dll"}
	default:
		return []string{"libpiper.so"}
	}
}

// openLibrary loads libpiper from path, or by its conventional name when path
// is empty, and binds the calls the service makes. A library already loaded is
// reused.
func openLibrary(path string) (*library, error) {
	librariesMu.Lock()
	defer librariesMu.Unlock()
	if lib, ok := libraries[path]; ok {
		return lib, nil
	}
	handle, err := dlopenLibrary(path)
	if err != nil {
		return nil, err
	}
	lib := &library{path: path}
	for name, fn := range map[string]any{
		"piper_create":                     &lib.create,
		"piper_free":                       &lib.free,
		"piper_default_synthesize_options": &lib.defaultSynthesizeOptions,
		"piper_synthesize_start":           &lib.synthesizeStart,
		"piper_synthesize_next":            &lib.synthesizeNext,
	} {
		sym, err := purego.Dlsym(handle, name)
		if err != nil {
			return nil, fmt.Errorf("%w: %s: %w", ErrLibraryNotAvailable, name, err)
		}
		purego.RegisterFunc(fn, sym)
	}
	libraries[path] = lib
	return lib, nil
}

// dlopenLibrary opens the library at path, which must exist, or the first of
// its conventional names the loader finds.
func dlopenLibrary(path string) (uintptr, error) {
	candidates := libraryNames()
	if path != "" {
		if _, err := os.Stat(path); err != nil {
			return 0, fmt.Errorf("%w: %w", ErrLibraryNotAvailable, err)
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
	return 0, fmt.Errorf("%w: tried %v: %w", ErrLibraryNotAvailable, candidates, last)
}

// espeakDataDir is the espeak-ng data directory installed with the library at
// libPath: share/espeak-ng-data beside its lib directory, as libpiper's
// install lays it out. It is "" when there is no such directory, or no path to
// look beside, and the library then falls back to the directory espeak-ng was
// built to read.
func espeakDataDir(libPath string) string {
	if libPath == "" {
		return ""
	}
	dir := filepath.Join(filepath.Dir(libPath), "..", "share", "espeak-ng-data")
	if info, err := os.Stat(dir); err == nil && info.IsDir() {
		return filepath.Clean(dir)
	}
	return ""
}

// loadVoice loads the voice model at modelPath, with its configuration at
// modelPath + ".json" and espeak-ng's data at espeakDataPath ("" for the
// library's default).
func (l *library) loadVoice(modelPath, espeakDataPath string) (voiceModel, error) {
	model := cString(modelPath)
	var espeak *byte
	if espeakDataPath != "" {
		espeak = &cString(espeakDataPath)[0]
	}
	synth := l.create(&model[0], nil, espeak)
	if synth == 0 {
		return nil, fmt.Errorf("%w: %s", errLoadVoice, modelPath)
	}
	state := &nativeHandle{lib: l, synth: synth, open: true}
	v := &nativeVoice{state: state}
	// A voice nothing refers to any more is freed, even if nobody closes it.
	runtime.AddCleanup(v, func(s *nativeHandle) { s.close() }, state)
	return v, nil
}

// cString is s as a NUL-terminated C string.
func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// nativeHandle is a synthesizer the library holds. It is kept apart from the
// nativeVoice that wraps it so it can be freed when the wrapper is collected.
type nativeHandle struct {
	// mu serializes the synthesizer: a synthesis is a start followed by calls
	// for each chunk, all on state the synthesizer holds.
	mu    sync.Mutex
	lib   *library
	synth uintptr
	open  bool
}

// close frees the synthesizer, once.
func (h *nativeHandle) close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.open {
		return
	}
	h.open = false
	h.lib.free(h.synth)
}

// nativeVoice is a voice loaded into libpiper.
type nativeVoice struct {
	state *nativeHandle
}

// synthesize synthesizes text with opts applied over the voice's defaults,
// handing yield each chunk of samples, in [-1, 1] as the model produces them,
// and its rate. One synthesis holds the voice until it ends.
func (v *nativeVoice) synthesize(
	ctx context.Context, text string, opts synthesisOptions, yield func(samples []float32, sampleRate int) error,
) error {
	h := v.state
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.open {
		return errVoiceClosed
	}
	options := h.lib.defaultSynthesizeOptions(h.synth)
	opts.applyTo(&options)

	cText := cString(text)
	if code := h.lib.synthesizeStart(h.synth, &cText[0], &options); code != codeOK {
		return fmt.Errorf("%w: start returned %d", errSynthesize, code)
	}
	for {
		var chunk audioChunk
		code := h.lib.synthesizeNext(h.synth, &chunk)
		if code != codeOK && code != codeDone {
			return fmt.Errorf("%w: next returned %d", errSynthesize, code)
		}
		if chunk.numSamples > 0 && chunk.samples != nil {
			// The library reuses the chunk's memory on the next call, so the
			// samples are copied out first.
			//nolint:gosec // the samples the library returned
			samples := append([]float32(nil), unsafe.Slice(chunk.samples, chunk.numSamples)...)
			if err := yield(samples, int(chunk.sampleRate)); err != nil {
				return err
			}
		}
		if chunk.isLast || code == codeDone {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

// close frees the voice.
func (v *nativeVoice) close() { v.state.close() }

// applyTo sets the options given on the library's options, leaving the rest
// as the voice's defaults.
func (o synthesisOptions) applyTo(options *synthesizeOptions) {
	if o.speakerID != nil {
		options.speakerID = int32(*o.speakerID) //nolint:gosec // a speaker index fits
	}
	if o.lengthScale != nil {
		options.lengthScale = float32(*o.lengthScale)
	}
	if o.noiseScale != nil {
		options.noiseScale = float32(*o.noiseScale)
	}
	if o.noiseWScale != nil {
		options.noiseWScale = float32(*o.noiseWScale)
	}
}
