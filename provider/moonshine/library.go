package moonshine

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// headerVersion is the version of the library's C ABI this binding is written
// against. It is handed to the library when a transcriber is loaded, and the
// library refuses a version it cannot serve.
const headerVersion int32 = 30000

// The status codes the library returns. A negative value is a failure; any
// other value is a success, or a handle.
const (
	codeUnknown         int32 = -1
	codeInvalidHandle   int32 = -2
	codeInvalidArgument int32 = -3
)

// Errors the library reports, by status code, and from loading it.
var (
	// ErrLibraryNotAvailable is returned when libmoonshine cannot be loaded, so
	// a caller can tell a missing install from a failure of the model.
	ErrLibraryNotAvailable = errors.New("moonshine: libmoonshine not available")
	// ErrUnknown is the library's unknown error.
	ErrUnknown = errors.New("moonshine: unknown error")
	// ErrInvalidHandle is the library being handed a transcriber it does not
	// hold.
	ErrInvalidHandle = errors.New("moonshine: invalid handle")
	// ErrInvalidArgument is the library being handed an argument it refuses.
	ErrInvalidArgument = errors.New("moonshine: invalid argument")
	// errUnknownCode is a status code the library does not document.
	errUnknownCode = errors.New("moonshine: unknown error code")
	// errLoadTranscriber reports a model the library could not load.
	errLoadTranscriber = errors.New("moonshine: failed to load transcriber")
	// errTranscriberClosed reports a transcriber used after it was freed.
	errTranscriberClosed = errors.New("moonshine: transcriber is not initialized")
)

// checkError is the error a status code stands for, nil for success.
func checkError(code int32) error {
	switch {
	case code >= 0:
		return nil
	case code == codeUnknown:
		return ErrUnknown
	case code == codeInvalidHandle:
		return ErrInvalidHandle
	case code == codeInvalidArgument:
		return ErrInvalidArgument
	default:
		return fmt.Errorf("%w: %d", errUnknownCode, code)
	}
}

// transcriberOption is one advanced transcriber option, a name and a value,
// as the library reads it (transcriber_option_t).
type transcriberOption struct {
	name  *byte
	value *byte
}

// transcriptC is a transcript as the library returns it (transcript_t).
type transcriptC struct {
	lines     *transcriptLineC
	lineCount uint64
}

// transcriptLineC is one line of a transcript as the library returns it
// (transcript_line_t). The service reads only the text; the other fields are
// laid out so the lines are read at the stride the library writes them.
type transcriptLineC struct {
	text                       *byte
	audioData                  *float32
	audioDataCount             uint64
	startTime                  float32
	duration                   float32
	id                         uint64
	isComplete                 int8
	isUpdated                  int8
	isNew                      int8
	hasTextChanged             int8
	haveSpeakersChanged        int8
	speakerSpans               unsafe.Pointer
	speakerSpanCount           uint64
	lastTranscriptionLatencyMs uint32
	words                      unsafe.Pointer
	wordCount                  uint64
}

// library is libmoonshine, with the calls the service makes bound.
type library struct {
	loadTranscriberFromFiles func(
		path string, arch uint32, options unsafe.Pointer, optionsCount uint64, version int32,
	) int32
	freeTranscriber            func(handle int32)
	transcribeWithoutStreaming func(
		handle int32, audio *float32, audioLength uint64, sampleRate int32, flags uint32, out **transcriptC,
	) int32
	errorToString func(code int32) string
}

// The libraries opened so far, by the path they were asked for, so every
// service naming the same library binds it once.
//
//nolint:gochecknoglobals // the loaded library is process-wide
var (
	librariesMu sync.Mutex
	libraries   = map[string]*library{}
)

// libraryNames is the names libmoonshine goes by on this platform, looked for
// on the loader's default search path when no path is configured.
func libraryNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"libmoonshine.dylib"}
	case "windows":
		return []string{"moonshine.dll"}
	default:
		return []string{"libmoonshine.so"}
	}
}

// openLibrary loads libmoonshine from path, or by its conventional name when
// path is empty, and binds the calls the service makes. A library already
// loaded is reused.
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
	lib := &library{}
	for name, fn := range map[string]any{
		"moonshine_load_transcriber_from_files":  &lib.loadTranscriberFromFiles,
		"moonshine_free_transcriber":             &lib.freeTranscriber,
		"moonshine_transcribe_without_streaming": &lib.transcribeWithoutStreaming,
		"moonshine_error_to_string":              &lib.errorToString,
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

// loadTranscriber loads the model at modelPath with the architecture and
// options given.
func (l *library) loadTranscriber(modelPath string, arch modelArch, options map[string]string) (transcriber, error) {
	// The option array and the strings it points at are Go memory handed to
	// the library, so they are pinned for the call.
	var pinner runtime.Pinner
	defer pinner.Unpin()
	opts := make([]transcriberOption, 0, len(options))
	for name, value := range options {
		n, v := cString(name), cString(value)
		pinner.Pin(&n[0])
		pinner.Pin(&v[0])
		opts = append(opts, transcriberOption{name: &n[0], value: &v[0]})
	}
	var optsPtr unsafe.Pointer
	if len(opts) > 0 {
		pinner.Pin(&opts[0])
		optsPtr = unsafe.Pointer(&opts[0]) //nolint:gosec // the option array the library reads, pinned
	}

	handle := l.loadTranscriberFromFiles(modelPath, uint32(arch), optsPtr, uint64(len(opts)), headerVersion)
	if handle < 0 {
		msg := l.errorToString(handle)
		if msg == "" {
			msg = "Unknown error"
		}
		return nil, fmt.Errorf("%w: %s: %w", errLoadTranscriber, msg, checkError(handle))
	}
	state := &nativeHandle{lib: l, handle: handle, open: true}
	t := &nativeTranscriber{state: state}
	// A transcriber nothing refers to any more is freed, so a model swapped
	// out is released even if nobody closes it.
	runtime.AddCleanup(t, func(s *nativeHandle) { s.free() }, state)
	return t, nil
}

// cString is s as a NUL-terminated C string.
func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// nativeHandle is a transcriber the library holds. It is kept apart from the
// nativeTranscriber that wraps it so it can be freed when the wrapper is
// collected.
type nativeHandle struct {
	mu     sync.Mutex
	lib    *library
	handle int32
	open   bool
}

// free releases the transcriber, once.
func (h *nativeHandle) free() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.open {
		return
	}
	h.open = false
	h.lib.freeTranscriber(h.handle)
}

// nativeTranscriber is a model loaded into libmoonshine.
type nativeTranscriber struct {
	state *nativeHandle
}

// transcribeWithoutStreaming transcribes audio, float samples in [-1, 1] at
// sampleRate, in one call. The transcript the library returns belongs to the
// transcriber and is overwritten by the next call, so its text is copied out
// before the transcriber is handed to anyone else.
func (t *nativeTranscriber) transcribeWithoutStreaming(audio []float32, sampleRate int) ([]string, error) {
	h := t.state
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.open {
		return nil, errTranscriberClosed
	}
	var samples *float32
	if len(audio) > 0 {
		samples = &audio[0]
	}
	var out *transcriptC
	code := h.lib.transcribeWithoutStreaming(
		h.handle, samples, uint64(len(audio)), int32(sampleRate), 0, &out, //nolint:gosec // a sample rate fits
	)
	if err := checkError(code); err != nil {
		return nil, err
	}
	return readTranscript(out), nil
}

// close frees the transcriber.
func (t *nativeTranscriber) close() { t.state.free() }

// readTranscript copies the text of each line of a transcript out of the
// library's memory, dropping any bytes that are not valid UTF-8.
func readTranscript(t *transcriptC) []string {
	if t == nil || t.lines == nil || t.lineCount == 0 {
		return nil
	}
	lines := unsafe.Slice(t.lines, t.lineCount) //nolint:gosec // the lines the library returned
	texts := make([]string, len(lines))
	for i := range lines {
		texts[i] = strings.ToValidUTF8(goString(lines[i].text), "")
	}
	return texts
}

// goString copies a NUL-terminated C string, "" for a nil one.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 { //nolint:gosec // a C string, read to its NUL
		n++
	}
	return string(unsafe.Slice(p, n)) //nolint:gosec // the n bytes before the NUL
}
