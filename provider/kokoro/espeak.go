package kokoro

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The espeak-ng constants the binding passes.
const (
	// audioOutputSynchronous is AUDIO_OUTPUT_SYNCHRONOUS: no audio device is
	// opened, which is all phonemization needs.
	audioOutputSynchronous int32 = 0x02
	// initializeDontExit is espeakINITIALIZE_DONT_EXIT: a library that cannot
	// read its data reports it rather than exiting the process.
	initializeDontExit int32 = 0x8000
	// textModeUTF8 is espeakCHARS_UTF8, the encoding of the text handed over.
	textModeUTF8 int32 = 1
	// nPathHome is the size of the buffer espeak keeps its data path in; a
	// longer path is silently replaced by its default.
	nPathHome = 160
)

// Errors the espeak-ng binding reports.
var (
	// ErrEspeakNotAvailable is returned when libespeak-ng cannot be loaded, so
	// a caller can tell a missing install from a failure of the model.
	ErrEspeakNotAvailable = errors.New("kokoro: libespeak-ng not available")
	// errEspeakVersion reports an espeak older than phonemization needs.
	errEspeakVersion = errors.New("kokoro: espeak-ng is too old, espeak>=1.48 is required")
	// errEspeakInit reports a library that could not be initialized.
	errEspeakInit = errors.New("kokoro: failed to initialize espeak shared library")
	// errEspeakData reports a data directory espeak cannot read.
	errEspeakData = errors.New("kokoro: no espeak data found")
	// errEspeakDataMismatch reports a library already initialized with other
	// data than a later service asks for.
	errEspeakDataMismatch = errors.New("kokoro: espeak-ng is already initialized with another data path")
	// errUnsupportedLanguage reports a language espeak has no voice for.
	errUnsupportedLanguage = errors.New("kokoro: language is not supported by the espeak backend")
	// errVoiceLoad reports a voice espeak failed to load.
	errVoiceLoad = errors.New("kokoro: failed to load espeak voice")
)

// espeakVoiceC is a voice as espeak describes it (espeak_VOICE). Only the
// name, the languages and the identifier are read; the rest is laid out so the
// struct has the library's size.
type espeakVoiceC struct {
	name       *byte
	languages  *byte
	identifier *byte
	_          [4]uint8       // gender, age, variant, xx1
	_          int32          // score
	_          unsafe.Pointer // spare
}

// espeakVoice is one voice espeak lists: its name, the language it speaks (the
// first it lists, the one it is the best voice for) and its identifier, the
// voice file relative to the data directory.
type espeakVoice struct {
	name       string
	language   string
	identifier string
}

// espeakLibrary is libespeak-ng, with the calls phonemization makes bound and
// the library initialized.
//
// espeak holds process-wide state: the voice in use, and the buffer the
// phonemes are written to. Every use goes through mu, and a phonemization sets
// its voice before it starts, so services speaking different languages share
// the one library.
type espeakLibrary struct {
	mu sync.Mutex

	info            func(pathData **byte) *byte
	initialize      func(output, bufLength int32, path *byte, options int32) int32
	listVoices      func(spec unsafe.Pointer) **espeakVoiceC
	setVoiceByName  func(name *byte) int32
	textToPhonemesC func(textPtr *unsafe.Pointer, textMode, phonemeMode int32) *byte

	// version is the library's version, as many numbers as it reports.
	version []int
	// dataPath is the data path the library was initialized with, "" for its
	// default.
	dataPath string
	// voice is the language code of the voice set last, "" for none.
	voice string
	// languages is the voices by language, each language with the first voice
	// listed for it, which espeak sorts by relevance.
	languages map[string]espeakVoice
}

// The libraries opened so far, by the path they were asked for, so every
// service naming the same library binds and initializes it once.
//
//nolint:gochecknoglobals // the loaded library is process-wide
var (
	espeakLibrariesMu sync.Mutex
	espeakLibraries   = map[string]*espeakLibrary{}
)

// espeakLibraryNames is the names libespeak-ng goes by on this platform, looked
// for on the loader's default search path when no path is configured. The
// versioned name comes first: a distribution's runtime package installs only
// that one, the unversioned link being part of its development package. The
// original espeak is looked for last.
func espeakLibraryNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"libespeak-ng.1.dylib", "libespeak-ng.dylib", "libespeak.dylib"}
	case "windows":
		return []string{"libespeak-ng.dll", "espeak-ng.dll", "libespeak.dll"}
	default:
		return []string{"libespeak-ng.so.1", "libespeak-ng.so", "libespeak.so.1", "libespeak.so"}
	}
}

// openEspeak loads libespeak-ng from libPath, or by its conventional names when
// libPath is empty, and initializes it with the data in dataPath, or its own
// default data when dataPath is empty. A library already loaded is reused.
func openEspeak(libPath, dataPath string) (*espeakLibrary, error) {
	espeakLibrariesMu.Lock()
	defer espeakLibrariesMu.Unlock()
	if lib, ok := espeakLibraries[libPath]; ok {
		if lib.dataPath != dataPath {
			return nil, fmt.Errorf("%w: %q, not %q", errEspeakDataMismatch, lib.dataPath, dataPath)
		}
		return lib, nil
	}
	handle, err := dlopenEspeak(libPath)
	if err != nil {
		return nil, err
	}
	lib := &espeakLibrary{dataPath: dataPath}
	for name, fn := range map[string]any{
		"espeak_Info":           &lib.info,
		"espeak_Initialize":     &lib.initialize,
		"espeak_ListVoices":     &lib.listVoices,
		"espeak_SetVoiceByName": &lib.setVoiceByName,
		"espeak_TextToPhonemes": &lib.textToPhonemesC,
	} {
		sym, err := purego.Dlsym(handle, name)
		if err != nil {
			return nil, fmt.Errorf("%w: %s: %w", ErrEspeakNotAvailable, name, err)
		}
		purego.RegisterFunc(fn, sym)
	}
	if err := lib.init(); err != nil {
		return nil, err
	}
	espeakLibraries[libPath] = lib
	return lib, nil
}

// dlopenEspeak opens the library at path, which must exist, or the first of
// its conventional names the loader finds.
func dlopenEspeak(path string) (uintptr, error) {
	candidates := espeakLibraryNames()
	if path != "" {
		if _, err := os.Stat(path); err != nil {
			return 0, fmt.Errorf("%w: %w", ErrEspeakNotAvailable, err)
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
	return 0, fmt.Errorf("%w: tried %v: %w", ErrEspeakNotAvailable, candidates, last)
}

// init checks the library's version, initializes it with its data, and lists
// its voices.
func (l *espeakLibrary) init() error {
	version, err := parseEspeakVersion(goString(l.info(new(*byte))))
	if err != nil {
		return err
	}
	if compareVersions(version, []int{1, 48}) < 0 {
		return fmt.Errorf("%w, got %v", errEspeakVersion, version)
	}
	l.version = version

	var path *byte
	var pinned []byte
	if l.dataPath != "" {
		arg, err := espeakDataPathArgument(l.dataPath, version)
		if err != nil {
			return err
		}
		pinned = cString(arg)
		path = &pinned[0]
	}
	// The library is asked not to exit the process when its data cannot be
	// read; whether it could is checked below, from where it says it looked.
	if l.initialize(audioOutputSynchronous, 0, path, initializeDontExit) <= 0 {
		return errEspeakInit
	}
	runtime.KeepAlive(pinned)
	var dataDir *byte
	l.info(&dataDir)
	if dir := goString(dataDir); dir != "" {
		if _, err := os.Stat(filepath.Join(dir, "phontab")); err != nil {
			return fmt.Errorf("%w in %s: %w", errEspeakData, dir, err)
		}
	}

	l.languages = map[string]espeakVoice{}
	for _, v := range l.availableVoices() {
		// The mbrola variants are listed under the same language as the plain
		// voice and can sort first, but need the mbrola binary, which is not
		// part of an espeak installation.
		if strings.HasPrefix(strings.ReplaceAll(v.identifier, "\\", "/"), "mb/") {
			continue
		}
		if _, ok := l.languages[v.language]; !ok {
			l.languages[v.language] = v
		}
	}
	return nil
}

// availableVoices lists the voices espeak has, in the order it sorts them.
func (l *espeakLibrary) availableVoices() []espeakVoice {
	list := l.listVoices(nil)
	var voices []espeakVoice
	for i := 0; list != nil; i++ {
		//nolint:gosec // the NULL-terminated array of voices the library returned
		entry := *(**espeakVoiceC)(unsafe.Add(unsafe.Pointer(list), uintptr(i)*unsafe.Sizeof(list)))
		if entry == nil {
			break
		}
		// The languages are a priority byte, then the language; only the
		// first language is read.
		lang := goString(entry.languages)
		if lang != "" {
			lang = lang[1:]
		}
		voices = append(voices, espeakVoice{
			name:       strings.ReplaceAll(goString(entry.name), "_", " "),
			language:   lang,
			identifier: goString(entry.identifier),
		})
	}
	return voices
}

// supportedLanguage is the language code espeak lists matching language, which
// is compared exactly, then ignoring case. It is "" when espeak has no voice
// for it.
func (l *espeakLibrary) supportedLanguage(language string) string {
	if _, ok := l.languages[language]; ok {
		return language
	}
	var matches []string
	for code := range l.languages {
		if strings.EqualFold(code, language) {
			matches = append(matches, code)
		}
	}
	if len(matches) == 1 {
		return matches[0]
	}
	return ""
}

// setVoice sets the voice espeak phonemizes with to the first one it lists
// for language, a code supportedLanguage returned. The caller holds mu.
func (l *espeakLibrary) setVoice(language string) error {
	if l.voice == language {
		return nil
	}
	v, ok := l.languages[language]
	if !ok {
		return fmt.Errorf("%w: invalid voice code %q", errVoiceLoad, language)
	}
	name := cString(v.identifier)
	if l.setVoiceByName(&name[0]) != 0 {
		return fmt.Errorf("%w %q", errVoiceLoad, language)
	}
	l.voice = language
	return nil
}

// phonemeMode is the phoneme mode espeak_TextToPhonemes is called with: IPA,
// with an underscore between the phonemes of a word. espeak up to 1.48.3 asks
// for it its own way.
func (l *espeakLibrary) phonemeMode() int32 {
	if compareVersions(l.version, []int{1, 48, 3}) <= 0 {
		return 0x03 | 0x01<<4
	}
	return int32('_')<<8 | 0x02
}

// textToPhonemes phonemizes text with the voice set, a clause at a time, and
// joins the clauses with a space. The caller holds mu.
func (l *espeakLibrary) textToPhonemes(text string) string {
	buf := cString(text)
	var pinner runtime.Pinner
	pinner.Pin(&buf[0])
	defer pinner.Unpin()
	// espeak advances the pointer past each clause it reads, and sets it to
	// NULL at the end of the text.
	ptr := unsafe.Pointer(&buf[0]) //nolint:gosec // the pinned text espeak reads
	mode := l.phonemeMode()
	var result []string
	for ptr != nil {
		if phonemes := goString(l.textToPhonemesC(&ptr, textModeUTF8, mode)); phonemes != "" {
			result = append(result, phonemes)
		}
	}
	return strings.Join(result, " ")
}

// espeakDataPathArgument is the data path to hand espeak_Initialize for
// dataPath, the espeak data directory or its parent. A path longer than
// espeak's buffer is replaced by a link to the data from a temporary
// directory, as espeak would silently use its default data otherwise.
func espeakDataPathArgument(dataPath string, version []int) (string, error) {
	var data string
	switch {
	case compareVersions(version, []int{1, 49}) < 0:
		if filepath.Base(dataPath) == "espeak-data" {
			dataPath = filepath.Dir(dataPath)
		}
		data = filepath.Join(dataPath, "espeak-data")
	case isFile(filepath.Join(dataPath, "phontab")):
		data = dataPath
	default:
		data = filepath.Join(dataPath, "espeak-ng-data")
	}
	if !isFile(filepath.Join(data, "phontab")) {
		return "", fmt.Errorf("%w in %s", errEspeakData, dataPath)
	}
	if len(data) >= nPathHome {
		dir, err := os.MkdirTemp("", "jargo-espeak-")
		if err != nil {
			return "", fmt.Errorf("kokoro: the path to the espeak data is too long for espeak (%d bytes at most) "+
				"and a shorter link to it cannot be made: %w", nPathHome-1, err)
		}
		name := "espeak-ng-data"
		if compareVersions(version, []int{1, 49}) < 0 {
			name = "espeak-data"
		}
		if err := os.Symlink(data, filepath.Join(dir, name)); err != nil {
			return "", fmt.Errorf("kokoro: the path to the espeak data is too long for espeak (%d bytes at most) "+
				"and a shorter link to it cannot be made: %w", nPathHome-1, err)
		}
		return dir, nil
	}
	return dataPath, nil
}

// isFile reports whether path is a regular file.
func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

// parseEspeakVersion reads the version espeak_Info reports, such as
// "1.52.0" or "1.49.3-dev", as its numbers.
func parseEspeakVersion(info string) ([]int, error) {
	fields := strings.Fields(info)
	if len(fields) == 0 {
		return nil, fmt.Errorf("%w: no version reported", ErrEspeakNotAvailable)
	}
	var version []int
	for part := range strings.SplitSeq(strings.ReplaceAll(fields[0], "-dev", ""), ".") {
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%w: version %q: %w", ErrEspeakNotAvailable, fields[0], err)
		}
		version = append(version, n)
	}
	return version, nil
}

// compareVersions compares two versions number by number, the way tuples
// compare: a version that is a prefix of the other is the smaller.
func compareVersions(a, b []int) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			if a[i] < b[i] {
				return -1
			}
			return 1
		}
	}
	switch {
	case len(a) < len(b):
		return -1
	case len(a) > len(b):
		return 1
	}
	return 0
}

// cString is s as a NUL-terminated C string.
func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
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
