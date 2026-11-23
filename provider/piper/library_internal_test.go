package piper

import (
	"bufio"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/onnxrt"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for the libpiper binding. The first ones load nothing; the last runs
// the real library, and skips when JARGO_PIPER_LIB does not name one.

// The structs are read at the offsets the library writes them at on a 64-bit
// platform: the options are 16 bytes, a chunk 72 with its samples first.
func TestStructLayoutsMatchTheLibrary(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("the layouts are checked for 64-bit platforms")
	}
	var o synthesizeOptions
	var c audioChunk
	for _, tc := range []struct {
		name      string
		got, want uintptr
	}{
		{"sizeof(piper_synthesize_options)", unsafe.Sizeof(o), 16},
		{"speaker_id", unsafe.Offsetof(o.speakerID), 0},
		{"length_scale", unsafe.Offsetof(o.lengthScale), 4},
		{"noise_scale", unsafe.Offsetof(o.noiseScale), 8},
		{"noise_w_scale", unsafe.Offsetof(o.noiseWScale), 12},
		{"sizeof(piper_audio_chunk)", unsafe.Sizeof(c), 72},
		{"samples", unsafe.Offsetof(c.samples), 0},
		{"num_samples", unsafe.Offsetof(c.numSamples), 8},
		{"sample_rate", unsafe.Offsetof(c.sampleRate), 16},
		{"is_last", unsafe.Offsetof(c.isLast), 20},
		{"phonemes", unsafe.Offsetof(c.phonemes), 24},
		{"num_phonemes", unsafe.Offsetof(c.numPhonemes), 32},
		{"phoneme_ids", unsafe.Offsetof(c.phonemeIDs), 40},
		{"num_phoneme_ids", unsafe.Offsetof(c.numPhonemeIDs), 48},
		{"alignments", unsafe.Offsetof(c.alignments), 56},
		{"num_alignments", unsafe.Offsetof(c.numAlignments), 64},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// Only the options given replace the voice's defaults.
func TestSynthesisOptionsApplyOverTheDefaults(t *testing.T) {
	defaults := synthesizeOptions{speakerID: 0, lengthScale: 1, noiseScale: 0.667, noiseWScale: 0.8}
	speaker, length := 3, 1.25

	got := defaults
	synthesisOptions{speakerID: &speaker, lengthScale: &length}.applyTo(&got)
	want := synthesizeOptions{speakerID: 3, lengthScale: 1.25, noiseScale: 0.667, noiseWScale: 0.8}
	if got != want {
		t.Errorf("options = %+v, want %+v", got, want)
	}

	got = defaults
	synthesisOptions{}.applyTo(&got)
	if got != defaults {
		t.Errorf("options = %+v, want the defaults", got)
	}
}

func TestCStringIsTerminated(t *testing.T) {
	if b := cString("abc"); string(b) != "abc\x00" {
		t.Errorf("cString = %q", b)
	}
}

// The espeak-ng data is looked for beside the library's lib directory, as
// libpiper installs it.
func TestEspeakDataDirIsBesideTheLibrary(t *testing.T) {
	root := t.TempDir()
	lib := filepath.Join(root, "lib", "libpiper.so")
	if got := espeakDataDir(lib); got != "" {
		t.Errorf("espeakDataDir with no data = %q, want none", got)
	}
	data := filepath.Join(root, "share", "espeak-ng-data")
	if err := os.MkdirAll(data, 0o750); err != nil {
		t.Fatal(err)
	}
	if got := espeakDataDir(lib); got != data {
		t.Errorf("espeakDataDir = %q, want %q", got, data)
	}
	if got := espeakDataDir(""); got != "" {
		t.Errorf("espeakDataDir with no library path = %q, want none", got)
	}
}

// A configured library that is not there is reported as unavailable before
// anything is downloaded.
func TestAMissingLibraryIsNotAvailable(t *testing.T) {
	_, err := openLibrary(filepath.Join(t.TempDir(), "libpiper.so"))
	if !errors.Is(err, ErrLibraryNotAvailable) {
		t.Fatalf("err = %v, want ErrLibraryNotAvailable", err)
	}
	dir := t.TempDir()
	_, err = NewTTS(Config{
		Voice:       "en_US-ryan-high",
		LibraryPath: filepath.Join(t.TempDir(), "libpiper.so"),
		DownloadDir: dir,
	})
	if !errors.Is(err, ErrLibraryNotAvailable) {
		t.Fatalf("NewTTS err = %v, want ErrLibraryNotAvailable", err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Errorf("%d files were downloaded for a missing library", len(entries))
	}
}

// The real library synthesizes speech with a real voice, downloaded into the
// user's cache the first time. With the ONNX Runtime the VAD uses loaded in the
// same process, both share one copy of it.
func TestRealLibrarySynthesizes(t *testing.T) {
	lib := os.Getenv(LibPathEnv)
	if lib == "" {
		t.Skipf("%s does not name a libpiper", LibPathEnv)
	}
	if testing.Short() {
		t.Skip("downloads a voice")
	}
	// The VAD's runtime is loaded first, when there is one, as a bot building
	// its VAD before its TTS does.
	ortLoaded := onnxrt.Init() == nil

	_, s, err := newTTS(Config{Voice: "en_US-ryan-high", LibraryPath: lib}, nativeDependencies())
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	defer func() { _ = s.Close() }()
	if s.SampleRate() != 22050 {
		t.Errorf("sample rate = %d, want en_US-ryan-high's 22050", s.SampleRate())
	}

	for _, length := range []float64{1, 2} {
		s.live.LengthScale = settings.Set(length)
		var samples int
		err := s.runTTS(context.Background(), "Hello from Piper. This is a test.", func(f frames.Frame) error {
			if a, ok := f.(*frames.TTSAudioRawFrame); ok {
				samples += len(a.Audio) / 2
			}
			return nil
		})
		if err != nil {
			t.Fatalf("runTTS: %v", err)
		}
		t.Logf("length scale %v: %d samples (%.2f s)", length, samples, float64(samples)/22050)
		if samples < 22050/2 {
			t.Errorf("length scale %v: %d samples, want at least half a second of speech", length, samples)
		}
	}

	if ortLoaded {
		copies := loadedCopies(t, "libonnxruntime")
		t.Logf("ONNX Runtime files mapped: %v", copies)
		if len(copies) != 1 {
			t.Errorf("ONNX Runtime files mapped = %v, want one", copies)
		}
	}
}

// loadedCopies is the distinct files whose name contains name that the
// process has mapped.
func loadedCopies(t *testing.T, name string) []string {
	t.Helper()
	f, err := os.Open("/proc/self/maps")
	if err != nil {
		t.Skipf("no /proc/self/maps: %v", err)
	}
	defer func() { _ = f.Close() }()
	seen := map[string]bool{}
	var out []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 6 || !strings.Contains(fields[5], name) || seen[fields[5]] {
			continue
		}
		seen[fields[5]] = true
		out = append(out, fields[5])
	}
	return out
}
