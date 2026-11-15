package moonshine

import (
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"unsafe"
)

// Tests for the libmoonshine binding that load nothing: the layouts the
// library's structs are read with, and what its status codes mean.

// The structs are read at the offsets the library writes them at on a 64-bit
// platform: a transcript is 16 bytes, a line 88 with its text first.
func TestStructLayoutsMatchTheLibrary(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("the layouts are checked for 64-bit platforms")
	}
	var tr transcriptC
	var line transcriptLineC
	var opt transcriberOption
	for _, tc := range []struct {
		name      string
		got, want uintptr
	}{
		{"sizeof(transcript_t)", unsafe.Sizeof(tr), 16},
		{"transcript_t.lines", unsafe.Offsetof(tr.lines), 0},
		{"transcript_t.line_count", unsafe.Offsetof(tr.lineCount), 8},
		{"sizeof(transcript_line_t)", unsafe.Sizeof(line), 88},
		{"text", unsafe.Offsetof(line.text), 0},
		{"audio_data", unsafe.Offsetof(line.audioData), 8},
		{"audio_data_count", unsafe.Offsetof(line.audioDataCount), 16},
		{"start_time", unsafe.Offsetof(line.startTime), 24},
		{"duration", unsafe.Offsetof(line.duration), 28},
		{"id", unsafe.Offsetof(line.id), 32},
		{"is_complete", unsafe.Offsetof(line.isComplete), 40},
		{"is_updated", unsafe.Offsetof(line.isUpdated), 41},
		{"is_new", unsafe.Offsetof(line.isNew), 42},
		{"has_text_changed", unsafe.Offsetof(line.hasTextChanged), 43},
		{"have_speakers_changed", unsafe.Offsetof(line.haveSpeakersChanged), 44},
		{"speaker_spans", unsafe.Offsetof(line.speakerSpans), 48},
		{"speaker_span_count", unsafe.Offsetof(line.speakerSpanCount), 56},
		{"last_transcription_latency_ms", unsafe.Offsetof(line.lastTranscriptionLatencyMs), 64},
		{"words", unsafe.Offsetof(line.words), 72},
		{"word_count", unsafe.Offsetof(line.wordCount), 80},
		{"sizeof(transcriber_option_t)", unsafe.Sizeof(opt), 16},
		{"transcriber_option_t.name", unsafe.Offsetof(opt.name), 0},
		{"transcriber_option_t.value", unsafe.Offsetof(opt.value), 8},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

func TestStatusCodesMapToErrors(t *testing.T) {
	for _, tc := range []struct {
		code int32
		want error
	}{
		{0, nil},
		{7, nil},
		{-1, ErrUnknown},
		{-2, ErrInvalidHandle},
		{-3, ErrInvalidArgument},
		{-42, errUnknownCode},
	} {
		if got := checkError(tc.code); !errors.Is(got, tc.want) || (tc.want == nil) != (got == nil) {
			t.Errorf("checkError(%d) = %v, want %v", tc.code, got, tc.want)
		}
	}
	if got := checkError(-42).Error(); got != "moonshine: unknown error code: -42" {
		t.Errorf("checkError(-42) = %q", got)
	}
}

func TestTheModelArchitecturesAreNumberedAsTheLibraryNumbersThem(t *testing.T) {
	for want, name := range []Model{
		ModelTiny, ModelBase, ModelTinyStreaming, ModelBaseStreaming, ModelSmallStreaming, ModelMediumStreaming,
	} {
		arch, err := stringToModelArch(string(name))
		if err != nil || uint32(arch) != uint32(want) {
			t.Errorf("%s = %d, %v; want %d", name, arch, err, want)
		}
		if arch.String() != string(name) {
			t.Errorf("arch %d is named %q, want %q", arch, arch.String(), name)
		}
	}
}

// cLine is a transcript line holding text as the library would.
func cLine(text string) transcriptLineC {
	if text == "\x00" {
		return transcriptLineC{}
	}
	b := cString(text)
	return transcriptLineC{text: &b[0]}
}

// The text of every line is copied out, a line with none reads as empty, and
// bytes that are not UTF-8 are dropped.
func TestReadTranscriptCopiesEachLine(t *testing.T) {
	lines := []transcriptLineC{cLine(" hello"), cLine("\x00"), cLine("th\xffere ")}
	got := readTranscript(&transcriptC{lines: &lines[0], lineCount: uint64(len(lines))})

	if want := []string{" hello", "", "there "}; !slices.Equal(got, want) {
		t.Errorf("lines = %q, want %q", got, want)
	}
	if got := readTranscript(nil); got != nil {
		t.Errorf("no transcript = %q", got)
	}
	if got := readTranscript(&transcriptC{}); got != nil {
		t.Errorf("an empty transcript = %q", got)
	}
}

func TestCStringIsTerminated(t *testing.T) {
	if b := cString("max_tokens_per_second"); b[len(b)-1] != 0 || goString(&b[0]) != "max_tokens_per_second" {
		t.Errorf("cString = %q", b)
	}
}

// A configured library that is not there is reported as unavailable before
// anything is loaded.
func TestAMissingLibraryIsNotAvailable(t *testing.T) {
	_, err := openLibrary(filepath.Join(t.TempDir(), "libmoonshine.so"))
	if !errors.Is(err, ErrLibraryNotAvailable) {
		t.Fatalf("err = %v, want ErrLibraryNotAvailable", err)
	}
	_, err = NewSTT(STTConfig{LibraryPath: filepath.Join(t.TempDir(), "libmoonshine.so")})
	if !errors.Is(err, ErrLibraryNotAvailable) {
		t.Fatalf("NewSTT err = %v, want ErrLibraryNotAvailable", err)
	}
}
