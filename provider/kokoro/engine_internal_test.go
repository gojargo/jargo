package kokoro

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// Tests for the model driver: the batching, the trimming, the voices file,
// and, when the model files and the runtimes are present, a real synthesis.

func TestSplitPhonemesShortTextIsOneBatch(t *testing.T) {
	if got := splitPhonemes("  həlˈoʊ, wˈɜːld! ", 510); !reflect.DeepEqual(got, []string{"həlˈoʊ, wˈɜːld!"}) {
		t.Errorf("splitPhonemes = %q", got)
	}
	if got := splitPhonemes("   ", 510); got != nil {
		t.Errorf("splitPhonemes of nothing = %q", got)
	}
}

// Long text is cut at sentences first, then clauses, then words, and the
// batches are balanced rather than filled to the limit.
func TestSplitPhonemesPrefersTheLeastDisruptiveCut(t *testing.T) {
	got := splitPhonemes("aaa bbb. ccc, ddd eee.", 10)
	want := []string{"aaa bbb.", "ccc,", "ddd eee."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitPhonemes = %q, want %q", got, want)
	}
	// Balanced: two batches of about the same size, not one full and a short
	// remainder.
	got = splitPhonemes("aa bb cc dd ee ff gg", 15)
	want = []string{"aa bb cc dd", "ee ff gg"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitPhonemes = %q, want %q", got, want)
	}
	// An unbroken run longer than the limit is sliced, not dropped.
	got = splitPhonemes(strings.Repeat("ə", 25), 10)
	if len(got) != 3 || len([]rune(strings.Join(got, ""))) != 25 {
		t.Errorf("splitPhonemes of a long run = %q", got)
	}
	for _, b := range splitPhonemes(strings.Repeat("ab cd, ef. ", 200), 510) {
		if n := len([]rune(b)); n == 0 || n > 510 {
			t.Errorf("batch of %d phonemes", n)
		}
	}
}

func TestPauseAfter(t *testing.T) {
	for _, tc := range []struct {
		text string
		want float64
	}{{"a.", 0.25}, {"a?\n", 0.25}, {"a\u2026", 0.25}, {"a,", 0.1}, {"a;", 0.1}, {"a", 0}} {
		if got := pauseAfter(tc.text, 0.25, 0.1); got != tc.want {
			t.Errorf("pauseAfter(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

// The silence before and after the sound is trimmed, a hop at a time.
func TestTrimRemovesLeadingAndTrailingSilence(t *testing.T) {
	y := make([]float32, 24000)
	for i := 8000; i < 16000; i++ {
		y[i] = float32(0.5 * math.Sin(float64(i)/10))
	}
	out, span := trim(y)
	if span[0] > 8000 || span[1] < 16000 || span[0] < 8000-2048 || span[1] > 16000+2048 {
		t.Errorf("trimmed to %v, want around [8000 16000]", span)
	}
	if len(out) != span[1]-span[0] {
		t.Errorf("trimmed %d samples for %v", len(out), span)
	}
	// A silent signal is all at the reference level: nothing is trimmed.
	if out, span := trim(make([]float32, 4000)); len(out) != 4000 || span != [2]int{0, 4000} {
		t.Errorf("silence trimmed to %v", span)
	}
}

// npz builds a voices file holding the arrays given, as NumPy writes them.
func npz(t *testing.T, arrays map[string][]float32) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, data := range arrays {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: name + ".npy", Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, 1, 256), }", len(data)/styleDim)
		header += strings.Repeat(" ", 63-(10+len(header))%64) + "\n"
		var b bytes.Buffer
		b.WriteString("\x93NUMPY\x01\x00")
		_ = binary.Write(&b, binary.LittleEndian, uint16(len(header)))
		b.WriteString(header)
		_ = binary.Write(&b, binary.LittleEndian, data)
		if _, err := w.Write(b.Bytes()); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// A voice has one style per phoneme count: n phonemes use row n-1, and the
// last row past the end.
func TestVoicesStylePerLength(t *testing.T) {
	data := make([]float32, 3*styleDim)
	for row := range 3 {
		data[row*styleDim] = float32(row + 1)
	}
	file := npz(t, map[string][]float32{"af_test": data})
	vs, err := loadVoices(bytes.NewReader(file), int64(len(file)))
	if err != nil {
		t.Fatalf("loadVoices: %v", err)
	}
	v, ok := vs.lookup("af_test")
	if !ok {
		t.Fatal("af_test not found")
	}
	if _, ok := vs.lookup("af_test.npy"); !ok {
		t.Error("af_test.npy not found")
	}
	if _, ok := vs.lookup("af_nope"); ok {
		t.Error("af_nope found")
	}
	for length, want := range map[int]float32{1: 1, 2: 2, 3: 3, 50: 3} {
		if got := v.style(length)[0]; got != want || len(v.style(length)) != styleDim {
			t.Errorf("style(%d) starts with %v, want %v", length, got, want)
		}
	}
}

// fakeSession stands in for the model, returning a second of a tone.
type fakeSession struct{ inputs [][]onnxrt.Tensor }

func (f *fakeSession) Run(inputs []onnxrt.Tensor) ([]onnxrt.Tensor, error) {
	f.inputs = append(f.inputs, inputs)
	audio := make([]float32, SampleRate)
	for i := range audio {
		audio[i] = float32(0.3 * math.Sin(float64(i)/5))
	}
	return []onnxrt.Tensor{onnxrt.Float32([]int64{int64(len(audio))}, audio)}, nil
}

func (*fakeSession) Close() error { return nil }

// fakeTokenizer phonemizes text as itself.
type fakeTokenizer struct{ *tokenizer }

func (fakeTokenizer) phonemize(text, _ string) (string, error) { return text, nil }

// The model is fed the tokens padded with a boundary on either side, the style
// for their count and the speed; each batch but the last is followed by the
// pause its punctuation calls for.
func TestCreateStreamFeedsTheModel(t *testing.T) {
	sess := &fakeSession{}
	style := make([]float32, 2*styleDim)
	style[styleDim] = 7
	k := &kokoro{
		sess:      sess,
		voices:    voices{"af_test": {rows: 2, styles: style}},
		tokenizer: fakeTokenizer{&tokenizer{vocab: defaultVocab}},
	}
	text := strings.Repeat("ab. ", 200)
	var got [][]float32
	for audio, err := range k.createStream(t.Context(), text, "af_test", "en-us", 1.25) {
		if err != nil {
			t.Fatalf("createStream: %v", err)
		}
		got = append(got, audio)
	}
	if len(got) != 2 || len(sess.inputs) != 2 {
		t.Fatalf("%d batches, %d runs, want 2", len(got), len(sess.inputs))
	}
	in := sess.inputs[0]
	ids := in[0].I64
	if ids[0] != 0 || ids[len(ids)-1] != 0 || ids[1] != defaultVocab['a'] || in[0].Shape[0] != 1 {
		t.Errorf("tokens = %v", ids[:4])
	}
	if in[1].F32[0] != 7 || in[2].F32[0] != 1.25 {
		t.Errorf("style starts %v, speed %v", in[1].F32[0], in[2].F32)
	}
	// The first batch ends a sentence: a quarter second of silence follows it.
	pause := got[0][len(got[0])-SampleRate/4:]
	for _, v := range pause {
		if v != 0 {
			t.Fatal("no pause after the first batch")
		}
	}
	if len(got[1]) >= len(got[0]) {
		t.Errorf("the last batch is %d samples, the first %d: the last should have no pause", len(got[1]), len(got[0]))
	}

	// A speed out of range, an unknown voice and text with nothing to say are
	// errors.
	for _, tc := range []struct {
		text, voice string
		speed       float64
		want        error
	}{
		{"ab", "af_test", 2.5, errSpeed},
		{"ab", "af_nope", 1, errVoice},
		{"  ", "af_test", 1, errNoPhonemes},
		{"012", "af_test", 1, errNoTokens},
	} {
		var err error
		for _, e := range k.createStream(t.Context(), tc.text, tc.voice, "en-us", tc.speed) {
			err = e
		}
		if !errors.Is(err, tc.want) {
			t.Errorf("%q %s %v: err = %v, want %v", tc.text, tc.voice, tc.speed, err, tc.want)
		}
	}
}

// modelFiles is the model and voices files in the default cache, or a skip
// when they, the ONNX Runtime or espeak-ng are absent.
func modelFiles(t *testing.T) (string, string) {
	t.Helper()
	dir, err := DefaultCacheDir()
	if err != nil {
		t.Skipf("no cache directory: %v", err)
	}
	model, voicesFile := filepath.Join(dir, modelFileName), filepath.Join(dir, voicesFileName)
	for _, f := range []string{model, voicesFile} {
		if _, err := os.Stat(f); err != nil {
			t.Skipf("Kokoro model files not downloaded: %v", err)
		}
	}
	if !onnxrt.Available() {
		t.Skip("ONNX Runtime not available")
	}
	systemEspeak(t)
	return model, voicesFile
}

// A short English sentence synthesized with af_heart is speech: a second or
// more of audio at 24 kHz, and not silent.
func TestSynthesizeWithTheModel(t *testing.T) {
	modelPath, voicesPath := modelFiles(t)
	k, err := newKokoro(modelPath, voicesPath, EspeakConfig{})
	if err != nil {
		t.Fatalf("newKokoro: %v", err)
	}
	defer func() { _ = k.close() }()
	if !k.hasVoice("af_heart") {
		t.Fatal("af_heart is not in the voices file")
	}
	start := time.Now()
	var audio []float32
	batches := 0
	for samples, err := range k.createStream(t.Context(), "Hello! This is a short test of the Kokoro voice.",
		"af_heart", "en-us", 1.0) {
		if err != nil {
			t.Fatalf("createStream: %v", err)
		}
		audio = append(audio, samples...)
		batches++
	}
	elapsed := time.Since(start)
	var sum, peak float64
	for _, v := range audio {
		sum += float64(v) * float64(v)
		peak = max(peak, math.Abs(float64(v)))
	}
	rms := math.Sqrt(sum / float64(max(len(audio), 1)))
	duration := float64(len(audio)) / SampleRate
	t.Logf("%d batches, %.2fs of audio at %d Hz in %v: RMS %.4f, peak %.4f", batches, duration, SampleRate,
		elapsed.Round(time.Millisecond), rms, peak)
	if duration < 1 || duration > 10 {
		t.Errorf("duration %.2fs, want a sentence's", duration)
	}
	if rms < 0.01 || peak < 0.1 {
		t.Errorf("RMS %.4f, peak %.4f: the audio is silent", rms, peak)
	}
}
