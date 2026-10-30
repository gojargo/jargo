package vad

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

var errConfidenceMismatch = errors.New("vad: confidence mismatch")

// toneFrames builds n frames of 512 mono 16-bit samples of a continuous 300 Hz
// sine at amplitude 0.5; silenceFrames builds n frames of zeros.
func toneFrames(n int) [][]byte {
	frames := make([][]byte, n)
	for f := range n {
		buf := make([]byte, 512*2)
		for i := range 512 {
			global := f*512 + i
			v := 0.5 * math.Sin(2*math.Pi*300*float64(global)/16000)
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(int16(v*32767)))
		}
		frames[f] = buf
	}
	return frames
}

func silenceFrames(n int) [][]byte {
	frames := make([][]byte, n)
	for f := range n {
		frames[f] = make([]byte, 512*2)
	}
	return frames
}

// speechFrames returns the first n frames of 512 samples of the 16 kHz speech
// fixture.
func speechFrames(t *testing.T, n int) [][]byte {
	t.Helper()
	pcm := readSpeechPCM(t)
	frames := make([][]byte, n)
	for f := range n {
		frames[f] = pcm[f*512*2 : (f+1)*512*2]
	}
	return frames
}

// TestSileroMatchesReference verifies the Silero confidence sequence against
// reference values, confirming the context and recurrent-state management
// match. The values come from running the embedded model in a standalone
// onnxruntime script over the same frames, zeroing the state and the 64-sample
// context at the start and carrying both from one frame to the next;
// regenerate them that way whenever the model changes. It needs the ONNX
// runtime.
func TestSileroMatchesReference(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	const tol = 1e-3
	toneWant := []float64{0.002657, 0.001875, 0.001425, 0.001144, 0.000907, 0.000863, 0.000693, 0.000712}
	silenceWant := []float64{0.001670, 0.006884, 0.008911, 0.007857, 0.005907, 0.005961, 0.005853, 0.005640}
	speechWant := []float64{
		0.423911, 0.939346, 0.951095, 0.988661, 0.977613, 0.985255, 0.999242, 0.999576,
		0.999373, 0.999875, 0.999624, 0.999816, 0.999824, 0.999856, 0.999968, 0.999962,
	}
	cases := []struct {
		name   string
		frames [][]byte
		want   []float64
	}{
		{"tone", toneFrames(8), toneWant},
		{"silence", silenceFrames(8), silenceWant},
		{"speech", speechFrames(t, 16), speechWant},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s, err := NewSilero()
			if err != nil {
				t.Fatalf("NewSilero: %v", err)
			}
			defer func() { _ = s.Close() }()
			if err := s.SetSampleRate(16000); err != nil {
				t.Fatalf("SetSampleRate: %v", err)
			}

			for i, frame := range c.frames {
				got := s.voiceConfidence(frame)
				if math.Abs(got-c.want[i]) > tol {
					t.Errorf("frame %d confidence = %v, want %v (tol %v)", i, got, c.want[i], tol)
				}
			}
		})
	}
}

// TestSileroVoiceConfidenceSilence checks that voiceConfidence scores a raw
// 16-bit buffer and that silence is not confidently classified as speech.
func TestSileroVoiceConfidenceSilence(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	s, err := NewSilero()
	if err != nil {
		t.Fatalf("NewSilero: %v", err)
	}
	defer func() { _ = s.Close() }()
	if err := s.SetSampleRate(16000); err != nil {
		t.Fatalf("SetSampleRate: %v", err)
	}

	got := s.voiceConfidence(make([]byte, 512*2))
	if got < 0 || got >= 0.5 {
		t.Errorf("silence confidence = %v, want in [0, 0.5)", got)
	}
}

// TestPCMToFloat32 checks that the 16-bit to float conversion feeds the model
// samples normalized by 32768.
func TestPCMToFloat32(t *testing.T) {
	samples := []int16{0, 16384, -32768, 32767}
	pcm := make([]byte, len(samples)*2)
	for i, v := range samples {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(v))
	}

	got := pcmToFloat32(pcm)
	for i, v := range samples {
		if want := float32(v) / 32768.0; got[i] != want {
			t.Errorf("sample %d = %v, want %v", i, got[i], want)
		}
	}
}

// TestSileroSharedModel verifies that multiple Silero analyzers sharing a single
// SileroModel produce the exact same confidence sequence as dedicated sessions,
// and that closing an analyzer does not invalidate the shared session.
func TestSileroSharedModel(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	model, err := NewSileroModel()
	if err != nil {
		t.Fatalf("NewSileroModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	const tol = 1e-3
	toneWant := []float64{0.002657, 0.001875, 0.001425, 0.001144, 0.000907, 0.000863, 0.000693, 0.000712}
	frames := toneFrames(8)

	// Analyzer 1 using shared model
	s1, err := NewSilero(WithModel(model))
	if err != nil {
		t.Fatalf("NewSilero with shared model: %v", err)
	}
	if err := s1.SetSampleRate(16000); err != nil {
		t.Fatalf("SetSampleRate: %v", err)
	}

	for i, frame := range frames {
		got := s1.voiceConfidence(frame)
		if math.Abs(got-toneWant[i]) > tol {
			t.Errorf("s1 frame %d confidence = %v, want %v", i, got, toneWant[i])
		}
	}

	// Close s1; this should NOT close the shared model
	if err := s1.Close(); err != nil {
		t.Fatalf("s1.Close: %v", err)
	}

	// Analyzer 2 using the same shared model must still function properly
	s2, err := NewSilero(WithModel(model))
	if err != nil {
		t.Fatalf("NewSilero s2 with shared model: %v", err)
	}
	defer func() { _ = s2.Close() }()
	if err := s2.SetSampleRate(16000); err != nil {
		t.Fatalf("SetSampleRate: %v", err)
	}

	for i, frame := range frames {
		got := s2.voiceConfidence(frame)
		if math.Abs(got-toneWant[i]) > tol {
			t.Errorf("s2 frame %d confidence = %v, want %v", i, got, toneWant[i])
		}
	}
}

// TestSileroSharedModelConcurrency verifies that multiple concurrent goroutines
// can run inference through separate Silero analyzers backed by one SileroModel
// without data races or corruption.
func TestSileroSharedModelConcurrency(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	model, err := NewSileroModel()
	if err != nil {
		t.Fatalf("NewSileroModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	const (
		numWorkers = 20
		numFrames  = 8
		tol        = 1e-3
	)
	toneWant := []float64{0.002657, 0.001875, 0.001425, 0.001144, 0.000907, 0.000863, 0.000693, 0.000712}

	errs := make(chan error, numWorkers)
	for range numWorkers {
		go func() {
			s, err := NewSilero(WithModel(model))
			if err != nil {
				errs <- err
				return
			}
			defer func() { _ = s.Close() }()

			if err := s.SetSampleRate(16000); err != nil {
				errs <- err
				return
			}

			frames := toneFrames(numFrames)
			for i, frame := range frames {
				got := s.voiceConfidence(frame)
				if math.Abs(got-toneWant[i]) > tol {
					errs <- fmt.Errorf("%w: frame %d confidence = %v, want %v", errConfidenceMismatch, i, got, toneWant[i])
					return
				}
			}
			errs <- nil
		}()
	}

	for range numWorkers {
		if err := <-errs; err != nil {
			t.Errorf("worker failed: %v", err)
		}
	}
}

func TestSileroModelEdgeCases(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	model, err := NewSileroModel()
	if err != nil {
		t.Fatalf("NewSileroModel: %v", err)
	}

	s, err := NewSilero(WithModel(model), WithParams(DefaultParams()))
	if err != nil {
		t.Fatalf("NewSilero with model: %v", err)
	}
	_ = s.Close()

	// NewSilero with nil or uninitialized model returns an error.
	if _, err := NewSilero(WithModel(nil)); !errors.Is(err, errClosedModel) {
		t.Errorf("NewSilero(WithModel(nil)) err = %v, want %v", err, errClosedModel)
	}

	emptyModel := &SileroModel{}
	if _, err := NewSilero(WithModel(emptyModel)); !errors.Is(err, errClosedModel) {
		t.Errorf("NewSilero(WithModel(emptyModel)) err = %v, want %v", err, errClosedModel)
	}

	var nilModel *SileroModel
	if err := nilModel.Close(); err != nil {
		t.Errorf("nilModel.Close(): %v", err)
	}
	if err := emptyModel.Close(); err != nil {
		t.Errorf("emptyModel.Close(): %v", err)
	}

	if err := model.Close(); err != nil {
		t.Errorf("model.Close(): %v", err)
	}

	// Model is now closed; NewSilero using it must return an error.
	if _, err := NewSilero(WithModel(model)); !errors.Is(err, errClosedModel) {
		t.Errorf("NewSilero with closed model err = %v, want %v", err, errClosedModel)
	}
}
