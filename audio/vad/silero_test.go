package vad

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

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

// TestSileroMatchesReference verifies the Silero confidence sequence against
// values produced by the reference Python implementation, confirming the
// context and recurrent-state management match. It needs the ONNX runtime.
func TestSileroMatchesReference(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	const tol = 1e-3
	toneWant := []float64{0.118424, 0.091276, 0.038707, 0.023131, 0.010957, 0.006557, 0.003468, 0.002861}
	silenceWant := []float64{0.023829, 0.011602, 0.006769, 0.003017, 0.002068, 0.001625, 0.001439, 0.001337}
	cases := []struct {
		name   string
		frames [][]byte
		want   []float64
	}{
		{"tone", toneFrames(8), toneWant},
		{"silence", silenceFrames(8), silenceWant},
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
