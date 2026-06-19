package vad

import (
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// BenchmarkSileroVoiceConfidence measures one Silero VAD inference over a single
// 512-sample (32 ms) frame, including the input-context handling and recurrent
// state feedback. It needs the ONNX runtime (set JARGO_ONNXRUNTIME_LIB).
func BenchmarkSileroVoiceConfidence(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	s, err := NewSilero()
	if err != nil {
		b.Fatalf("NewSilero: %v", err)
	}
	defer func() { _ = s.Close() }()
	if err := s.SetSampleRate(16000); err != nil {
		b.Fatalf("SetSampleRate: %v", err)
	}
	frame := toneFrames(1)[0]

	b.ReportAllocs()
	for b.Loop() {
		s.voiceConfidence(frame)
	}
}
