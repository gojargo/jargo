package turn

import (
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// BenchmarkSmartTurnPredict measures one end-of-turn prediction: Whisper feature
// extraction plus the smart-turn-v3 ONNX inference, over a 2-second utterance.
// It needs the ONNX runtime (set JARGO_ONNXRUNTIME_LIB).
func BenchmarkSmartTurnPredict(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	s, err := NewSmartTurnV3()
	if err != nil {
		b.Fatalf("NewSmartTurnV3: %v", err)
	}
	defer func() { _ = s.Close() }()
	s.SetSampleRate(16000)

	audio := make([]float32, 32000)
	copy(audio, refAudio())

	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := s.predictEndpoint(audio); err != nil {
			b.Fatal(err)
		}
	}
}
