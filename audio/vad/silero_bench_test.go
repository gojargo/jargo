package vad

import (
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// BenchmarkNewSilero_Dedicated measures creating new Silero analyzers with a dedicated session each time (Before).
func BenchmarkNewSilero_Dedicated(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s, err := NewSilero()
		if err != nil {
			b.Fatalf("NewSilero: %v", err)
		}
		_ = s.Close()
	}
}

// BenchmarkNewSilero_Shared measures creating new Silero analyzers reusing a shared SileroModel (After).
func BenchmarkNewSilero_Shared(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	model, err := NewSileroModel()
	if err != nil {
		b.Fatalf("NewSileroModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := NewSilero(WithModel(model))
		if err != nil {
			b.Fatalf("NewSilero: %v", err)
		}
		_ = s.Close()
	}
}

// BenchmarkSilero_Inference_Dedicated measures single-frame inference latency with a dedicated session.
func BenchmarkSilero_Inference_Dedicated(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	s, err := NewSilero()
	if err != nil {
		b.Fatalf("NewSilero: %v", err)
	}
	defer func() { _ = s.Close() }()
	_ = s.SetSampleRate(16000)
	frame := toneFrames(1)[0]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.voiceConfidence(frame)
	}
}

// BenchmarkSilero_Inference_Shared measures single-frame inference latency using a shared model.
func BenchmarkSilero_Inference_Shared(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	model, err := NewSileroModel()
	if err != nil {
		b.Fatalf("NewSileroModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	s, err := NewSilero(WithModel(model))
	if err != nil {
		b.Fatalf("NewSilero: %v", err)
	}
	defer func() { _ = s.Close() }()
	_ = s.SetSampleRate(16000)
	frame := toneFrames(1)[0]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.voiceConfidence(frame)
	}
}

// BenchmarkSilero_Concurrent_Shared measures throughput when concurrent
// streams evaluate frames against a single shared model.
func BenchmarkSilero_Concurrent_Shared(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	model, err := NewSileroModel()
	if err != nil {
		b.Fatalf("NewSileroModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	frame := toneFrames(1)[0]
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		s, err := NewSilero(WithModel(model))
		if err != nil {
			b.Error(err)
			return
		}
		defer func() { _ = s.Close() }()
		_ = s.SetSampleRate(16000)

		for pb.Next() {
			_ = s.voiceConfidence(frame)
		}
	})
}
