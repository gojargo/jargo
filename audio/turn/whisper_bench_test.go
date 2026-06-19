package turn

import "testing"

// BenchmarkComputeLogMel measures the pure-Go Whisper log-mel feature
// extraction over a 2-second utterance (padded to the model's 8-second window).
// This is the fairest language-to-language comparison against Pipecat, since
// both implement the same math (jargo in Go, Pipecat in numpy).
func BenchmarkComputeLogMel(b *testing.B) {
	audio := refAudio()
	b.ReportAllocs()
	for b.Loop() {
		_ = computeLogMel(audio)
	}
}
