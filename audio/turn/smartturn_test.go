package turn

import (
	"math"
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// TestSmartTurnModelMatchesReference runs the embedded smart-turn-v3 model over
// features computed for the reference signal and checks the completion
// probability against the Python reference. It needs the ONNX runtime.
func TestSmartTurnModelMatchesReference(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}

	s, err := NewSmartTurnV3()
	if err != nil {
		t.Fatalf("NewSmartTurnV3: %v", err)
	}
	defer func() { _ = s.Close() }()

	features := computeLogMel(refAudio())
	prob, err := s.runModel(features)
	if err != nil {
		t.Fatalf("runModel: %v", err)
	}

	const (
		want = 0.980147
		tol  = 1e-3
	)
	if math.Abs(prob-want) > tol {
		t.Errorf("smart-turn probability = %v, want %v (tol %v)", prob, want, tol)
	}
}
