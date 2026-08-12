package turn

import (
	"math"
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// TestSmartTurnModelMatchesReference runs the embedded smart-turn-v3 model over
// features computed for the reference signal and checks the completion
// probability against the Python reference. It needs the ONNX runtime.
//
// The tolerance is wide because of what this test can actually pin down. The
// features are jargo's own arithmetic, and TestComputeLogMel holds them to 1e-6
// against the same reference; everything after that is the runtime's inference,
// which picks its kernels from what the host CPU offers. The same model over the
// same features drifts by a few thousandths between machines, so a tolerance
// tight enough to catch that would fail on the next processor rather than on a
// defect. What is left to check here is that the model is wired to the features
// correctly, and a wiring error moves the probability by tenths, not
// thousandths.
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
		tol  = 1e-2
	)
	if math.Abs(prob-want) > tol {
		t.Errorf("smart-turn probability = %v, want %v (tol %v)", prob, want, tol)
	}
	// And the decision the pipeline actually takes from it.
	if prob <= 0.5 {
		t.Errorf("smart-turn probability = %v, want the turn read as complete", prob)
	}
}
