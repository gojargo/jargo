package nvidia_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/nvidia"
)

// TestNewLLM checks the NVIDIA NIM OpenAI-compatible LLM shim wires the right
// service name and default model into the shared client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "NvidiaLLM", "nvidia/nemotron-3-super-120b-a12b", nvidia.NewLLM)
}
