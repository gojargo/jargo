package novasonic

import "testing"

// TestDefaultModel pins the model an empty Config selects. Bedrock removed the
// first-generation amazon.nova-sonic-v1:0 at its end of life and rejects
// requests naming it, so the default must be Nova 2 Sonic.
func TestDefaultModel(t *testing.T) {
	if got, want := New(Config{}).Model(), "amazon.nova-2-sonic-v1:0"; got != want {
		t.Errorf("Model() = %q, want %q", got, want)
	}
	if got := New(Config{Model: "custom-model"}).Model(); got != "custom-model" {
		t.Errorf("Model() = %q, want the configured model", got)
	}
}
