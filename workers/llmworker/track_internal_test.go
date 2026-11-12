package llmworker

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/service/llm"
)

// TestAPanickingToolReleasesTheWorker checks a tool handler that panics still
// counts as finished: the LLM service recovers the panic and settles the call,
// and a call left counted would hold the worker's frames and handover back for
// good.
func TestAPanickingToolReleasesTheWorker(t *testing.T) {
	w := &Worker{}
	handler := w.track(func(context.Context, llm.FunctionCallParams) error {
		panic("tool bug") //nolint:forbidigo // the handler's bug under test
	})

	func() {
		defer func() { _ = recover() }()
		_ = handler(t.Context(), llm.FunctionCallParams{})
	}()

	if w.ToolCallActive() {
		t.Error("ToolCallActive = true after the handler panicked, want the call released")
	}
}
