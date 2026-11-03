package live

import (
	"context"
	"net/http"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// gatedConnector dials url and reports that its deployment takes no
// non-blocking tool calls, as the Vertex AI endpoint does.
type gatedConnector struct{ url string }

func (c gatedConnector) Endpoint(context.Context) (string, http.Header, error) {
	return c.url, http.Header{}, nil
}
func (gatedConnector) ModelPath(model string) string        { return "models/" + model }
func (gatedConnector) SupportsNonBlockingTools(string) bool { return false }

// A connector that takes no non-blocking calls overrides the model: on a model
// that would take them, asynchronous tools are still not declared NON_BLOCKING,
// and synchronous ones are not declared BLOCKING either, since tagging is gated
// on the same support.
func TestAConnectorWithoutNonBlockingToolsDeclaresNoBehavior(t *testing.T) {
	s := NewWithConnector("Gated", gatedConnector{}, Config{
		Model: "gemini-3.8-live",
		Tools: frames.ToolsSchema{Standard: []frames.Tool{{Name: "sync_tool"}, {Name: "async_tool"}}},
	})
	noop := func(ctx context.Context, p llm.FunctionCallParams) error { return p.Result(ctx, "", nil) }
	s.RegisterFunction("sync_tool", noop)
	s.RegisterFunction("async_tool", noop, llm.WithCancelOnInterruption(false))

	tools, _ := setupOf(t, s)["tools"].([]map[string]any)
	if len(tools) == 0 {
		t.Fatal("the setup carries no tools")
	}
	for _, group := range tools {
		decls, _ := group["functionDeclarations"].([]map[string]any)
		for _, d := range decls {
			if b, ok := d["behavior"]; ok {
				t.Errorf("%v declared with behavior %v, want none", d["name"], b)
			}
		}
	}
	if s.AcceptsIntermediateFunctionCallResults() {
		t.Error("AcceptsIntermediateFunctionCallResults() = true, want false")
	}
}

// Nor does an asynchronous tool's result carry a scheduling hint.
func TestAConnectorWithoutNonBlockingToolsSendsNoScheduling(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, NewWithConnector("Gated", gatedConnector{url: wsURL(srv)},
		Config{Model: "gemini-3.8-live"}))

	s.handleFunctionCallResult(t.Context(), resultFrame(true, true))
	settle()

	got := srv.functionResponses()
	if len(got) != 1 {
		t.Fatalf("responses = %v, want one", got)
	}
	inner, _ := got[0]["response"].(map[string]any)
	if _, ok := got[0]["scheduling"]; ok {
		t.Errorf("response = %v, want no scheduling", got[0])
	}
	if _, ok := inner["scheduling"]; ok {
		t.Errorf("response = %v, want no scheduling inside it", inner)
	}
}
