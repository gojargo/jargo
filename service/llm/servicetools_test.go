package llm_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// ownToolsGen is a service configured with tools of its own, as the realtime
// services are.
type ownToolsGen struct {
	fakeGen
	tools frames.ToolsSchema
}

func (g *ownToolsGen) ServiceTools() frames.ToolsSchema { return g.tools }

// sampleHandler is a handler carried on a configured tool.
func sampleHandler(ctx context.Context, p llm.FunctionCallParams) error {
	return p.Result(ctx, "{}", nil)
}

// sampleTool advertises a tool named name carrying h.
func sampleTool(name string, h llm.FunctionCallHandler) frames.Tool {
	return frames.Tool{
		Name: name, Description: "A sample tool.", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: h,
	}
}

func withOwnTools(tools frames.ToolsSchema) *llm.Base {
	return llm.New("OwnToolsLLM", &ownToolsGen{tools: tools})
}

// A configured tool's handler registers when the conversation advertises no
// tools.
func TestServiceSchemaHandlerRegisters(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("sample", sampleHandler)}})
	// An empty conversation falls back to the service tools.
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !svc.HasFunction("sample") {
		t.Fatal("the configured tool's handler was not registered")
	}
}

// The options a configured tool carries are honored when it registers.
func TestServiceAsyncToolOptionHonored(t *testing.T) {
	keep := false
	tool := sampleTool("sample", sampleHandler)
	tool.CancelOnInterruption = &keep
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{tool}})
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !svc.FunctionIsAsync("sample") {
		t.Fatal("a configured tool that is not canceled on interruption should register as async")
	}
}

// A service with no configured tools registers nothing from them.
func TestNoServiceToolsIsSafe(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{})
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if svc.HasFunction("sample") {
		t.Fatal("nothing should be registered")
	}
}

// Configured tools in a provider's own format carry no handler, so they
// register nothing.
func TestCustomServiceToolsRegisterNothing(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{frames.AdapterTypeGemini: {map[string]any{"function_declarations": []any{}}}},
	})
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if svc.HasFunction("function_declarations") || svc.HasFunction("sample") {
		t.Fatal("provider-format tools should register nothing")
	}
}

// When the conversation advertises tools of its own, the configured ones are
// not used.
func TestContextToolsTakePrecedence(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_init", sampleHandler)}})
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{
		Standard: []frames.Tool{sampleTool("from_context", sampleHandler)},
	})
	if !svc.HasFunction("from_context") {
		t.Error("the conversation's tool was not registered")
	}
	if svc.HasFunction("from_init") {
		t.Error("the configured tool should not be used while the conversation advertises tools")
	}
}

// A conversation advertising only tools in a provider's own format still
// advertises tools, so the configured ones are not used. An empty toolset is
// one with neither standard nor custom tools.
func TestContextCustomToolsAlsoTakePrecedence(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_init", sampleHandler)}})
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{frames.AdapterTypeGemini: {map[string]any{"google_search": map[string]any{}}}},
	})
	if svc.HasFunction("from_init") {
		t.Error("the configured tool should not be used while the conversation advertises custom tools")
	}
}

// Falling back to the configured tools is a sync like any other: once the
// conversation advertises tools, the handlers registered from the configured
// ones are dropped, and they come back when it advertises none again.
func TestServiceToolHandlersFollowTheFallback(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_init", sampleHandler)}})
	ctx := context.Background()
	svc.SyncToolHandlers(ctx, frames.ToolsSchema{})
	svc.SyncToolHandlers(ctx, frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_context", sampleHandler)}})
	if svc.HasFunction("from_init") {
		t.Error("the configured tool's handler should be dropped once the conversation advertises tools")
	}
	svc.SyncToolHandlers(ctx, frames.ToolsSchema{})
	if !svc.HasFunction("from_init") || svc.HasFunction("from_context") {
		t.Error("with no tools advertised, the handlers should be the configured tools' again")
	}
}

// startService runs svc in a pipeline until the StartFrame has passed it, and
// returns a function that stops the pipeline.
func startService(t *testing.T, svc *llm.Base) func() {
	t.Helper()
	started := make(chan struct{}, 1)
	probe := newProbe(func(f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			select {
			case started <- struct{}{}:
			default:
			}
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not start")
	}
	return func() {
		task.StopWhenDone()
		<-runDone
	}
}

// A service's own tools have their handlers registered as soon as it starts: a
// realtime service can run one of its configured tools before the first
// conversation reaches it, so the handlers cannot wait for it.
func TestServiceToolsRegisteredOnStart(t *testing.T) {
	svc := withOwnTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("lookup_order", sampleHandler)}})
	if svc.HasFunction("lookup_order") {
		t.Fatal("registered before the service started")
	}
	stop := startService(t, svc)
	defer stop()
	if !svc.HasFunction("lookup_order") {
		t.Fatal("the configured tool's handler was not registered on start")
	}
}

// A service without tools of its own registers nothing on start.
func TestStartWithoutServiceToolsRegistersNothing(t *testing.T) {
	svc := llm.New("FakeLLM", &fakeGen{})
	stop := startService(t, svc)
	defer stop()
	if svc.HasFunction("lookup_order") {
		t.Fatal("nothing should be registered")
	}
}

// Syncing settles the system instruction from what it registered: a toolset
// bringing an asynchronous tool brings the guidance on asynchronous results
// with it, and withdrawing it takes the guidance away. (Not a port of an
// upstream test: it covers the instruction being recomposed at the end of every
// sync, as upstream does.)
func TestSyncComposesTheAsyncToolGuidance(t *testing.T) {
	keep := false
	tool := sampleTool("slow_lookup", sampleHandler)
	tool.CancelOnInterruption = &keep
	svc := llm.New("FakeLLM", &fakeGen{})
	svc.SetSystemInstruction("be brief")
	ctx := context.Background()

	svc.SyncToolHandlers(ctx, frames.ToolsSchema{Standard: []frames.Tool{tool}})
	if !strings.Contains(svc.SystemInstruction(), frames.AsyncToolInstructions) {
		t.Fatalf("instruction lacks the async-tool guidance: %q", svc.SystemInstruction())
	}
	svc.SyncToolHandlers(ctx, frames.ToolsSchema{})
	if got := svc.SystemInstruction(); got != "be brief" {
		t.Fatalf("instruction = %q, want the base prompt alone", got)
	}
}
