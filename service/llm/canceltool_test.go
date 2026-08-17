package llm_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// advertisingGen exposes the adapter the service adds its built-in tools to, so
// a test can read back what the model would actually be offered.
type advertisingGen struct{ adapter adapter.Base }

func (g *advertisingGen) LLMAdapter() llm.BuiltinToolHolder { return &g.adapter }

func (g *advertisingGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *advertisingGen) GenerateWithTools(context.Context, *frames.LLMContext, llm.Sink) error {
	return nil
}

// advertised returns the names of the built-in tools the service offers.
func advertised(g *advertisingGen) []string {
	var names []string
	for _, t := range g.adapter.WithBuiltins(frames.ToolsSchema{}).Standard {
		names = append(names, t.Name)
	}
	slices.Sort(names)
	return names
}

func noop(context.Context, llm.FunctionCallParams) error { return nil }

// newInstructedService builds a service whose base system prompt is set, which
// is what the composed instruction is built on top of.
func newInstructedService(t *testing.T, gen *advertisingGen, prompt string) *llm.Base {
	t.Helper()
	svc := llm.New("FakeLLM", gen)
	svc.SetSystemInstruction(prompt)
	return svc
}

func TestACancellableToolGetsItsOwnCancelTool(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)

	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	want := []string{llm.CancelToolName("write_report")}
	if got := advertised(gen); !slices.Equal(got, want) {
		t.Errorf("advertised %v, want %v", got, want)
	}
	if !svc.HasFunction(llm.CancelToolName("write_report")) {
		t.Error("the cancel tool has no handler, so a call of it would go unanswered")
	}
}

func TestTheCancelToolDoesNotAskForAToolCallID(t *testing.T) {
	// Requiring it would send every cancellation looking for an id first,
	// including the common case of one call running.
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)
	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	var schema struct {
		Properties map[string]any `json:"properties"`
		Required   []string       `json:"required"`
	}
	for _, tool := range gen.adapter.WithBuiltins(frames.ToolsSchema{}).Standard {
		if tool.Name != llm.CancelToolName("write_report") {
			continue
		}
		if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
			t.Fatal(err)
		}
	}
	if len(schema.Required) != 0 {
		t.Errorf("required = %v, want nothing required", schema.Required)
	}
	if _, ok := schema.Properties["tool_call_id"]; !ok {
		t.Errorf("properties = %v, want a tool_call_id among them", schema.Properties)
	}
}

func TestAToolThatDidNotOptInGetsNone(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)

	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))
	svc.RegisterFunction("get_current_weather", noop, llm.WithCancelOnInterruption(false))

	// The weather tool has no cancel tool, so there is nothing to call against it.
	if got := advertised(gen); slices.Contains(got, llm.CancelToolName("get_current_weather")) {
		t.Errorf("advertised %v, want no cancel tool for the one that did not opt in", got)
	}
}

func TestNothingAdvertisedWithoutACancellableTool(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)

	svc.RegisterFunction("get_current_weather", noop, llm.WithCancelOnInterruption(false))

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want nothing", got)
	}
}

func TestWithdrawnWhenTheCancellableToolGoes(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)
	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	svc.UnregisterFunction("write_report")

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want nothing left", got)
	}
	if svc.HasFunction(llm.CancelToolName("write_report")) {
		t.Error("the cancel tool outlived the tool it cancels")
	}
}

func TestCancellableByLLMIsIgnoredOnASynchronousTool(t *testing.T) {
	// There is no moment at which the model could cancel a call it is waiting on.
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)

	svc.RegisterFunction("lookup", noop, llm.WithCancellableByLLM(true))

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want nothing: a synchronous call cannot be canceled", got)
	}
}

func TestTheDeprecatedFlagCoversEveryAsyncTool(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen, llm.WithAsyncToolCancellation())

	svc.RegisterFunction("get_current_weather", noop, llm.WithCancelOnInterruption(false))

	want := []string{llm.CancelToolName("get_current_weather")}
	if got := advertised(gen); !slices.Equal(got, want) {
		t.Errorf("advertised %v, want %v", got, want)
	}
}

func TestTheDeprecatedFlagNeedsAnAsyncTool(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen, llm.WithAsyncToolCancellation())

	svc.RegisterFunction("lookup", noop)

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want nothing", got)
	}
}

func TestTheGuidanceIsComposedOnce(t *testing.T) {
	// Several cancellable tools share one block of guidance; repeating it is
	// nothing but tokens, and reads to a model as emphasis nobody meant.
	svc := newInstructedService(t, &advertisingGen{}, "be brief")
	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))
	svc.RegisterFunction("fetch_archive", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	system := svc.SystemInstruction()

	if got := strings.Count(system, "ASYNC TOOL CANCELLATION"); got != 1 {
		t.Errorf("the guidance appears %d times, want 1:\n%s", got, system)
	}
	if !strings.HasPrefix(system, "be brief") {
		t.Errorf("system = %q, want the application's own prompt first", system)
	}
}

func TestNoGuidanceWithoutACancellableTool(t *testing.T) {
	svc := newInstructedService(t, &advertisingGen{}, "be brief")
	svc.RegisterFunction("lookup", noop)

	if system := svc.SystemInstruction(); strings.Contains(system, "ASYNC TOOL CANCELLATION") {
		t.Errorf("system = %q, want no cancellation guidance", system)
	}
}

func TestTheGuidanceGoesWhenTheToolDoes(t *testing.T) {
	svc := newInstructedService(t, &advertisingGen{}, "be brief")
	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))
	if !strings.Contains(svc.SystemInstruction(), "ASYNC TOOL CANCELLATION") {
		t.Fatal("the guidance was never composed in")
	}

	svc.UnregisterFunction("write_report")

	if system := svc.SystemInstruction(); strings.Contains(system, "ASYNC TOOL CANCELLATION") {
		t.Errorf("system = %q, want the guidance taken back out", system)
	}
}

func TestRegisteringOverACancelToolNameIsRefused(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)
	svc.RegisterFunction("write_report", noop,
		llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	name := llm.CancelToolName("write_report")
	var ran bool
	svc.RegisterFunction(name, func(context.Context, llm.FunctionCallParams) error {
		ran = true
		return nil
	})

	// The service owns the name, so the registration is refused rather than
	// replacing the handler that stops the work.
	if ran {
		t.Error("the replacement handler ran")
	}
	if !svc.HasFunction(name) {
		t.Error("the cancel tool lost its handler")
	}
}

// scriptedGen requests the tool calls it was given, one turn's worth per
// inference, and answers in words once the script runs out.
type scriptedGen struct {
	adapter adapter.Base
	mu      sync.Mutex
	turns   [][]frames.ToolCall
	turn    int
}

func (g *scriptedGen) LLMAdapter() llm.BuiltinToolHolder { return &g.adapter }

func (g *scriptedGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *scriptedGen) GenerateWithTools(_ context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	g.mu.Lock()
	turn := g.turn
	g.turn++
	g.mu.Unlock()
	if turn >= len(g.turns) {
		return sink.Text("done")
	}
	for _, c := range g.turns[turn] {
		if err := sink.Tool(c); err != nil {
			return err
		}
	}
	return nil
}

// cancelResult drives a service through a scripted pair of turns and returns
// what the cancel tool reported. running are the calls the first turn starts,
// and cancelArgs the arguments the second turn calls the cancel tool with.
func cancelResult(
	t *testing.T, running []frames.ToolCall, target, cancelArgs string,
) map[string]any {
	t.Helper()
	cancelName := llm.CancelToolName(target)
	gen := &scriptedGen{turns: [][]frames.ToolCall{
		running,
		{{ID: "cancel_call", Name: cancelName, Args: json.RawMessage(cancelArgs)}},
	}}
	svc := llm.New("FakeLLM", gen)

	started := make(chan struct{}, len(running))
	svc.RegisterFunction(target, func(ctx context.Context, p llm.FunctionCallParams) error {
		started <- struct{}{}
		<-ctx.Done()
		return nil
	}, llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	reported := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok && fr.ToolCallID == "cancel_call" {
			select {
			case reported <- fr.Result:
			default:
			}
		}
	})
	convo := toolConvo(target)
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	for range running {
		select {
		case <-started:
		case <-time.After(3 * time.Second):
			t.Fatal("a call the test needs running never started")
		}
	}
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	var payload map[string]any
	select {
	case raw := <-reported:
		if err := json.Unmarshal([]byte(raw), &payload); err != nil {
			t.Fatalf("decoding %q: %v", raw, err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the cancel tool never reported")
	}

	task.Cancel(context.Background(), "")
	<-runDone
	return payload
}

func TestCancellingFallsBackToTheOnlyRunningCall(t *testing.T) {
	// Omitting the tool_call_id with one call running means that call, which is
	// why the schema does not ask for it.
	got := cancelResult(t, []frames.ToolCall{
		{ID: "call-1", Name: "write_report", Args: json.RawMessage(`{}`)},
	}, "write_report", `{}`)

	if got["cancelled"] != "call-1" { //nolint:misspell // the key the model is told to expect
		t.Errorf("cancelled = %v, want call-1", got["cancelled"]) //nolint:misspell // ditto
	}
}

func TestCancellingByToolCallID(t *testing.T) {
	got := cancelResult(t, []frames.ToolCall{
		{ID: "call-1", Name: "write_report", Args: json.RawMessage(`{}`)},
		{ID: "call-2", Name: "write_report", Args: json.RawMessage(`{}`)},
	}, "write_report", `{"tool_call_id":"call-2"}`)

	if got["cancelled"] != "call-2" { //nolint:misspell // the key the model is told to expect
		t.Errorf("cancelled = %v, want call-2", got["cancelled"]) //nolint:misspell // ditto
	}
}

func TestRefusedWhenNoSuchCallIsRunning(t *testing.T) {
	got := cancelResult(t, nil, "write_report", `{}`)

	if got["cancelled"] != nil { //nolint:misspell // the key the model is told to expect
		t.Errorf("cancelled = %v, want nothing cancelled", got["cancelled"]) //nolint:misspell // ditto
	}
	reason, _ := got["reason"].(string)
	if !strings.Contains(reason, "no write_report call is running") {
		t.Errorf("reason = %q, want it to say nothing is running", reason)
	}
}

func TestAnAmbiguousCallAsksForAToolCallID(t *testing.T) {
	got := cancelResult(t, []frames.ToolCall{
		{ID: "call-1", Name: "write_report", Args: json.RawMessage(`{}`)},
		{ID: "call-3", Name: "write_report", Args: json.RawMessage(`{}`)},
	}, "write_report", `{}`)

	if got["cancelled"] != nil { //nolint:misspell // the key the model is told to expect
		t.Errorf("cancelled = %v, want nothing cancelled", got["cancelled"]) //nolint:misspell // ditto
	}
	// The choices are spelled out in the reason itself, since a model acts on
	// what a refusal says far more readily than on a field beside it.
	reason, _ := got["reason"].(string)
	for _, id := range []string{"call-1", "call-3"} {
		if !strings.Contains(reason, id) {
			t.Errorf("reason = %q, want it to name %s", reason, id)
		}
	}
	running, ok := got["running"].([]any)
	if !ok {
		t.Fatalf("running = %v, want the ids to choose between", got["running"])
	}
	ids := map[string]bool{}
	for _, entry := range running {
		call, ok := entry.(map[string]any)
		if !ok {
			t.Fatalf("running entry = %v, want a call", entry)
		}
		id, _ := call["tool_call_id"].(string)
		ids[id] = true
	}
	if !ids["call-1"] || !ids["call-3"] {
		t.Errorf("running = %v, want both ids", got["running"])
	}
}

func TestRefusedWhenNoRunningCallHasThatID(t *testing.T) {
	got := cancelResult(t, []frames.ToolCall{
		{ID: "call-1", Name: "write_report", Args: json.RawMessage(`{}`)},
	}, "write_report", `{"tool_call_id":"call-9"}`)

	if got["cancelled"] != nil { //nolint:misspell // the key the model is told to expect
		t.Errorf("cancelled = %v, want nothing cancelled", got["cancelled"]) //nolint:misspell // ditto
	}
	reason, _ := got["reason"].(string)
	if !strings.Contains(reason, "tool_call_id") {
		t.Errorf("reason = %q, want it to say the id matched nothing", reason)
	}
}
