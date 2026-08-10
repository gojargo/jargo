package flows

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// fakeEnq records the frames a FlowManager queues and, like the pipeline,
// reports each one as having reached the end so the actions waiting on them do
// not hang.
type fakeEnq struct {
	mu       sync.Mutex
	queued   []frames.Frame
	watchers []func(frames.Frame)
}

func (e *fakeEnq) QueueFrame(f frames.Frame) {
	e.mu.Lock()
	e.queued = append(e.queued, f)
	watchers := make([]func(frames.Frame), len(e.watchers))
	copy(watchers, e.watchers)
	e.mu.Unlock()
	for _, w := range watchers {
		w(f)
	}
}

func (e *fakeEnq) QueueFrames(fs []frames.Frame) {
	for _, f := range fs {
		e.QueueFrame(f)
	}
}

func (e *fakeEnq) OnReachedDownstream(fn func(frames.Frame)) {
	e.mu.Lock()
	e.watchers = append(e.watchers, fn)
	e.mu.Unlock()
}

func (e *fakeEnq) frames() []frames.Frame {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]frames.Frame(nil), e.queued...)
}

func (e *fakeEnq) reset() {
	e.mu.Lock()
	e.queued = nil
	e.mu.Unlock()
}

// advertisedTools returns the tools of the most recent LLMSetToolsFrame queued.
// The manager advertises a node's tools this way, each carrying the handler the
// LLM service registers when it sees them.
func advertisedTools(fs []frames.Frame) []frames.Tool {
	var tools []frames.Tool
	for _, f := range fs {
		if set, ok := f.(*frames.LLMSetToolsFrame); ok {
			tools = set.Tools
		}
	}
	return tools
}

// advertisedHandler returns the handler advertised for name.
func advertisedHandler(t *testing.T, fs []frames.Frame, name string) llm.FunctionCallHandler {
	t.Helper()
	for _, tool := range advertisedTools(fs) {
		if tool.Name != name {
			continue
		}
		h, ok := tool.Handler.(llm.FunctionCallHandler)
		if !ok {
			t.Fatalf("tool %q carries a %T, want llm.FunctionCallHandler", name, tool.Handler)
		}
		return h
	}
	t.Fatalf("no tool advertised for %q", name)
	return nil
}

// call runs an advertised handler the way the service would and reports what it
// delivered through the result callback.
func call(
	t *testing.T, h llm.FunctionCallHandler,
) (result string, props *frames.FunctionCallResultProperties, err error) {
	t.Helper()
	reported := false
	err = h(context.Background(), llm.FunctionCallParams{
		FunctionName: "test",
		ToolCallID:   "call-1",
		Arguments:    json.RawMessage(`{}`),
		Result: func(_ context.Context, r string, p *frames.FunctionCallResultProperties) error {
			reported, result, props = true, r, p
			return nil
		},
	})
	if err == nil && !reported {
		t.Fatal("handler returned without reporting a result")
	}
	return result, props, err
}

// fakeInferencer stands in for the LLM the summary strategy runs on.
type fakeInferencer struct {
	summary string
	err     error
	prompt  string
	convo   *frames.LLMContext
}

func (f *fakeInferencer) RunInference(
	_ context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	f.prompt, f.convo = opts.SystemInstruction, convo
	return f.summary, f.err
}

func newManager(t *testing.T, opts ...func(*Config)) (*FlowManager, *fakeEnq) {
	t.Helper()
	enq := &fakeEnq{}
	cfg := Config{
		Enqueuer:    enq,
		Watcher:     enq,
		Aggregators: aggregators.New(frames.NewLLMContext("")),
	}
	for _, o := range opts {
		o(&cfg)
	}
	fm, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return fm, enq
}

// node builds a minimal valid node: one whose TaskMessages were given.
func node(name string, fns ...NodeFunction) *NodeConfig {
	return &NodeConfig{Name: name, TaskMessages: []frames.Message{}, Functions: fns}
}

func stay(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
	return "ok", nil, nil
}

func TestNewRequiresAnEnqueuer(t *testing.T) {
	if _, err := New(Config{}); !errors.Is(err, ErrFlow) {
		t.Fatalf("New(empty) = %v, want an ErrFlow", err)
	}
}

func TestSetNodeBeforeInitializeIsRefused(t *testing.T) {
	fm, _ := newManager(t)
	err := fm.SetNode(context.Background(), node("second"))
	if !errors.Is(err, ErrFlowTransition) {
		t.Errorf("SetNode before Initialize = %v, want ErrFlowTransition", err)
	}
	// Every flow error is also an ErrFlow, so a caller may match either.
	if !errors.Is(err, ErrFlow) {
		t.Errorf("SetNode before Initialize = %v, want it to match ErrFlow too", err)
	}
}

func TestInitializeWithoutANodeEntersNothing(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), nil); err != nil {
		t.Fatalf("Initialize(nil): %v", err)
	}
	if got := fm.CurrentNode(); got != "" {
		t.Errorf("CurrentNode = %q, want empty", got)
	}
	if len(enq.frames()) != 0 {
		t.Errorf("queued %d frames, want 0", len(enq.frames()))
	}
	// Initialized, so a node may now be set.
	if err := fm.SetNode(context.Background(), node("first")); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first", fm.CurrentNode())
	}
}

func TestInitializeTwiceIsIgnored(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	enq.reset()
	if err := fm.Initialize(context.Background(), node("second")); err != nil {
		t.Fatalf("second Initialize: %v", err)
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first: the second Initialize must not enter a node",
			fm.CurrentNode())
	}
	if len(enq.frames()) != 0 {
		t.Errorf("queued %d frames on the second Initialize, want 0", len(enq.frames()))
	}
}

func TestNodeRequiresTaskMessages(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), nil); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	// A nil TaskMessages is a node that never said what it is for.
	err := fm.SetNode(context.Background(), &NodeConfig{Name: "bad"})
	if !errors.Is(err, ErrFlow) {
		t.Errorf("SetNode(no TaskMessages) = %v, want ErrFlow", err)
	}
	if !strings.Contains(err.Error(), "TaskMessages") {
		t.Errorf("error = %q, want it to name TaskMessages", err)
	}
	// An empty non-nil TaskMessages is a node that deliberately adds nothing.
	if err := fm.SetNode(context.Background(), node("ok")); err != nil {
		t.Errorf("SetNode(empty TaskMessages) = %v, want it accepted", err)
	}
}

func TestSetNodeQueuesPersonaObjectiveAndTools(t *testing.T) {
	fm, enq := newManager(t)
	n := &NodeConfig{
		Name:         "start",
		RoleMessage:  "be nice",
		TaskMessages: []frames.Message{{Role: frames.RoleDeveloper, Text: "greet"}},
		Functions:    []NodeFunction{{Name: "go_next", Handler: stay}},
	}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	fs := enq.frames()

	// The persona is the LLM service's system instruction, not something said.
	var settingsIdx, appendIdx, toolsIdx, runIdx = -1, -1, -1, -1
	for i, f := range fs {
		switch fr := f.(type) {
		case *frames.LLMUpdateSettingsFrame:
			settingsIdx = i
			delta, ok := fr.Delta.(*settings.LLM)
			if !ok {
				t.Fatalf("settings delta = %T, want *settings.LLM", fr.Delta)
			}
			if got, _ := delta.SystemInstruction.Value(); got != "be nice" {
				t.Errorf("system instruction = %q, want %q", got, "be nice")
			}
		case *frames.LLMMessagesAppendFrame:
			appendIdx = i
			if len(fr.Messages) != 1 || fr.Messages[0].Text != "greet" {
				t.Errorf("appended = %+v, want one 'greet' message", fr.Messages)
			}
			// The role is kept: a developer message is not something the user said.
			if fr.Messages[0].Role != frames.RoleDeveloper {
				t.Errorf("role = %q, want developer", fr.Messages[0].Role)
			}
		case *frames.LLMSetToolsFrame:
			toolsIdx = i
		case *frames.LLMRunFrame:
			runIdx = i
		}
	}

	if settingsIdx < 0 || appendIdx < 0 || toolsIdx < 0 || runIdx < 0 {
		t.Fatalf("missing frames: settings=%d append=%d tools=%d run=%d",
			settingsIdx, appendIdx, toolsIdx, runIdx)
	}
	// The persona is applied before the objective that assumes it, and the node
	// is asked to speak only once its context and tools are in place.
	if settingsIdx > appendIdx {
		t.Error("the settings frame must come before the messages frame")
	}
	if toolsIdx > runIdx {
		t.Error("the tools frame must come before the run frame")
	}

	tools := advertisedTools(fs)
	if len(tools) != 1 || tools[0].Name != "go_next" {
		t.Fatalf("tools = %+v, want [go_next]", tools)
	}
	if tools[0].Handler == nil {
		t.Error("the advertised tool carries no handler")
	}
	if fm.CurrentNode() != "start" {
		t.Errorf("CurrentNode = %q, want start", fm.CurrentNode())
	}
}

func TestNodeWithoutRoleMessageLeavesThePersonaAlone(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), &NodeConfig{
		Name: "first", RoleMessage: "be nice", TaskMessages: []frames.Message{},
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	enq.reset()

	if err := fm.SetNode(context.Background(), node("second")); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.LLMUpdateSettingsFrame); ok {
			t.Error("a node with no RoleMessage must not touch the system instruction")
		}
	}
}

func TestWaitNodeDoesNotAskForAResponse(t *testing.T) {
	fm, enq := newManager(t)
	no := false
	n := &NodeConfig{Name: "inbound", TaskMessages: []frames.Message{}, RespondImmediately: &no}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.LLMRunFrame); ok {
			t.Error("a node that waits for the user must not ask for a response")
		}
	}
}

func TestGlobalFunctionsComeBeforeTheNodesOwn(t *testing.T) {
	fm, enq := newManager(t, func(c *Config) {
		c.GlobalFunctions = []NodeFunction{{Name: "help", Handler: stay}}
	})
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "order", Handler: stay})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	tools := advertisedTools(enq.frames())
	if len(tools) != 2 || tools[0].Name != "help" || tools[1].Name != "order" {
		t.Errorf("tools = %+v, want [help order]", tools)
	}
}

func TestFlowFunctionsSurviveAnInterruptionByDefault(t *testing.T) {
	fm, enq := newManager(t)
	timeout := 12.5
	yes := true
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "defaults", Handler: stay},
		NodeFunction{Name: "overrides", Handler: stay, CancelOnInterruption: &yes, TimeoutSecs: &timeout},
	)); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	tools := advertisedTools(enq.frames())
	byName := map[string]frames.Tool{}
	for _, tool := range tools {
		byName[tool.Name] = tool
	}

	// A flow function is doing work the conversation still needs whether or not
	// the user spoke over it, so the flow default is not to cancel. That is not
	// the LLM service's own default, so it must be stated.
	d := byName["defaults"]
	if d.CancelOnInterruption == nil || *d.CancelOnInterruption {
		t.Errorf("defaults CancelOnInterruption = %v, want an explicit false", d.CancelOnInterruption)
	}
	if d.TimeoutSecs != nil {
		t.Errorf("defaults TimeoutSecs = %v, want nil", d.TimeoutSecs)
	}

	o := byName["overrides"]
	if o.CancelOnInterruption == nil || !*o.CancelOnInterruption {
		t.Errorf("overrides CancelOnInterruption = %v, want true", o.CancelOnInterruption)
	}
	if o.TimeoutSecs == nil || *o.TimeoutSecs != 12.5 {
		t.Errorf("overrides TimeoutSecs = %v, want 12.5", o.TimeoutSecs)
	}
}

func TestAdvertisedParametersDescribeTheArguments(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), node("first", NodeFunction{
		Name:       "record",
		Handler:    stay,
		Properties: map[string]any{"drink": map[string]any{"type": "string"}},
		Required:   []string{"drink"},
	})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	var schema struct {
		Type       string                    `json:"type"`
		Properties map[string]map[string]any `json:"properties"`
		Required   []string                  `json:"required"`
	}
	if err := json.Unmarshal(advertisedTools(enq.frames())[0].Parameters, &schema); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	if schema.Type != "object" {
		t.Errorf("type = %q, want object", schema.Type)
	}
	if schema.Properties["drink"]["type"] != "string" {
		t.Errorf("properties = %+v, want drink to be a string", schema.Properties)
	}
	if len(schema.Required) != 1 || schema.Required[0] != "drink" {
		t.Errorf("required = %v, want [drink]", schema.Required)
	}
}

func TestFunctionWithoutNameOrHandlerIsRefused(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), nil); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	err := fm.SetNode(context.Background(), node("a", NodeFunction{Handler: stay}))
	if !errors.Is(err, ErrInvalidFunction) {
		t.Errorf("function with no name = %v, want ErrInvalidFunction", err)
	}
	err = fm.SetNode(context.Background(), node("b", NodeFunction{Name: "x"}))
	if !errors.Is(err, ErrInvalidFunction) {
		t.Errorf("function with no handler = %v, want ErrInvalidFunction", err)
	}
}

func TestNodeFunctionStaysAndAnswers(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "note", Handler: stay})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	res, props, err := call(t, advertisedHandler(t, enq.frames(), "note"))
	if err != nil {
		t.Fatalf("note handler: %v", err)
	}
	if res != "ok" {
		t.Errorf("result = %q, want ok", res)
	}
	// A node function answers from its result.
	if props == nil || props.RunLLM == nil || !*props.RunLLM {
		t.Errorf("properties = %+v, want RunLLM true", props)
	}
	if props.OnContextUpdated != nil {
		t.Error("a node function schedules no transition")
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first", fm.CurrentNode())
	}
}

// A tool that has already spoken for itself, one that plays audio of its own or
// ends the call, must not have the model answer from its result: that answer is
// synthesized over whatever the tool started.
func TestNoResponseStaysSilentAndDoesNotTransition(t *testing.T) {
	fm, enq := newManager(t)
	play := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "playing", NoResponse, nil
	}
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "play", Handler: play})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	h := advertisedHandler(t, enq.frames(), "play")
	enq.reset()

	res, props, err := call(t, h)
	if err != nil {
		t.Fatalf("play handler: %v", err)
	}
	// The model is still owed its answer: it has to know what was put on when the
	// user speaks next.
	if res != "playing" {
		t.Errorf("result = %q, want playing", res)
	}
	if props == nil || props.RunLLM == nil || *props.RunLLM {
		t.Errorf("properties = %+v, want RunLLM false", props)
	}
	if props.OnContextUpdated != nil {
		t.Error("NoResponse schedules no transition")
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first", fm.CurrentNode())
	}
	if len(enq.frames()) != 0 {
		t.Errorf("queued %d frames, want 0: NoResponse writes nothing", len(enq.frames()))
	}
}

func TestEdgeFunctionDefersTheTransition(t *testing.T) {
	fm, enq := newManager(t)
	next := node("second", NodeFunction{Name: "stay", Handler: stay})
	next.RoleMessage = "second persona"
	advance := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "", next, nil
	}
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "advance", Handler: advance})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	h := advertisedHandler(t, enq.frames(), "advance")
	enq.reset()

	res, props, err := call(t, h)
	if err != nil {
		t.Fatalf("advance handler: %v", err)
	}
	// A function that transitioned without a result of its own is still owed an
	// answer, so an acknowledgement stands in.
	if res != acknowledged {
		t.Errorf("result = %q, want %q", res, acknowledged)
	}
	// The transition waits for the result to reach the conversation, so nothing
	// is generated here and nothing has moved yet.
	if props == nil || props.RunLLM == nil || *props.RunLLM {
		t.Errorf("properties = %+v, want RunLLM false", props)
	}
	if props.OnContextUpdated == nil {
		t.Fatal("an edge function must schedule its transition")
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first: the move waits for the callback",
			fm.CurrentNode())
	}
	if len(enq.frames()) != 0 {
		t.Errorf("queued %d frames before the callback, want 0", len(enq.frames()))
	}

	// The callback runs once the result is in the conversation.
	if err := props.OnContextUpdated(context.Background()); err != nil {
		t.Fatalf("OnContextUpdated: %v", err)
	}
	if fm.CurrentNode() != "second" {
		t.Errorf("CurrentNode = %q, want second", fm.CurrentNode())
	}
	tools := advertisedTools(enq.frames())
	if len(tools) != 1 || tools[0].Name != "stay" {
		t.Errorf("tools = %+v, want [stay]", tools)
	}
}

func TestTransitionRunsOnlyOnce(t *testing.T) {
	fm, enq := newManager(t)
	next := node("second")
	advance := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "done", next, nil
	}
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "advance", Handler: advance})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	_, props, err := call(t, advertisedHandler(t, enq.frames(), "advance"))
	if err != nil {
		t.Fatalf("advance handler: %v", err)
	}

	ctx := context.Background()
	if err := props.OnContextUpdated(ctx); err != nil {
		t.Fatalf("first OnContextUpdated: %v", err)
	}
	enq.reset()
	// A second callback for the same transition finds nothing pending.
	if err := props.OnContextUpdated(ctx); err != nil {
		t.Fatalf("second OnContextUpdated: %v", err)
	}
	if len(enq.frames()) != 0 {
		t.Errorf("queued %d frames on the second callback, want 0", len(enq.frames()))
	}
}

func TestFailingHandlerAnswersTheModel(t *testing.T) {
	fm, enq := newManager(t)
	boom := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "", nil, errors.New("kaboom")
	}
	if err := fm.Initialize(context.Background(), node("first",
		NodeFunction{Name: "boom", Handler: boom})); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	res, _, err := call(t, advertisedHandler(t, enq.frames(), "boom"))
	if err != nil {
		t.Fatalf("handler returned an error instead of reporting it: %v", err)
	}
	// The model called the function and is owed an answer; reporting nothing
	// would leave the call showing as in progress forever.
	if !strings.Contains(res, "error") || !strings.Contains(res, "kaboom") {
		t.Errorf("result = %q, want it to carry the failure", res)
	}
}

func TestContextStrategyReset(t *testing.T) {
	reset := ContextStrategyConfig{Strategy: ContextStrategyReset}
	fm, enq := newManager(t, func(c *Config) { c.ContextStrategy = &reset })
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	var sawUpdate bool
	for _, f := range enq.frames() {
		switch f.(type) {
		case *frames.LLMMessagesUpdateFrame:
			sawUpdate = true
		case *frames.LLMMessagesAppendFrame:
			t.Error("the reset strategy must replace the conversation, not append to it")
		}
	}
	if !sawUpdate {
		t.Error("the reset strategy must queue a messages-update frame")
	}
}

func TestContextStrategyDefaultsToAppendingOnEveryNode(t *testing.T) {
	fm, enq := newManager(t)
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	// Even the opening node appends, so anything a pre-action already put in the
	// conversation survives.
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.LLMMessagesUpdateFrame); ok {
			t.Error("the default strategy must append, not replace")
		}
	}
	enq.reset()
	if err := fm.SetNode(context.Background(), node("second")); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	var sawAppend bool
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			sawAppend = true
		}
	}
	if !sawAppend {
		t.Error("a later node must append too")
	}
}

func TestNodeStrategyOverridesTheManagers(t *testing.T) {
	appendAll := ContextStrategyConfig{Strategy: ContextStrategyAppend}
	fm, enq := newManager(t, func(c *Config) { c.ContextStrategy = &appendAll })
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	enq.reset()

	reset := ContextStrategyConfig{Strategy: ContextStrategyReset}
	n := node("second")
	n.ContextStrategy = &reset
	if err := fm.SetNode(context.Background(), n); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	var sawUpdate bool
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.LLMMessagesUpdateFrame); ok {
			sawUpdate = true
		}
	}
	if !sawUpdate {
		t.Error("the node's strategy must override the manager's")
	}
}

func TestResetWithSummaryNeedsAPrompt(t *testing.T) {
	bad := ContextStrategyConfig{Strategy: ContextStrategyResetWithSummary}
	if err := bad.Validate(); err == nil {
		t.Error("reset-with-summary without a prompt must be refused")
	}
	good := ContextStrategyConfig{
		Strategy: ContextStrategyResetWithSummary, SummaryPrompt: "Summarize",
	}
	if err := good.Validate(); err != nil {
		t.Errorf("reset-with-summary with a prompt = %v, want it accepted", err)
	}
}

func TestResetWithSummaryPutsTheSummaryFirst(t *testing.T) {
	inf := &fakeInferencer{summary: "they ordered a latte"}
	strategy := ContextStrategyConfig{
		Strategy: ContextStrategyResetWithSummary, SummaryPrompt: "Summarize the conversation",
	}
	fm, enq := newManager(t, func(c *Config) {
		c.ContextStrategy = &strategy
		c.LLM = inf
	})
	n := node("first")
	n.TaskMessages = []frames.Message{{Role: frames.RoleDeveloper, Text: "new task"}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	var update *frames.LLMMessagesUpdateFrame
	for _, f := range enq.frames() {
		if u, ok := f.(*frames.LLMMessagesUpdateFrame); ok {
			update = u
		}
	}
	if update == nil {
		t.Fatal("reset-with-summary must replace the conversation")
	}
	if len(update.Messages) != 2 {
		t.Fatalf("messages = %+v, want the summary then the task", update.Messages)
	}
	if !strings.Contains(update.Messages[0].Text, "they ordered a latte") {
		t.Errorf("first message = %q, want it to carry the summary", update.Messages[0].Text)
	}
	if update.Messages[0].Role != frames.RoleDeveloper {
		t.Errorf("summary role = %q, want developer", update.Messages[0].Role)
	}
	if update.Messages[1].Text != "new task" {
		t.Errorf("second message = %q, want the node's task", update.Messages[1].Text)
	}
	if inf.prompt != "Summarize the conversation" {
		t.Errorf("summary prompt = %q, want the configured one", inf.prompt)
	}
}

func TestSummaryFailureFallsBackToAppending(t *testing.T) {
	inf := &fakeInferencer{err: errors.New("no")}
	strategy := ContextStrategyConfig{
		Strategy: ContextStrategyResetWithSummary, SummaryPrompt: "Summarize",
	}
	fm, enq := newManager(t, func(c *Config) {
		c.ContextStrategy = &strategy
		c.LLM = inf
	})
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	// Without a summary a reset would throw the conversation away, so it appends
	// instead and nothing is lost.
	var sawAppend bool
	for _, f := range enq.frames() {
		switch f.(type) {
		case *frames.LLMMessagesAppendFrame:
			sawAppend = true
		case *frames.LLMMessagesUpdateFrame:
			t.Error("a failed summary must not reset the conversation")
		}
	}
	if !sawAppend {
		t.Error("a failed summary must fall back to appending")
	}
}

func TestPreAndPostActionsSpeak(t *testing.T) {
	fm, enq := newManager(t)
	n := node("first")
	n.PreActions = []ActionConfig{{Type: "tts_say", Text: "Pre action"}}
	n.PostActions = []ActionConfig{{Type: "tts_say", Text: "Post action"}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	var spoken []string
	for _, f := range enq.frames() {
		if s, ok := f.(*frames.TTSSpeakFrame); ok {
			spoken = append(spoken, s.Text)
		}
	}
	if len(spoken) != 2 || spoken[0] != "Pre action" || spoken[1] != "Post action" {
		t.Errorf("spoken = %v, want [Pre action, Post action]", spoken)
	}
}

func TestTTSActionCarriesAppendTextToContext(t *testing.T) {
	fm, enq := newManager(t)
	no := false
	n := node("first")
	n.PreActions = []ActionConfig{
		{Type: "tts_say", Text: "kept"},
		{Type: "tts_say", Text: "dropped", AppendTextToContext: &no},
	}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	var spoken []*frames.TTSSpeakFrame
	for _, f := range enq.frames() {
		if s, ok := f.(*frames.TTSSpeakFrame); ok {
			spoken = append(spoken, s)
		}
	}
	if len(spoken) != 2 {
		t.Fatalf("spoke %d times, want 2", len(spoken))
	}
	if !spoken[0].AppendToContext {
		t.Error("the default must write the spoken text to the conversation")
	}
	if spoken[1].AppendToContext {
		t.Error("AppendTextToContext false must keep the text out of the conversation")
	}
}

func TestEndConversationSpeaksThenEnds(t *testing.T) {
	fm, enq := newManager(t)
	n := node("first")
	n.PreActions = []ActionConfig{{Type: "end_conversation", Text: "Goodbye!"}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	var speakIdx, endIdx = -1, -1
	for i, f := range enq.frames() {
		switch f.(type) {
		case *frames.TTSSpeakFrame:
			speakIdx = i
		case *frames.EndFrame:
			endIdx = i
		}
	}
	if speakIdx < 0 || endIdx < 0 {
		t.Fatalf("speak=%d end=%d, want both", speakIdx, endIdx)
	}
	if speakIdx > endIdx {
		t.Error("the goodbye must be spoken before the conversation ends")
	}
}

func TestActionsStopAtEndConversation(t *testing.T) {
	fm, enq := newManager(t)
	n := node("first")
	n.PreActions = []ActionConfig{
		{Type: "end_conversation", Text: "Goodbye!"},
		{Type: "tts_say", Text: "never said"},
	}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	for _, f := range enq.frames() {
		if s, ok := f.(*frames.TTSSpeakFrame); ok && s.Text == "never said" {
			t.Error("nothing after end_conversation may run: the pipeline is stopping")
		}
	}
}

func TestFunctionActionsRunInOrder(t *testing.T) {
	fm, _ := newManager(t)
	var mu sync.Mutex
	var order []string
	record := func(name string) ActionHandler {
		return func(context.Context, ActionConfig, *FlowManager) error {
			mu.Lock()
			order = append(order, name)
			mu.Unlock()
			return nil
		}
	}
	n := node("first")
	n.PreActions = []ActionConfig{
		{Type: "function", Handler: record("first")},
		{Type: "function", Handler: record("second")},
	}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Errorf("order = %v, want [first second]", order)
	}
}

func TestUnknownActionIsRefused(t *testing.T) {
	fm, _ := newManager(t)
	n := node("first")
	n.PreActions = []ActionConfig{{Type: "nope"}}
	err := fm.Initialize(context.Background(), n)
	if !errors.Is(err, ErrAction) {
		t.Errorf("unknown action = %v, want ErrAction", err)
	}
	// Initialization failed, so the manager reports it as such.
	if !errors.Is(err, ErrFlowInitialization) {
		t.Errorf("error = %v, want it to report a failed initialization", err)
	}
}

func TestActionMissingItsTypeIsRefused(t *testing.T) {
	fm, _ := newManager(t)
	n := node("first")
	n.PreActions = []ActionConfig{{Text: "no type"}}
	if err := fm.Initialize(context.Background(), n); !errors.Is(err, ErrAction) {
		t.Errorf("action with no type = %v, want ErrAction", err)
	}
}

func TestCustomActionFromConfigIsRegistered(t *testing.T) {
	fm, _ := newManager(t)
	ran := false
	n := node("first")
	n.PreActions = []ActionConfig{{
		Type: "notify",
		Handler: func(context.Context, ActionConfig, *FlowManager) error {
			ran = true
			return nil
		},
	}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if !ran {
		t.Error("an action carrying a handler must register and run it")
	}
}

func TestRegisteredActionIsReusable(t *testing.T) {
	fm, _ := newManager(t)
	count := 0
	if err := fm.RegisterAction("notify", func(context.Context, ActionConfig, *FlowManager) error {
		count++
		return nil
	}); err != nil {
		t.Fatalf("RegisterAction: %v", err)
	}
	n := node("first")
	n.PreActions = []ActionConfig{{Type: "notify"}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if count != 1 {
		t.Errorf("ran %d times, want 1", count)
	}
}

func TestPostActionsAreDeferredOnAWaitNode(t *testing.T) {
	fm, enq := newManager(t)
	no := false
	n := node("first")
	n.RespondImmediately = &no
	n.PostActions = []ActionConfig{{Type: "tts_say", Text: "after the turn"}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	for _, f := range enq.frames() {
		if _, ok := f.(*frames.TTSSpeakFrame); ok {
			t.Fatal("a wait node's post-actions must wait for the assistant's turn")
		}
	}

	// The assistant's turn ends, which is when they run.
	enq.QueueFrame(frames.NewBotStoppedSpeakingFrame())

	var spoken []string
	for _, f := range enq.frames() {
		if s, ok := f.(*frames.TTSSpeakFrame); ok {
			spoken = append(spoken, s.Text)
		}
	}
	if len(spoken) != 1 || spoken[0] != "after the turn" {
		t.Errorf("spoken = %v, want [after the turn]", spoken)
	}
}

func TestLeavingANodeDropsItsDeferredPostActions(t *testing.T) {
	fm, enq := newManager(t)
	no := false
	first := node("first")
	first.RespondImmediately = &no
	first.PostActions = []ActionConfig{{Type: "tts_say", Text: "belongs to first"}}
	if err := fm.Initialize(context.Background(), first); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if err := fm.SetNode(context.Background(), node("second")); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	enq.reset()

	enq.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	for _, f := range enq.frames() {
		if s, ok := f.(*frames.TTSSpeakFrame); ok && s.Text == "belongs to first" {
			t.Error("the post-actions of the node that was left must not run")
		}
	}
}

func TestStateSurvivesTransitions(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	fm.State().Set("drink", "latte")
	if err := fm.SetNode(context.Background(), node("second")); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	got, ok := fm.State().Get("drink")
	if !ok || got != "latte" {
		t.Errorf("state = %v (%v), want latte", got, ok)
	}
	fm.State().Delete("drink")
	if _, ok := fm.State().Get("drink"); ok {
		t.Error("Delete must remove the key")
	}
}

func TestCurrentContextReadsTheConversation(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), node("first")); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if _, err := fm.CurrentContext(); err != nil {
		t.Errorf("CurrentContext: %v", err)
	}

	// Without aggregators there is no conversation to read.
	bare, err := New(Config{Enqueuer: &fakeEnq{}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := bare.CurrentContext(); !errors.Is(err, ErrFlow) {
		t.Errorf("CurrentContext without aggregators = %v, want ErrFlow", err)
	}
}

func TestUnnamedNodesAreToldApart(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), &NodeConfig{
		TaskMessages: []frames.Message{},
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	first := fm.CurrentNode()
	if first == "" {
		t.Fatal("an unnamed node must still be given a name")
	}
	if err := fm.SetNode(context.Background(), &NodeConfig{
		TaskMessages: []frames.Message{},
	}); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
	if fm.CurrentNode() == first {
		t.Error("two unnamed nodes must be told apart")
	}
}

func TestFailureNamesTheNode(t *testing.T) {
	fm, _ := newManager(t)
	if err := fm.Initialize(context.Background(), nil); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	n := node("checkout")
	n.PreActions = []ActionConfig{{Type: "nope"}}
	err := fm.SetNode(context.Background(), n)
	if !errors.Is(err, ErrAction) {
		t.Errorf("err = %v, want it to match ErrAction", err)
	}
	if !strings.Contains(err.Error(), "checkout") {
		t.Errorf("err = %q, want it to name the node that failed", err)
	}
}
