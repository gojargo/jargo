package flows

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// Tests for function and node metadata, the EventNodeEntered event, the RTVI
// flow-node-entered frame and the {{ key }} placeholders in a node's prompts.

// managerWithEntered is a manager, and the list its EventNodeEntered handler
// appends to.
func managerWithEntered(t *testing.T, opts ...func(*Config)) (*FlowManager, *fakeEnq, *[]NodeEntered) {
	t.Helper()
	fm, enq := newManager(t, opts...)
	entered := &[]NodeEntered{}
	event.On(fm.Events(), EventNodeEntered, func(_ context.Context, e NodeEntered) {
		*entered = append(*entered, e)
	})
	return fm, enq, entered
}

// taskNode is a node whose task message is its name.
func taskNode(name string, fns ...NodeFunction) *NodeConfig {
	return &NodeConfig{
		Name:         name,
		TaskMessages: []frames.Message{{Role: frames.RoleDeveloper, Text: name}},
		Functions:    fns,
	}
}

// withMetadata sets a node's metadata.
func withMetadata(n *NodeConfig, metadata map[string]any) *NodeConfig {
	n.Metadata = metadata
	return n
}

// edge is a function that moves the flow to next.
func edge(name string, next *NodeConfig, metadata map[string]any) NodeFunction {
	return NodeFunction{
		Name:        name,
		Description: name,
		Metadata:    metadata,
		Handler: func(context.Context, json.RawMessage, *FlowManager) (string, *NodeConfig, error) {
			return "", next, nil
		},
	}
}

// invoke calls an advertised tool the way the LLM service would, then lets the
// context-updated callback fire so a pending transition executes.
func invoke(t *testing.T, enq *fakeEnq, name string) {
	t.Helper()
	_, props, err := call(t, advertisedHandler(t, enq.frames(), name))
	if err != nil {
		t.Fatalf("calling %s: %v", name, err)
	}
	if props != nil && props.OnContextUpdated != nil {
		if err := props.OnContextUpdated(context.Background()); err != nil {
			t.Fatalf("OnContextUpdated after %s: %v", name, err)
		}
	}
}

// queuedFrameTypes is the type of every frame queued, in order.
func queuedFrameTypes(enq *fakeEnq) []string {
	var out []string
	for _, f := range enq.frames() {
		out = append(out, strings.TrimPrefix(fmt.Sprintf("%T", f), "*"))
	}
	return out
}

const (
	typeLLMRun          = "frames.LLMRunFrame"
	typeLLMSetTools     = "frames.LLMSetToolsFrame"
	typeFlowNodeEntered = "rtvi.FlowNodeEnteredFrame"
)

// rtviMessages is the data of every flow-node-entered frame queued, as sent.
func rtviMessages(t *testing.T, enq *fakeEnq) []string {
	t.Helper()
	var out []string
	for _, f := range enq.frames() {
		fr, ok := f.(*rtvi.FlowNodeEnteredFrame)
		if !ok {
			continue
		}
		raw, err := json.Marshal(fr.Data)
		if err != nil {
			t.Fatalf("encoding the node message: %v", err)
		}
		out = append(out, string(raw))
	}
	return out
}

func mustInitialize(t *testing.T, fm *FlowManager, n *NodeConfig) {
	t.Helper()
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
}

func mustSetNode(t *testing.T, fm *FlowManager, n *NodeConfig) {
	t.Helper()
	if err := fm.SetNode(context.Background(), n); err != nil {
		t.Fatalf("SetNode: %v", err)
	}
}

// Metadata stays out of the schema the LLM is given.
func TestFunctionMetadataStaysOutOfTheLLMSchema(t *testing.T) {
	fn := NodeFunction{
		Name:        "n",
		Description: "d",
		Properties:  map[string]any{"size": map[string]any{"type": "string"}},
		Required:    []string{"size"},
		Handler:     stay,
		Metadata:    map[string]any{"x": 1},
	}
	params, err := fn.parameters()
	if err != nil {
		t.Fatalf("parameters: %v", err)
	}
	want := `{"properties":{"size":{"type":"string"}},"required":["size"],"type":"object"}`
	if string(params) != want {
		t.Errorf("parameters = %s, want %s", params, want)
	}
	if (NodeFunction{Name: "n", Handler: stay}).Metadata != nil {
		t.Error("a function given no metadata has some")
	}
}

// A node with metadata is set as any other, and its tools are advertised
// without it.
func TestNodeMetadataPassesThrough(t *testing.T) {
	fm, enq := newManager(t)
	tagged := withMetadata(taskNode("tagged", NodeFunction{
		Name:        "test_function",
		Description: "Test function",
		Handler:     stay,
		Metadata:    map[string]any{"exit": "done"},
	}), map[string]any{"graph_node": "n1"})

	mustInitialize(t, fm, tagged)

	if fm.CurrentNode() != "tagged" {
		t.Errorf("CurrentNode = %q, want tagged", fm.CurrentNode())
	}
	tools := advertisedTools(enq.frames())
	if len(tools) != 1 || tools[0].Name != "test_function" {
		t.Fatalf("advertised tools = %+v, want test_function alone", tools)
	}
	raw, err := json.Marshal(map[string]any{
		"name": tools[0].Name, "description": tools[0].Description, "parameters": tools[0].Parameters,
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "exit") || strings.Contains(string(raw), "metadata") {
		t.Errorf("advertised tool = %s, want no metadata in it", raw)
	}
}

// A node built in code has nothing to decide a deferred transition.
func TestTransitionInYAMLIsRejectedOutsideAConfiguredFlow(t *testing.T) {
	fm, enq := newManager(t)
	defers := NodeFunction{
		Name: "defers",
		Handler: func(context.Context, json.RawMessage, *FlowManager) (string, *NodeConfig, error) {
			return `{"ok": true}`, TransitionInYAML, nil
		},
	}
	mustInitialize(t, fm, taskNode("start", defers))

	result, _, err := call(t, advertisedHandler(t, enq.frames(), "defers"))
	if err != nil {
		t.Fatalf("calling defers: %v", err)
	}
	var reported struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	if err := json.Unmarshal([]byte(result), &reported); err != nil {
		t.Fatalf("result %q is not JSON: %v", result, err)
	}
	if reported.Status != "error" {
		t.Errorf("status = %q, want error", reported.Status)
	}
	for _, want := range []string{"TransitionInYAML", "not built from a flow config"} {
		if !strings.Contains(reported.Error, want) {
			t.Errorf("error = %q, want it to contain %q", reported.Error, want)
		}
	}
	if fm.CurrentNode() != "start" {
		t.Errorf("CurrentNode = %q, want the flow to stay on start", fm.CurrentNode())
	}
}

func TestRTVIClientIsToldAboutEachNode(t *testing.T) {
	booking := withMetadata(taskNode("booking"), map[string]any{"step": 2})
	start := taskNode("start", edge("begin", booking, map[string]any{"exit": "begun"}))
	fm, enq := newManager(t)

	mustInitialize(t, fm, start)
	if got, want := rtviMessages(t, enq), []string{`{"node":"start"}`}; !slices.Equal(got, want) {
		t.Fatalf("node messages = %v, want %v", got, want)
	}

	invoke(t, enq, "begin")
	want := `{"node":"booking","previous_node":"start","function":"begin",` +
		`"metadata":{"step":2},"function_metadata":{"exit":"begun"}}`
	if got := rtviMessages(t, enq); len(got) != 2 || got[1] != want {
		t.Fatalf("node messages = %v, want the second to be %s", got, want)
	}
}

func TestParentOfASubflowNode(t *testing.T) {
	fm, enq, entered := managerWithEntered(t)
	mustInitialize(t, fm, taskNode("start"))
	ask := taskNode("verify.ask")
	ask.Parent = "verify"
	mustSetNode(t, fm, ask)

	if (*entered)[0].Parent != "" {
		t.Errorf("the first node's parent = %q, want none", (*entered)[0].Parent)
	}
	if (*entered)[1].Parent != "verify" {
		t.Errorf("the subflow node's parent = %q, want verify", (*entered)[1].Parent)
	}
	want := []string{
		`{"node":"start"}`,
		`{"node":"verify.ask","previous_node":"start","parent":"verify"}`,
	}
	if got := rtviMessages(t, enq); !slices.Equal(got, want) {
		t.Errorf("node messages = %v, want %v", got, want)
	}
}

func TestMetadataThatIsNotJSONIsLeftOutOfTheRTVIMessage(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(previous) })

	lock := make(chan struct{})
	booking := withMetadata(taskNode("booking"), map[string]any{"lock": lock})
	start := taskNode("start", edge("begin", booking, map[string]any{"handle": func() {}}))
	fm, enq, entered := managerWithEntered(t)

	mustInitialize(t, fm, start)
	invoke(t, enq, "begin")

	// The node is entered, and handlers get the metadata as the application
	// wrote it.
	if fm.CurrentNode() != "booking" {
		t.Errorf("CurrentNode = %q, want booking", fm.CurrentNode())
	}
	if types := queuedFrameTypes(enq); types[len(types)-1] != typeLLMRun {
		t.Errorf("the last frame queued is a %s, want the LLM run", types[len(types)-1])
	}
	if got, ok := (*entered)[1].Metadata["lock"].(chan struct{}); !ok || got != lock {
		t.Errorf("handler metadata = %v, want the application's own value", (*entered)[1].Metadata)
	}
	want := `{"node":"booking","previous_node":"start","function":"begin"}`
	if got := rtviMessages(t, enq); len(got) != 2 || got[1] != want {
		t.Errorf("node messages = %v, want the second to be %s", got, want)
	}
	if n := strings.Count(logs.String(), "level=WARN"); n != 2 {
		t.Errorf("%d warnings logged, want 2:\n%s", n, logs.String())
	}
}

func TestRTVIMessageFollowsTheHandlersAndPrecedesTheLLMRun(t *testing.T) {
	fm, enq := newManager(t)
	queuedAtHandler := -1
	event.On(fm.Events(), EventNodeEntered, func(context.Context, NodeEntered) {
		queuedAtHandler = len(enq.frames())
	})

	mustInitialize(t, fm, taskNode("start"))

	types := queuedFrameTypes(enq)
	if queuedAtHandler != len(types)-2 {
		t.Fatalf("the handler ran with %d of %d frames queued, want it just before the last two",
			queuedAtHandler, len(types))
	}
	if got, want := types[len(types)-2:], []string{typeFlowNodeEntered, typeLLMRun}; !slices.Equal(got, want) {
		t.Errorf("the last frames queued = %v, want %v", got, want)
	}
}

func TestInitializeEntersTheFirstNode(t *testing.T) {
	fm, _, entered := managerWithEntered(t)
	mustInitialize(t, fm, taskNode("start"))

	if want := []NodeEntered{{Node: "start"}}; !reflect.DeepEqual(*entered, want) {
		t.Errorf("entered = %+v, want %+v", *entered, want)
	}
}

func TestInitializeWithoutANodeFiresNoNodeEntered(t *testing.T) {
	fm, _, entered := managerWithEntered(t)
	mustInitialize(t, fm, nil)
	if len(*entered) != 0 {
		t.Errorf("entered = %+v, want nothing", *entered)
	}
}

func TestEdgeFunctionTransitionIsReported(t *testing.T) {
	booking := withMetadata(taskNode("booking"), map[string]any{"step": 2})
	start := taskNode("start",
		edge("begin", booking, map[string]any{"exit": "begun"}),
		edge("skip", booking, nil),
	)
	fm, enq, entered := managerWithEntered(t)
	mustInitialize(t, fm, start)

	invoke(t, enq, "begin")

	want := NodeEntered{
		Node:             "booking",
		PreviousNode:     "start",
		Function:         "begin",
		Metadata:         map[string]any{"step": 2},
		FunctionMetadata: map[string]any{"exit": "begun"},
	}
	if !reflect.DeepEqual((*entered)[1], want) {
		t.Errorf("entered = %+v, want %+v", (*entered)[1], want)
	}
}

func TestFunctionWithoutMetadataIsReported(t *testing.T) {
	start := taskNode("start", edge("skip", taskNode("booking"), nil))
	fm, enq, entered := managerWithEntered(t)
	mustInitialize(t, fm, start)

	invoke(t, enq, "skip")

	want := NodeEntered{Node: "booking", PreviousNode: "start", Function: "skip"}
	if !reflect.DeepEqual((*entered)[1], want) {
		t.Errorf("entered = %+v, want %+v", (*entered)[1], want)
	}
}

func TestGlobalFunctionTransitionIsReported(t *testing.T) {
	helpDesk := taskNode("help_desk")
	fm, enq, entered := managerWithEntered(t, func(c *Config) {
		c.GlobalFunctions = []NodeFunction{edge("get_help", helpDesk, map[string]any{"exit": "help"})}
	})
	mustInitialize(t, fm, taskNode("start"))

	invoke(t, enq, "get_help")

	if (*entered)[1].Function != "get_help" {
		t.Errorf("function = %q, want get_help", (*entered)[1].Function)
	}
	if want := map[string]any{"exit": "help"}; !reflect.DeepEqual((*entered)[1].FunctionMetadata, want) {
		t.Errorf("function metadata = %v, want %v", (*entered)[1].FunctionMetadata, want)
	}
}

func TestFunctionMetadataComesFromTheNodeBeingLeft(t *testing.T) {
	end := taskNode("end")
	middle := taskNode("middle", edge("go", end, map[string]any{"from": "middle"}))
	start := taskNode("start", edge("go", middle, map[string]any{"from": "start"}))
	fm, enq, entered := managerWithEntered(t)
	mustInitialize(t, fm, start)

	invoke(t, enq, "go")
	invoke(t, enq, "go")

	var nodes []string
	for _, e := range *entered {
		nodes = append(nodes, e.Node)
	}
	if want := []string{"start", "middle", "end"}; !slices.Equal(nodes, want) {
		t.Fatalf("entered = %v, want %v", nodes, want)
	}
	if want := map[string]any{"from": "start"}; !reflect.DeepEqual((*entered)[1].FunctionMetadata, want) {
		t.Errorf("the first move's function metadata = %v, want %v", (*entered)[1].FunctionMetadata, want)
	}
	if want := map[string]any{"from": "middle"}; !reflect.DeepEqual((*entered)[2].FunctionMetadata, want) {
		t.Errorf("the second move's function metadata = %v, want %v", (*entered)[2].FunctionMetadata, want)
	}
}

func TestSetNodeIsReported(t *testing.T) {
	fm, _, entered := managerWithEntered(t)
	mustInitialize(t, fm, withMetadata(taskNode("start"), map[string]any{"step": 1}))
	mustSetNode(t, fm, taskNode("other"))

	if want := map[string]any{"step": 1}; !reflect.DeepEqual((*entered)[0].Metadata, want) {
		t.Errorf("the first node's metadata = %v, want %v", (*entered)[0].Metadata, want)
	}
	if want := (NodeEntered{Node: "other", PreviousNode: "start"}); !reflect.DeepEqual((*entered)[1], want) {
		t.Errorf("entered = %+v, want %+v", (*entered)[1], want)
	}
}

func TestReenteringANodeFiresAgain(t *testing.T) {
	fm, _, entered := managerWithEntered(t)
	mustInitialize(t, fm, taskNode("start"))
	mustSetNode(t, fm, taskNode("start"))

	want := []NodeEntered{{Node: "start"}, {Node: "start", PreviousNode: "start"}}
	if !reflect.DeepEqual(*entered, want) {
		t.Errorf("entered = %+v, want %+v", *entered, want)
	}
}

func TestHandlerRunsAfterTheNodeIsSetAndBeforeTheLLMRuns(t *testing.T) {
	fm, enq := newManager(t)
	var current string
	var before []string
	calls := 0
	event.On(fm.Events(), EventNodeEntered, func(context.Context, NodeEntered) {
		calls++
		current = fm.CurrentNode()
		before = queuedFrameTypes(enq)
	})

	mustInitialize(t, fm, taskNode("start"))

	if calls != 1 {
		t.Fatalf("the handler ran %d times, want once", calls)
	}
	if current != "start" {
		t.Errorf("CurrentNode in the handler = %q, want start", current)
	}
	if !slices.Contains(before, typeLLMSetTools) {
		t.Errorf("frames queued before the handler = %v, want the tools among them", before)
	}
	if slices.Contains(before, typeLLMRun) {
		t.Errorf("frames queued before the handler = %v, want no LLM run yet", before)
	}
	if types := queuedFrameTypes(enq); types[len(types)-1] != typeLLMRun {
		t.Errorf("the last frame queued is a %s, want the LLM run", types[len(types)-1])
	}
}

func TestFailingHandlerDoesNotStopTheTransition(t *testing.T) {
	fm, enq := newManager(t)
	var later []string
	event.On(fm.Events(), EventNodeEntered, func(context.Context, NodeEntered) {
		panic("handler failed") //nolint:forbidigo // the failing handler under test
	})
	event.On(fm.Events(), EventNodeEntered, func(_ context.Context, e NodeEntered) {
		later = append(later, e.Node)
	})

	mustInitialize(t, fm, taskNode("start", edge("go", taskNode("next"), nil)))
	invoke(t, enq, "go")

	if fm.CurrentNode() != "next" {
		t.Errorf("CurrentNode = %q, want next", fm.CurrentNode())
	}
	if want := []string{"start", "next"}; !slices.Equal(later, want) {
		t.Errorf("the handler after the failing one saw %v, want %v", later, want)
	}
	runs := 0
	for _, frameType := range queuedFrameTypes(enq) {
		if frameType == typeLLMRun {
			runs++
		}
	}
	if runs != 2 {
		t.Errorf("%d LLM runs queued, want 2", runs)
	}
}

func TestFramesAreTheSameWithAndWithoutAHandler(t *testing.T) {
	flow := func() *NodeConfig {
		next := withMetadata(taskNode("next"), map[string]any{"step": 2})
		return taskNode("start", edge("go", next, nil))
	}

	withHandler, withEnq, entered := managerWithEntered(t)
	mustInitialize(t, withHandler, flow())
	invoke(t, withEnq, "go")

	without, withoutEnq := newManager(t)
	mustInitialize(t, without, flow())
	invoke(t, withoutEnq, "go")

	if len(*entered) != 2 {
		t.Errorf("entered %d nodes, want 2", len(*entered))
	}
	if without.CurrentNode() != "next" {
		t.Errorf("CurrentNode = %q, want next", without.CurrentNode())
	}
	if got, want := queuedFrameTypes(withEnq), queuedFrameTypes(withoutEnq); !slices.Equal(got, want) {
		t.Errorf("frames with a handler = %v, without = %v", got, want)
	}
}

// renderManager is a manager whose state holds what the prompts below refer to.
func renderManager(t *testing.T) (*FlowManager, *fakeEnq) {
	t.Helper()
	fm, enq := newManager(t)
	fm.State().Set("restaurant", "Luigi's")
	fm.State().Set("caller", "friend")
	return fm, enq
}

// queuedMessageTexts is the text of every message queued for the context.
func queuedMessageTexts(enq *fakeEnq) []string {
	var out []string
	for _, f := range enq.frames() {
		var messages []frames.Message
		switch fr := f.(type) {
		case *frames.LLMMessagesAppendFrame:
			messages = fr.Messages
		case *frames.LLMMessagesUpdateFrame:
			messages = fr.Messages
		}
		for _, m := range messages {
			out = append(out, m.Text)
		}
	}
	return out
}

func TestPlaceholdersAreFilledFromStateOnEntry(t *testing.T) {
	fm, enq := renderManager(t)
	fm.State().Set("caller", "Ada")
	greet := &NodeConfig{
		Name:         "greet",
		RoleMessage:  "You work for {{ restaurant }}.",
		TaskMessages: []frames.Message{{Role: frames.RoleDeveloper, Text: "Greet the {{ caller }}."}},
	}
	mustInitialize(t, fm, greet)

	if !slices.Contains(queuedMessageTexts(enq), "Greet the Ada.") {
		t.Errorf("queued messages = %q, want the placeholder filled", queuedMessageTexts(enq))
	}
	// The flow's own node is untouched; rendering happens on a copy.
	if got := greet.TaskMessages[0].Text; got != "Greet the {{ caller }}." {
		t.Errorf("the node's own task message = %q, want it left as written", got)
	}
	if got := greet.RoleMessage; got != "You work for {{ restaurant }}." {
		t.Errorf("the node's own role message = %q, want it left as written", got)
	}
}

func TestRenderNodeWalksIntoMaps(t *testing.T) {
	fm, _ := renderManager(t)
	fm.State().Set("order", map[string]any{"size": "large", "type": "cheese"})
	ran := false
	check := func(context.Context, ActionConfig, *FlowManager) error {
		ran = true
		return nil
	}
	n := &NodeConfig{
		RoleMessage: "You work for {{ restaurant }}.",
		TaskMessages: []frames.Message{
			{Role: frames.RoleDeveloper, Text: "Confirm the {{ order.size }} pizza."},
		},
		PostActions: []ActionConfig{
			{Type: "tts_say", Text: "One {{order.type}} pizza."},
			{Type: "function", Handler: check},
		},
	}
	rendered, err := fm.renderNode("n", n)
	if err != nil {
		t.Fatalf("renderNode: %v", err)
	}
	if rendered.RoleMessage != "You work for Luigi's." {
		t.Errorf("role message = %q", rendered.RoleMessage)
	}
	if got := rendered.TaskMessages[0].Text; got != "Confirm the large pizza." {
		t.Errorf("task message = %q", got)
	}
	if got := rendered.PostActions[0].Text; got != "One cheese pizza." {
		t.Errorf("tts_say text = %q", got)
	}
	if err := rendered.PostActions[1].Handler(context.Background(), rendered.PostActions[1], fm); err != nil || !ran {
		t.Errorf("the function action's handler was not carried over (ran=%t, err=%v)", ran, err)
	}
}

func TestBackslashEscapesAPlaceholder(t *testing.T) {
	fm, _ := renderManager(t)
	n := &NodeConfig{TaskMessages: []frames.Message{
		{Role: frames.RoleDeveloper, Text: `Write \{{ caller }} for {{ caller }}.`},
	}}
	rendered, err := fm.renderNode("n", n)
	if err != nil {
		t.Fatalf("renderNode: %v", err)
	}
	if got := rendered.TaskMessages[0].Text; got != "Write {{ caller }} for friend." {
		t.Errorf("task message = %q", got)
	}
}

func TestMissingStateKeyFailsOnEntry(t *testing.T) {
	fm, _ := renderManager(t)
	fm.State().Delete("caller")
	greet := &NodeConfig{
		Name:         "greet",
		TaskMessages: []frames.Message{{Role: frames.RoleDeveloper, Text: "Greet the {{ caller }}."}},
	}
	err := fm.Initialize(context.Background(), greet)
	if !errors.Is(err, ErrFlow) {
		t.Fatalf("Initialize = %v, want an ErrFlow", err)
	}
	if want := "uses '{{ caller }}', which is not in state"; !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err, want)
	}
}
