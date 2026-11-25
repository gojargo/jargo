package flows

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"testing"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Tests for a Flow: a FlowConfig joined to the handlers it names.

// toolbox is a set of handlers, as an application would write it, and what its
// action handler was called with.
type toolbox struct {
	handlers      Handlers
	kitchenChecks []ActionConfig
}

// inYAML is a tool that returns result and leaves the transition to the config.
func inYAML(description string, result func(args map[string]any, fm *FlowManager) any) NodeFunction {
	return NodeFunction{
		Description: description,
		Handler: func(_ context.Context, raw json.RawMessage, fm *FlowManager) (string, *NodeConfig, error) {
			var args map[string]any
			if err := json.Unmarshal(raw, &args); err != nil {
				return "", nil, err
			}
			value := result(args, fm)
			if value == nil {
				return "", TransitionInYAML, nil
			}
			out, err := json.Marshal(value)
			return string(out), TransitionInYAML, err
		},
	}
}

func newToolbox() *toolbox {
	box := &toolbox{}
	cancel, timeout := true, 12.0

	selectPizzaOrder := inYAML("Record the pizza order details.", func(args map[string]any, fm *FlowManager) any {
		if fm != nil {
			fm.State().Set("order", map[string]any{"size": args["size"], "pizza_type": args["pizza_type"]})
		}
		return map[string]any{"size": args["size"], "type": args["pizza_type"], "status": "ok"}
	})
	selectPizzaOrder.Properties = map[string]any{
		"size":       map[string]any{"type": "string", "description": `One of "small", "medium", or "large".`},
		"pizza_type": map[string]any{"type": "string", "description": "The kind of pizza."},
	}
	selectPizzaOrder.Required = []string{"size", "pizza_type"}
	selectPizzaOrder.CancelOnInterruption = &cancel
	selectPizzaOrder.TimeoutSecs = &timeout

	fixed := func(result string, next *NodeConfig) Handler {
		return func(context.Context, json.RawMessage, *FlowManager) (string, *NodeConfig, error) {
			return result, next, nil
		}
	}

	box.handlers = Handlers{
		"choose_pizza": inYAML("User wants to order pizza.", func(map[string]any, *FlowManager) any {
			return nil
		}),
		"select_pizza_order": selectPizzaOrder,
		"report": inYAML("Report a status the config can branch on.", func(args map[string]any, _ *FlowManager) any {
			return map[string]any{"status": args["status"]}
		}),
		"report_flag": inYAML("Report a boolean the config can branch on.", func(args map[string]any, _ *FlowManager) any {
			return map[string]any{"flag": args["flag"]}
		}),
		"report_flag_int": inYAML("Report an integer to branch on.", func(args map[string]any, _ *FlowManager) any {
			return map[string]any{"code": args["code"]}
		}),
		"get_delivery_estimate": inYAML("Get a delivery estimate.", func(map[string]any, *FlowManager) any {
			return map[string]any{"time": "30 minutes"}
		}),
		"stay_quiet": NodeFunction{
			Description: "Do work without prompting a bot response.",
			Handler:     fixed(`{"ok": true}`, NoResponse),
		},
		"returns_none": NodeFunction{
			Description: "Break the contract with nil as the next node.",
			Handler:     fixed(`{"ok": true}`, nil),
		},
		"returns_node": NodeFunction{
			Description: "Break the contract with a node.",
			Handler:     fixed("", &NodeConfig{TaskMessages: []frames.Message{}}),
		},
		"returns_non_mapping": NodeFunction{
			Description: "Return a result a branch cannot read.",
			Handler:     fixed(`"sold out"`, TransitionInYAML),
		},
		"no_handler": NodeFunction{Description: "A tool that was never given its code."},
		"check_kitchen_status": func(_ context.Context, action ActionConfig, _ *FlowManager) error {
			box.kitchenChecks = append(box.kitchenChecks, action)
			return nil
		},
		"NOT_A_TOOL": "just a constant",
	}
	return box
}

// flowConfigData is a three-node flow document: initial, pizza and confirm.
func flowConfigData() dict {
	return dict{
		"initial_node": "initial",
		"nodes": dict{
			"initial": dict{
				"role_message":  "You work for {{ restaurant }}.",
				"task_messages": []any{dict{"role": "developer", "content": "Greet the {{ caller }}."}},
				"pre_actions":   []any{dict{"type": "function", "handler": "check_kitchen_status"}},
				"functions":     []any{dict{"name": "choose_pizza", "transition_to": "pizza"}},
			},
			"pizza": dict{
				"task_messages":       []any{dict{"role": "developer", "content": "Take the order."}},
				"context_strategy":    "reset",
				"respond_immediately": false,
				"functions":           []any{dict{"name": "select_pizza_order", "transition_to": "confirm"}},
				"post_actions":        []any{dict{"type": "tts_say", "text": "Thanks, {{ caller }}!"}},
			},
			"confirm": dict{
				"task_messages": []any{dict{"role": "developer", "content": "Confirm."}},
			},
		},
		"global_functions": []any{dict{"name": "get_delivery_estimate"}},
	}
}

func flowConfig(t *testing.T, overrides ...func(dict)) *FlowConfig {
	t.Helper()
	data := flowConfigData()
	for _, override := range overrides {
		override(data)
	}
	return mustLoadMap(t, data)
}

// singleNode is a config with one node, "a", offering one function, plus the
// nodes "next" and "other".
func singleNode(t *testing.T, function dict, nodeExtras ...dict) *FlowConfig {
	t.Helper()
	a := dict{
		"task_messages": []any{dict{"role": "developer", "content": "a"}},
		"functions":     []any{function},
	}
	for _, extras := range nodeExtras {
		maps.Copy(a, extras)
	}
	return mustLoadMap(t, dict{
		"initial_node": "a",
		"nodes": dict{
			"a":     a,
			"next":  dict{"task_messages": []any{dict{"role": "developer", "content": "next"}}},
			"other": dict{"task_messages": []any{dict{"role": "developer", "content": "other"}}},
		},
	})
}

func makeFlow(t *testing.T, config *FlowConfig, handlers ...Handlers) *Flow {
	t.Helper()
	if handlers == nil {
		handlers = []Handlers{newToolbox().handlers}
	}
	flow, err := NewFlow(config, handlers...)
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	return flow
}

// flowProblems builds a flow that must fail, and returns why.
func flowProblems(t *testing.T, config *FlowConfig, handlers ...Handlers) *FlowReferenceError {
	t.Helper()
	if handlers == nil {
		handlers = []Handlers{newToolbox().handlers}
	}
	_, err := NewFlow(config, handlers...)
	var reference *FlowReferenceError
	if !errors.As(err, &reference) {
		t.Fatalf("NewFlow returned %v, want a FlowReferenceError", err)
	}
	if !errors.Is(err, ErrFlow) {
		t.Error("a FlowReferenceError should match ErrFlow")
	}
	return reference
}

func flowNode(t *testing.T, flow *Flow, name string) *NodeConfig {
	t.Helper()
	node, err := flow.Node(name)
	if err != nil {
		t.Fatalf("Node(%q): %v", name, err)
	}
	return node
}

func assertSays(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("got error %v, want one saying %q", err, want)
	}
}

func nodeFunctionNames(functions []NodeFunction) []string {
	names := make([]string, len(functions))
	for i := range functions {
		names[i] = functions[i].Name
	}
	return names
}

// Construction

func TestFlowKeepsItsConfig(t *testing.T) {
	config := flowConfig(t)
	if makeFlow(t, config).Config() != config {
		t.Error("the flow should hand back the config it was built from")
	}
}

func TestFlowHandlersFromAMap(t *testing.T) {
	box := newToolbox()
	handlers := Handlers{
		"choose_pizza":          box.handlers["choose_pizza"],
		"select_pizza_order":    box.handlers["select_pizza_order"],
		"get_delivery_estimate": box.handlers["get_delivery_estimate"],
		"check_kitchen_status":  box.handlers["check_kitchen_status"],
		"unrelated":             42,
	}
	flow := makeFlow(t, flowConfig(t), handlers)
	assertEqual(t, flow.InitialNode().Functions[0].Name, "choose_pizza")
}

func TestFlowHandlersFromSeveralNamespaces(t *testing.T) {
	box := newToolbox()
	shared := Handlers{
		"choose_pizza":          box.handlers["choose_pizza"],
		"get_delivery_estimate": box.handlers["get_delivery_estimate"],
	}
	actions := Handlers{"check_kitchen_status": box.handlers["check_kitchen_status"]}
	rest := Handlers{"select_pizza_order": box.handlers["select_pizza_order"]}

	flow := makeFlow(t, flowConfig(t), shared, actions, rest)
	handler := flowNode(t, flow, "initial").PreActions[0].Handler
	if handler == nil {
		t.Fatal("the pre-action should carry the handler the config names")
	}
	if err := handler(context.Background(), ActionConfig{Type: "function"}, nil); err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(box.kitchenChecks), 1)
	assertEqual(t, flowNode(t, flow, "pizza").Functions[0].Name, "select_pizza_order")
}

func TestFlowFunctionCarriesTheConfiguredName(t *testing.T) {
	config := mustLoadMap(t, dict{
		"initial_node": "a",
		"nodes": dict{
			"a": dict{
				"task_messages": []any{dict{"role": "developer", "content": "a"}},
				"functions": []any{
					dict{"name": "accept", "transition_to": "next"},
					dict{"name": "reject", "transition_to": "other"},
				},
			},
			"next":  dict{"task_messages": []any{dict{"role": "developer", "content": "n"}}},
			"other": dict{"task_messages": []any{dict{"role": "developer", "content": "o"}}},
		},
	})
	choosePizza := newToolbox().handlers["choose_pizza"]
	flow := makeFlow(t, config, Handlers{"accept": choosePizza, "reject": choosePizza})
	assertEqual(t, nodeFunctionNames(flowNode(t, flow, "a").Functions), []string{"accept", "reject"})
}

func TestFlowNameInTwoNamespacesIsAmbiguous(t *testing.T) {
	override := NodeFunction{Description: "Override.", Handler: stay}
	reference := flowProblems(t, singleNode(t, dict{"name": "choose_pizza"}),
		Handlers{"choose_pizza": override}, newToolbox().handlers)

	problem := reference.Problems[0]
	assertEqual(t, problem.Code, ProblemAmbiguousTool)
	assertEqual(t, problem.Function, "choose_pizza")
	for _, want := range []string{
		"tool 'choose_pizza' is defined in more than one of the handlers", "handlers[0]", "handlers[1]",
	} {
		if !strings.Contains(problem.Message, want) {
			t.Errorf("message %q does not say %q", problem.Message, want)
		}
	}
}

func TestFlowHandlerInTwoNamespacesIsAmbiguous(t *testing.T) {
	other := ActionHandler(func(context.Context, ActionConfig, *FlowManager) error { return nil })
	reference := flowProblems(t, flowConfig(t), newToolbox().handlers, Handlers{"check_kitchen_status": other})
	assertEqual(t, reference.Problems[0].Code, ProblemAmbiguousHandler)
}

func TestFlowMissingAcrossEveryNamespace(t *testing.T) {
	reference := flowProblems(t, singleNode(t, dict{"name": "no_such_tool"}), Handlers{}, Handlers{})
	assertSays(t, reference, "references tool 'no_such_tool'")
}

func TestFlowTransitionOnlyFunctionNeedsNoTool(t *testing.T) {
	config := singleNode(t, dict{
		"name":            "start_over",
		"transition_only": true,
		"description":     "Start over at {{ restaurant }}.",
		"transition_to":   "next",
	})
	flow := makeFlow(t, config, Handlers{})
	function := flowNode(t, flow, "a").Functions[0]
	assertEqual(t, function.Name, "start_over")
	assertEqual(t, function.Description, "Start over at {{ restaurant }}.")
	assertEqual(t, function.Properties, map[string]any{})
	assertEqual(t, function.Required, []string{})
}

func TestFlowMissingTool(t *testing.T) {
	reference := flowProblems(t, singleNode(t, dict{"name": "no_such_tool"}))
	assertSays(t, reference, "node 'a' references tool 'no_such_tool'")
}

func TestFlowMissingGlobalTool(t *testing.T) {
	config := flowConfig(t, func(data dict) {
		data["global_functions"] = []any{dict{"name": "no_such_tool"}}
	})
	assertSays(t, flowProblems(t, config), "global_functions references tool 'no_such_tool'")
}

func TestFlowMissingActionHandler(t *testing.T) {
	config := singleNode(t, dict{"name": "choose_pizza"},
		dict{"pre_actions": []any{dict{"type": "function", "handler": "no_such_handler"}}})
	assertSays(t, flowProblems(t, config), "pre_actions references action handler 'no_such_handler'")
}

func TestFlowReferenceThatIsNotATool(t *testing.T) {
	reference := flowProblems(t, singleNode(t, dict{"name": "NOT_A_TOOL"}))
	assertEqual(t, reference.Problems[0].Code, ProblemInvalidTool)
	assertSays(t, reference, "is not a NodeFunction")

	// An action handler is held to its own type.
	config := singleNode(t, dict{"name": "choose_pizza"},
		dict{"pre_actions": []any{dict{"type": "function", "handler": "choose_pizza"}}})
	reference = flowProblems(t, config)
	assertEqual(t, reference.Problems[0].Code, ProblemMissingHandler)
	assertSays(t, reference, "action handler 'choose_pizza' is not an ActionHandler")
}

func TestFlowToolWithoutAHandlerIsInvalid(t *testing.T) {
	reference := flowProblems(t, singleNode(t, dict{"name": "no_handler"}))
	assertEqual(t, reference.Problems[0].Code, ProblemInvalidTool)
	assertSays(t, reference, "not a valid NodeFunction")
	assertSays(t, reference, "it has no handler")
}

func TestFlowEveryProblemIsReportedTogether(t *testing.T) {
	config := singleNode(t, dict{"name": "no_such_tool"},
		dict{"pre_actions": []any{dict{"type": "function", "handler": "no_such_handler"}}})
	reference := flowProblems(t, config)

	codes := make([]string, len(reference.Problems))
	for i, problem := range reference.Problems {
		codes[i] = problem.Code
	}
	assertEqual(t, codes, []string{ProblemMissingTool, ProblemMissingHandler})
	assertEqual(t, reference.Problems[0].Node, "a")
	assertEqual(t, reference.Problems[0].Function, "no_such_tool")
	assertSays(t, reference, "2 problems")
	assertSays(t, reference, "no_such_handler")
}

func TestFlowPlaceholdersAreLeftForTheManager(t *testing.T) {
	initial := flowNode(t, makeFlow(t, flowConfig(t)), "initial")
	assertEqual(t, initial.RoleMessage, "You work for {{ restaurant }}.")
}

func TestFlowInitialNode(t *testing.T) {
	flow := makeFlow(t, flowConfig(t))
	if flow.InitialNode() != flowNode(t, flow, "initial") {
		t.Error("the initial node should be the node the config names")
	}
}

func TestFlowUnknownNode(t *testing.T) {
	_, err := makeFlow(t, flowConfig(t)).Node("nope")
	if !errors.Is(err, ErrFlow) {
		t.Fatalf("got %v, want an ErrFlow", err)
	}
	assertSays(t, err, "flow has no node 'nope'")
}

func TestFlowNodeShape(t *testing.T) {
	flow := makeFlow(t, flowConfig(t))

	pizza := flowNode(t, flow, "pizza")
	assertEqual(t, pizza.Name, "pizza")
	assertEqual(t, pizza.TaskMessages, []frames.Message{{Role: frames.RoleDeveloper, Text: "Take the order."}})
	if pizza.respondsImmediately() {
		t.Error("pizza should not respond immediately")
	}
	assertEqual(t, pizza.ContextStrategy, &ContextStrategyConfig{Strategy: ContextStrategyReset})
	assertEqual(t, pizza.RoleMessage, "")
	assertEqual(t, len(pizza.PreActions), 0)

	initial := flowNode(t, flow, "initial")
	if !initial.respondsImmediately() {
		t.Error("initial should respond immediately")
	}
	if initial.ContextStrategy != nil {
		t.Error("initial should leave the context strategy to the manager")
	}

	assertEqual(t, len(flowNode(t, flow, "confirm").Functions), 0)
}

func TestFlowFunctionFromANodeFunction(t *testing.T) {
	function := flowNode(t, makeFlow(t, flowConfig(t)), "pizza").Functions[0]
	assertEqual(t, function.Name, "select_pizza_order")
	assertEqual(t, function.Description, "Record the pizza order details.")
	assertEqual(t, sortedKeys(function.Properties), []string{"pizza_type", "size"})
	size, _ := function.Properties["size"].(map[string]any)
	assertEqual(t, size["description"], `One of "small", "medium", or "large".`)
	assertEqual(t, function.Required, []string{"size", "pizza_type"})
	if function.CancelOnInterruption == nil || !*function.CancelOnInterruption {
		t.Error("the tool's cancel-on-interruption option was lost")
	}
	if function.TimeoutSecs == nil || *function.TimeoutSecs != 12 {
		t.Error("the tool's timeout was lost")
	}
}

func TestFlowToolWithoutOptionsKeepsTheDefaults(t *testing.T) {
	function := flowNode(t, makeFlow(t, flowConfig(t)), "initial").Functions[0]
	if function.CancelOnInterruption != nil || function.TimeoutSecs != nil {
		t.Error("a tool without call options should leave them unset")
	}
}

func TestFlowActions(t *testing.T) {
	box := newToolbox()
	pre := flowNode(t, makeFlow(t, flowConfig(t), box.handlers), "initial").PreActions
	assertEqual(t, len(pre), 1)
	assertEqual(t, pre[0].Type, "function")
	if err := pre[0].Handler(context.Background(), pre[0], nil); err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(box.kitchenChecks), 1)
}

func TestFlowCustomActionHandlerResolvesByName(t *testing.T) {
	box := newToolbox()
	config := singleNode(t, dict{"name": "choose_pizza"}, dict{"post_actions": []any{
		dict{"type": "notify", "handler": "check_kitchen_status", "channel": "#x"},
		dict{"type": "audit"},
	}})
	post := flowNode(t, makeFlow(t, config, box.handlers), "a").PostActions

	assertEqual(t, post[0].Type, "notify")
	assertEqual(t, post[0].Params, map[string]any{"channel": "#x"})
	if err := post[0].Handler(context.Background(), post[0], nil); err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(box.kitchenChecks), 1)
	assertEqual(t, post[1].Type, "audit")
	if post[1].Handler != nil || post[1].Params != nil {
		t.Error("an action without a handler or extra keys should carry neither")
	}
}

func TestFlowCustomActionHandlerMustExist(t *testing.T) {
	config := singleNode(t, dict{"name": "choose_pizza"},
		dict{"post_actions": []any{dict{"type": "notify", "handler": "nope"}}})
	assertSays(t, flowProblems(t, config), "post_actions references action handler 'nope'")

	post := flowNode(t, makeFlow(t, flowConfig(t)), "pizza").PostActions
	assertEqual(t, len(post), 1)
	assertEqual(t, post[0].Type, "tts_say")
	assertEqual(t, post[0].Text, "Thanks, {{ caller }}!")
}

// The keys an action passes through land where the action reads them: text and
// append_text_to_context on their fields, the rest in Params.
func TestFlowActionExtras(t *testing.T) {
	config := singleNode(t, dict{"name": "choose_pizza"}, dict{"post_actions": []any{
		dict{"type": "tts_say", "text": "Bye.", "append_text_to_context": false, "voice": "calm"},
	}})
	action := flowNode(t, makeFlow(t, config), "a").PostActions[0]
	assertEqual(t, action.Text, "Bye.")
	if action.AppendTextToContext == nil || *action.AppendTextToContext {
		t.Error("append_text_to_context should be false")
	}
	assertEqual(t, action.Params, map[string]any{"voice": "calm"})
}

func TestFlowNodeMetadata(t *testing.T) {
	config := flowConfig(t)
	config.Nodes["pizza"].Metadata = map[string]any{"graph_node": "n2", "node_type": "order"}
	flow := makeFlow(t, config)
	assertEqual(t, flowNode(t, flow, "pizza").Metadata, map[string]any{"graph_node": "n2", "node_type": "order"})
	if flowNode(t, flow, "initial").Metadata != nil {
		t.Error("a node without metadata should have none")
	}
}

func TestFlowFunctionMetadata(t *testing.T) {
	config := singleNode(t, dict{"name": "report", "transition_to": "next", "metadata": dict{"exit": "ok"}})
	a := config.Nodes["a"]
	a.Functions = append(a.Functions,
		FlowConfigFunction{
			Name:           "skip",
			TransitionOnly: true,
			Description:    "Skip ahead.",
			TransitionTo:   FlowConfigTransition{Node: "other"},
			Metadata:       map[string]any{"exit": "skipped"},
		},
		FlowConfigFunction{Name: "stay_quiet"},
	)
	config.GlobalFunctions = append(config.GlobalFunctions,
		FlowConfigFunction{Name: "get_delivery_estimate", Metadata: map[string]any{"scope": "global"}})
	if err := config.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	flow := makeFlow(t, config)

	functions := flowNode(t, flow, "a").Functions
	assertEqual(t, functions[0].Metadata, map[string]any{"exit": "ok"})
	assertEqual(t, functions[1].Metadata, map[string]any{"exit": "skipped"})
	if functions[2].Metadata != nil {
		t.Error("a function without metadata should have none")
	}
	assertEqual(t, flow.GlobalFunctions()[0].Metadata, map[string]any{"scope": "global"})
}

func TestFlowGlobalFunctions(t *testing.T) {
	flow := makeFlow(t, flowConfig(t))
	globals := flow.GlobalFunctions()
	assertEqual(t, nodeFunctionNames(globals), []string{"get_delivery_estimate"})

	// A fresh slice each time, so the caller may extend it.
	globals[0].Name = "changed"
	assertEqual(t, flow.GlobalFunctions()[0].Name, "get_delivery_estimate")
}

// The call-time contract

// callFunction calls the first function of a node the way the manager would.
func callFunction(t *testing.T, flow *Flow, node, args string) (string, *NodeConfig, error) {
	t.Helper()
	if args == "" {
		args = "{}"
	}
	fm, _ := newManager(t)
	return flowNode(t, flow, node).Functions[0].Handler(context.Background(), json.RawMessage(args), fm)
}

// nextNode is the node the first function of a node leads to.
func nextNode(t *testing.T, flow *Flow, node, args string) *NodeConfig {
	t.Helper()
	_, next, err := callFunction(t, flow, node, args)
	if err != nil {
		t.Fatalf("calling the function of %q: %v", node, err)
	}
	return next
}

func TestFlowEmptyResultWithTransitionUsesTheConfigEdge(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "choose_pizza", "transition_to": "next"}))
	result, next, err := callFunction(t, flow, "a", "")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, result, "")
	if next != flowNode(t, flow, "next") {
		t.Error("the tool should lead to the node the config names")
	}
}

func TestFlowTransitionOnlyFunctionMovesToItsNode(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{
		"name":            "start_over",
		"transition_only": true,
		"description":     "Start over.",
		"transition_to":   "next",
	}), Handlers{})
	result, next, err := callFunction(t, flow, "a", "")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, result, "")
	if next != flowNode(t, flow, "next") {
		t.Error("the function should lead to its node")
	}
}

func TestFlowToolWithoutTransitionStays(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "get_delivery_estimate"}))
	result, next, err := callFunction(t, flow, "a", "")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, result, `{"time":"30 minutes"}`)
	if next != nil {
		t.Error("a tool the config gives no transition should stay on its node")
	}
}

func TestFlowArgumentsReachTheTool(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "select_pizza_order", "transition_to": "next"}))
	fm, _ := newManager(t)
	result, next, err := flowNode(t, flow, "a").Functions[0].Handler(
		context.Background(), json.RawMessage(`{"size": "large", "pizza_type": "cheese"}`), fm)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, `"size":"large"`) {
		t.Errorf("result %s does not carry the size", result)
	}
	order, _ := fm.State().Get("order")
	assertEqual(t, order, map[string]any{"size": "large", "pizza_type": "cheese"})
	if next != flowNode(t, flow, "next") {
		t.Error("the tool should lead to the node the config names")
	}
}

func TestFlowNoResponsePassesThrough(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "stay_quiet", "transition_to": "next"}))
	result, next, err := callFunction(t, flow, "a", "")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, result, `{"ok": true}`)
	if next != NoResponse {
		t.Error("NoResponse should pass through, whatever the config's transition")
	}
}

func TestFlowNilNextNodeIsRejected(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "returns_none", "transition_to": "next"}))
	_, _, err := callFunction(t, flow, "a", "")
	if !errors.Is(err, ErrFlow) {
		t.Fatalf("got %v, want an ErrFlow", err)
	}
	assertSays(t, err, "node 'a' tool 'returns_none' returned nil as the next node")
	assertSays(t, err, "return (result, TransitionInYAML)")
}

func TestFlowNodeReturnIsRejected(t *testing.T) {
	for _, function := range []dict{
		{"name": "returns_node"},
		{"name": "returns_node", "transition_to": "next"},
	} {
		flow := makeFlow(t, singleNode(t, function))
		_, _, err := callFunction(t, flow, "a", "")
		if !errors.Is(err, ErrFlow) {
			t.Fatalf("got %v, want an ErrFlow", err)
		}
		assertSays(t, err, "return (result, TransitionInYAML)")
	}
}

// Branches

// branchFlow is a flow whose node "a" offers tool with the branch given.
func branchFlow(t *testing.T, tool string, branch dict) *Flow {
	t.Helper()
	return makeFlow(t, singleNode(t, dict{"name": tool, "transition_to": branch}))
}

func statusBranch() dict {
	return dict{"field": "status", "cases": dict{"ok": "next", "sold_out": "other"}}
}

func TestBranchCaseMatch(t *testing.T) {
	flow := branchFlow(t, "report", statusBranch())
	if nextNode(t, flow, "a", `{"status": "sold_out"}`) != flowNode(t, flow, "other") {
		t.Error("sold_out should lead to other")
	}
}

func TestBranchDefault(t *testing.T) {
	branch := statusBranch()
	branch["default"] = "next"
	flow := branchFlow(t, "report", branch)
	if nextNode(t, flow, "a", `{"status": "weird"}`) != flowNode(t, flow, "next") {
		t.Error("an unmatched value should lead to the default")
	}
}

func TestBranchNoMatchNoDefaultStays(t *testing.T) {
	flow := branchFlow(t, "report", statusBranch())
	if nextNode(t, flow, "a", `{"status": "weird"}`) != nil {
		t.Error("an unmatched value with no default should stay")
	}
}

func TestBranchBooleanValuesMatchBooleanOrStringKeys(t *testing.T) {
	for _, key := range []any{true, "true", "True"} {
		flow := branchFlow(t, "report_flag", dict{"field": "flag", "cases": map[any]any{key: "next"}})
		if nextNode(t, flow, "a", `{"flag": true}`) != flowNode(t, flow, "next") {
			t.Errorf("a result of true should match the case key %#v", key)
		}
	}
	flow := branchFlow(t, "report_flag", dict{"field": "flag", "cases": map[any]any{false: "next"}})
	if nextNode(t, flow, "a", `{"flag": true}`) != nil {
		t.Error("a result of true should not match the case key false")
	}
}

func TestBranchIntegerValuesMatchIntegerOrStringKeys(t *testing.T) {
	for _, key := range []any{7, "7"} {
		flow := branchFlow(t, "report_flag_int", dict{"field": "code", "cases": map[any]any{key: "next"}})
		if nextNode(t, flow, "a", `{"code": 7}`) != flowNode(t, flow, "next") {
			t.Errorf("a result of 7 should match the case key %#v", key)
		}
	}
}

func TestBranchMissingFieldIsAnError(t *testing.T) {
	flow := branchFlow(t, "report", dict{"field": "missing", "cases": dict{"x": "next"}})
	_, _, err := callFunction(t, flow, "a", `{"status": "ok"}`)
	if !errors.Is(err, ErrFlow) {
		t.Fatalf("got %v, want an ErrFlow", err)
	}
	assertSays(t, err, "branches on result field 'missing'")
}

func TestBranchResultThatIsNotAnObjectIsAnError(t *testing.T) {
	flow := branchFlow(t, "returns_non_mapping", statusBranch())
	_, _, err := callFunction(t, flow, "a", "")
	assertSays(t, err, "has no such field")
}

// Subflows

// composedFlowData is a flow that verifies the caller in a subflow, books, and
// closes in another.
func composedFlowData() dict {
	ask := promptNode(goTo("answered", "check"))
	ask["metadata"] = dict{"step": "ask"}
	return dict{
		"initial_node": "start",
		"flows": dict{
			"verification": dict{
				"initial_node": "ask",
				"nodes": dict{
					"ask": ask,
					"check": promptNode(dict{
						"name": "report_flag",
						"transition_to": dict{
							"field": "flag",
							"cases": dict{"true": "return", "false": "ask"},
						},
					}),
				},
			},
			"closing": dict{
				"initial_node": "anything_else",
				"nodes":        dict{"anything_else": promptNode(goTo("more", "return"))},
			},
		},
		"nodes": dict{
			"start":  promptNode(goTo("begin", "verify")),
			"verify": dict{"flow": "verification", "return_to": "book"},
			"book": promptNode(dict{
				"name":          "report",
				"transition_to": dict{"field": "status", "cases": dict{"booked": "goodbye"}},
			}),
			"goodbye": dict{"flow": "closing", "return_to": "book"},
		},
	}
}

func composedFlowConfig(t *testing.T, overrides ...func(dict)) *FlowConfig {
	t.Helper()
	data := composedFlowData()
	for _, override := range overrides {
		override(data)
	}
	return mustLoadMap(t, data)
}

func TestSubflowIsBuiltUnderTheNodeThatRunsIt(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	assertEqual(t, sortedKeys(flow.nodes),
		[]string{"book", "goodbye.anything_else", "start", "verify.ask", "verify.check"})
	assertEqual(t, flowNode(t, flow, "verify.ask").Name, "verify.ask")
	assertEqual(t, flowNode(t, flow, "verify.ask").Metadata, map[string]any{"step": "ask"})
	for _, name := range []string{"ask", "verification.ask"} {
		if _, err := flow.Node(name); !errors.Is(err, ErrFlow) {
			t.Errorf("Node(%q) returned %v, want an ErrFlow", name, err)
		}
	}
}

func TestSubflowNodeNamesTheNodeThatRunsIt(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	assertEqual(t, flowNode(t, flow, "verify.ask").Parent, "verify")
	assertEqual(t, flowNode(t, flow, "verify.check").Parent, "verify")
	assertEqual(t, flowNode(t, flow, "goodbye.anything_else").Parent, "goodbye")
	assertEqual(t, flowNode(t, flow, "start").Parent, "")
	assertEqual(t, flowNode(t, flow, "book").Parent, "")
}

func TestNodeThatRunsASubflowIsItsFirstNode(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	if flowNode(t, flow, "verify") != flowNode(t, flow, "verify.ask") {
		t.Error("verify should be the first node of the subflow it runs")
	}
	if flowNode(t, flow, "goodbye") != flowNode(t, flow, "goodbye.anything_else") {
		t.Error("goodbye should be the first node of the subflow it runs")
	}
}

func TestFlowWithoutSubflowsBuildsItsOwnNodesOnly(t *testing.T) {
	config := flowConfig(t)
	flow := makeFlow(t, config)
	assertEqual(t, sortedKeys(flow.nodes), sortedKeys(config.Nodes))
	for name := range config.Nodes {
		assertEqual(t, flowNode(t, flow, name).Name, name)
	}
}

func TestEnteringANodeThatRunsASubflowGoesToItsFirstNode(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	if nextNode(t, flow, "start", "") != flowNode(t, flow, "verify.ask") {
		t.Error("begin should enter the verification subflow at its first node")
	}
	if nextNode(t, flow, "book", `{"status": "booked"}`) != flowNode(t, flow, "goodbye.anything_else") {
		t.Error("a booking should enter the closing subflow at its first node")
	}
}

func TestTargetsInsideASubflowStayInsideIt(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	if nextNode(t, flow, "verify.ask", "") != flowNode(t, flow, "verify.check") {
		t.Error("answered should lead to the subflow's check node")
	}
	if nextNode(t, flow, "verify.check", `{"flag": false}`) != flowNode(t, flow, "verify.ask") {
		t.Error("a failed check should lead back to the subflow's ask node")
	}
}

func TestReturnGoesToTheReturnToOfTheNodeThatRanIt(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	if nextNode(t, flow, "verify.check", `{"flag": true}`) != flowNode(t, flow, "book") {
		t.Error("return from verification should lead to book")
	}
	if nextNode(t, flow, "goodbye.anything_else", "") != flowNode(t, flow, "book") {
		t.Error("return from closing should lead to book")
	}
}

func TestReturnToANodeThatRunsASubflow(t *testing.T) {
	config := composedFlowConfig(t)
	config.Nodes["verify"].ReturnTo = "goodbye"
	flow := makeFlow(t, config)
	if nextNode(t, flow, "verify.check", `{"flag": true}`) != flowNode(t, flow, "goodbye.anything_else") {
		t.Error("return should enter the subflow that return_to runs")
	}
}

func TestFlowInitialNodeMayRunASubflow(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t, func(data dict) { data["initial_node"] = "verify" }))
	if flow.InitialNode() != flowNode(t, flow, "verify.ask") {
		t.Error("the flow should start at the first node of the subflow")
	}
}

func TestOneSubflowRunByTwoNodes(t *testing.T) {
	config := composedFlowConfig(t)
	config.Nodes["verify_again"] = &FlowConfigNode{Flow: "verification", ReturnTo: "start"}
	if err := config.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	flow := makeFlow(t, config)

	if flowNode(t, flow, "verify.ask") == flowNode(t, flow, "verify_again.ask") {
		t.Error("each node that runs a subflow should get its own copy of the subflow's nodes")
	}
	for _, tc := range []struct{ node, args, want string }{
		{"verify.ask", "", "verify.check"},
		{"verify_again.ask", "", "verify_again.check"},
		{"verify.check", `{"flag": true}`, "book"},
		{"verify_again.check", `{"flag": true}`, "start"},
	} {
		if nextNode(t, flow, tc.node, tc.args) != flowNode(t, flow, tc.want) {
			t.Errorf("%s should lead to %s", tc.node, tc.want)
		}
	}
}

func TestSubflowThatNeverReturnsNeedsNoReturnTo(t *testing.T) {
	config := composedFlowConfig(t)
	config.Flows["closing"].Nodes["anything_else"].Functions[0].TransitionTo = FlowConfigTransition{Node: "anything_else"}
	config.Nodes["goodbye"].ReturnTo = ""
	if err := config.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	assertEqual(t, flowNode(t, makeFlow(t, config), "goodbye").Name, "goodbye.anything_else")
}

func TestFlowGlobalFunctionMayEnterASubflow(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t, func(data dict) {
		data["global_functions"] = []any{goTo("reverify", "verify")}
	}))
	_, next, err := flow.GlobalFunctions()[0].Handler(context.Background(), json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if next != flowNode(t, flow, "verify.ask") {
		t.Error("the global function should enter the subflow at its first node")
	}
}

func TestNodeNamedReturnIsAnOrdinaryNodeOutsideASubflow(t *testing.T) {
	config := singleNode(t, goTo("back", "next"))
	named := *config.Nodes["next"]
	config.Nodes["return"] = &named
	config.Nodes["a"].Functions[0].TransitionTo = FlowConfigTransition{Node: "return"}
	flow := makeFlow(t, config)
	if nextNode(t, flow, "a", "") != flowNode(t, flow, "return") {
		t.Error("return should name the node called return")
	}
}

func TestProblemsInASubflowNameTheBuiltNode(t *testing.T) {
	config := composedFlowConfig(t)
	// A map has no order of its own, so the config is given the order a
	// document would have written its nodes in.
	config.nodeOrder = []string{"start", "verify", "book", "goodbye"}
	config.Flows["verification"].Nodes["check"].Functions[0].Name = "no_such_tool"
	config.Nodes["book"].Functions[0].Name = "also_missing"
	config.Nodes["verify_again"] = &FlowConfigNode{Flow: "verification", ReturnTo: "start"}
	reference := flowProblems(t, config)

	type found struct{ code, node, function string }
	problems := make([]found, len(reference.Problems))
	for i, p := range reference.Problems {
		problems[i] = found{p.Code, p.Node, p.Function}
	}
	assertEqual(t, problems, []found{
		{ProblemMissingTool, "verify.check", "no_such_tool"},
		{ProblemMissingTool, "book", "also_missing"},
		{ProblemMissingTool, "verify_again.check", "no_such_tool"},
	})
	assertSays(t, reference, "node 'verify.check' references tool 'no_such_tool'")
}

// Problems come in the order the document wrote its nodes in.
func TestProblemsFollowDocumentOrder(t *testing.T) {
	config, err := FlowConfigFromYAML(`
initial_node: start
nodes:
  start:
    task_messages: []
    functions: [{name: zeta}]
  middle:
    task_messages: []
    functions: [{name: alpha}]
  end:
    task_messages: []
    functions: [{name: mu}]
`, "")
	if err != nil {
		t.Fatalf("FlowConfigFromYAML: %v", err)
	}
	reference := flowProblems(t, config, Handlers{})
	functions := make([]string, len(reference.Problems))
	for i, p := range reference.Problems {
		functions[i] = p.Function
	}
	assertEqual(t, functions, []string{"zeta", "alpha", "mu"})
}

// A configured flow driven through the real FlowManager.

// flowManager is a manager for flow, with the state the config's prompts refer
// to as {{ restaurant }} and {{ caller }}.
func flowManager(t *testing.T, flow *Flow) (*FlowManager, *fakeEnq) {
	t.Helper()
	fm, enq := newManager(t, func(c *Config) { c.GlobalFunctions = flow.GlobalFunctions() })
	fm.State().Set("restaurant", "Luigi's")
	fm.State().Set("caller", "friend")
	return fm, enq
}

// invokeWith calls an advertised tool with arguments the way the LLM service
// would, then lets the context-updated callback fire so a pending transition
// executes. It returns the result the tool reported.
func invokeWith(t *testing.T, enq *fakeEnq, name, args string) string {
	t.Helper()
	if args == "" {
		args = "{}"
	}
	handler := advertisedHandler(t, enq.frames(), name)
	var (
		result string
		props  *frames.FunctionCallResultProperties
	)
	params := llm.FunctionCallParams{
		FunctionName: name,
		ToolCallID:   "call-1",
		Arguments:    json.RawMessage(args),
		Result: func(_ context.Context, r string, p *frames.FunctionCallResultProperties) error {
			result, props = r, p
			return nil
		},
	}
	if err := handler(context.Background(), params); err != nil {
		t.Fatalf("calling %s: %v", name, err)
	}
	if props != nil && props.OnContextUpdated != nil {
		if err := props.OnContextUpdated(context.Background()); err != nil {
			t.Fatalf("OnContextUpdated after %s: %v", name, err)
		}
	}
	return result
}

func advertisedNames(enq *fakeEnq) []string {
	var names []string
	for _, tool := range advertisedTools(enq.frames()) {
		names = append(names, tool.Name)
	}
	return names
}

func TestConfiguredFlowEndToEnd(t *testing.T) {
	box := newToolbox()
	flow := makeFlow(t, flowConfig(t), box.handlers)
	fm, enq := flowManager(t, flow)
	mustInitialize(t, fm, flow.InitialNode())

	assertEqual(t, fm.CurrentNode(), "initial")
	assertEqual(t, len(box.kitchenChecks), 1)
	assertEqual(t, advertisedNames(enq), []string{"get_delivery_estimate", "choose_pizza"})

	assertEqual(t, invokeWith(t, enq, "choose_pizza", ""), acknowledged)
	assertEqual(t, fm.CurrentNode(), "pizza")
	assertEqual(t, advertisedNames(enq), []string{"get_delivery_estimate", "select_pizza_order"})

	result := invokeWith(t, enq, "select_pizza_order", `{"size": "small", "pizza_type": "cheese"}`)
	if !strings.Contains(result, `"status":"ok"`) {
		t.Errorf("result %s does not report the order", result)
	}
	order, _ := fm.State().Get("order")
	assertEqual(t, order, map[string]any{"size": "small", "pizza_type": "cheese"})
	assertEqual(t, fm.CurrentNode(), "confirm")

	assertEqual(t, invokeWith(t, enq, "get_delivery_estimate", ""), `{"time":"30 minutes"}`)
	assertEqual(t, fm.CurrentNode(), "confirm")
}

func TestTransitionOnlyFunctionThroughTheManager(t *testing.T) {
	config := flowConfig(t)
	confirm := config.Nodes["confirm"]
	confirm.Functions = append(confirm.Functions, FlowConfigFunction{
		Name:           "start_over",
		TransitionOnly: true,
		Description:    "Start the order over.",
		TransitionTo:   FlowConfigTransition{Node: "initial"},
	})
	flow := makeFlow(t, config)
	fm, enq := flowManager(t, flow)
	mustInitialize(t, fm, flowNode(t, flow, "confirm"))

	assertEqual(t, invokeWith(t, enq, "start_over", ""), acknowledged)
	assertEqual(t, fm.CurrentNode(), "initial")
}

func TestConfiguredMetadataStaysOutOfTheLLMContext(t *testing.T) {
	config := flowConfig(t)
	config.Nodes["initial"].Metadata = map[string]any{"graph_node": "secret-node-id"}
	config.Nodes["initial"].Functions[0].Metadata = map[string]any{"exit": "secret-exit-id"}
	flow := makeFlow(t, config)
	fm, enq := flowManager(t, flow)
	mustInitialize(t, fm, flow.InitialNode())

	assertEqual(t, fm.CurrentNode(), "initial")
	for _, f := range enq.frames() {
		switch frame := f.(type) {
		case *frames.LLMMessagesAppendFrame:
			if text := fmt.Sprintf("%+v", frame.Messages); strings.Contains(text, "secret-") {
				t.Errorf("metadata reached the conversation: %s", text)
			}
		case *frames.LLMSetToolsFrame:
			for _, tool := range frame.Tools {
				if text := tool.Name + tool.Description + string(tool.Parameters); strings.Contains(text, "secret-") {
					t.Errorf("metadata reached the schema of %s: %s", tool.Name, text)
				}
			}
		}
	}
}

func TestNodeEnteredThroughASubflow(t *testing.T) {
	flow := makeFlow(t, composedFlowConfig(t))
	fm, enq := flowManager(t, flow)
	var entered []NodeEntered
	event.On(fm.Events(), EventNodeEntered, func(_ context.Context, e NodeEntered) {
		entered = append(entered, e)
	})

	mustInitialize(t, fm, flow.InitialNode())
	invokeWith(t, enq, "begin", "")
	invokeWith(t, enq, "answered", "")
	invokeWith(t, enq, "report_flag", `{"flag": false}`)
	invokeWith(t, enq, "answered", "")
	invokeWith(t, enq, "report_flag", `{"flag": true}`)
	invokeWith(t, enq, "report", `{"status": "booked"}`)
	invokeWith(t, enq, "more", "")

	type move struct{ previous, function, node, parent string }
	moves := make([]move, len(entered))
	for i, e := range entered {
		moves[i] = move{e.PreviousNode, e.Function, e.Node, e.Parent}
	}
	assertEqual(t, moves, []move{
		{"", "", "start", ""},
		{"start", "begin", "verify.ask", "verify"},
		{"verify.ask", "answered", "verify.check", "verify"},
		{"verify.check", "report_flag", "verify.ask", "verify"},
		{"verify.ask", "answered", "verify.check", "verify"},
		{"verify.check", "report_flag", "book", ""},
		{"book", "report", "goodbye.anything_else", "goodbye"},
		{"goodbye.anything_else", "more", "book", ""},
	})
	assertEqual(t, entered[1].Metadata, map[string]any{"step": "ask"})
	assertEqual(t, fm.CurrentNode(), "book")
}

func TestConfiguredPromptsRenderFromStateOnEntry(t *testing.T) {
	flow := makeFlow(t, flowConfig(t))
	fm, enq := flowManager(t, flow)
	fm.State().Set("caller", "Ada")
	mustInitialize(t, fm, flow.InitialNode())

	found := false
	for _, text := range queuedMessageTexts(enq) {
		found = found || text == "Greet the Ada."
	}
	if !found {
		t.Errorf("the rendered prompt was not queued: %v", queuedMessageTexts(enq))
	}
	// The flow's own node is untouched; rendering happens on a copy.
	assertEqual(t, flow.InitialNode().TaskMessages[0].Text, "Greet the {{ caller }}.")
}

func TestConfiguredMissingStateKeyFailsOnEntry(t *testing.T) {
	flow := makeFlow(t, flowConfig(t))
	fm, _ := flowManager(t, flow)
	fm.State().Delete("caller")
	err := fm.Initialize(context.Background(), flow.InitialNode())
	if !errors.Is(err, ErrFlow) {
		t.Fatalf("got %v, want an ErrFlow", err)
	}
	assertSays(t, err, "uses '{{ caller }}', which is not in state")
}

func TestContractViolationReachesTheLLMAsAnError(t *testing.T) {
	flow := makeFlow(t, singleNode(t, dict{"name": "returns_none"}))
	fm, enq := flowManager(t, flow)
	mustInitialize(t, fm, flow.InitialNode())

	var result struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	raw := invokeWith(t, enq, "returns_none", "")
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("the result %s is not JSON: %v", raw, err)
	}
	assertEqual(t, result.Status, "error")
	if !strings.Contains(result.Error, "returned nil as the next node") {
		t.Errorf("the error %q does not say what the tool did", result.Error)
	}
	assertEqual(t, fm.CurrentNode(), "a")
}
