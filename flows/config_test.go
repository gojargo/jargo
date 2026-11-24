package flows

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/utils/yamlutil"
	"gopkg.in/yaml.v3"
)

// Tests for loading and validating a FlowConfig.

const foodOrderingYAML = `
initial_node: initial

nodes:
  initial:
    role_message: You are an order-taking assistant for {{ restaurant_name }}.
    task_messages:
      - role: developer
        content: Greet the caller and ask whether they want pizza or sushi.
    pre_actions:
      - type: function
        handler: check_kitchen_status
    functions:
      - name: choose_pizza
        transition_to: choose_pizza
      - name: choose_sushi
        transition_to: choose_sushi

  choose_pizza:
    task_messages:
      - role: developer
        content: Take a pizza order.
    functions:
      - name: select_pizza_order
        transition_to:
          field: status
          cases:
            ok: confirm
            unavailable: choose_pizza
          default: confirm

  choose_sushi:
    task_messages:
      - role: developer
        content: Take a sushi order.
    context_strategy: reset
    respond_immediately: false
    functions:
      - name: select_sushi_order
        transition_to: confirm

  confirm:
    task_messages:
      - role: developer
        content: Read the order back.
    functions:
      - name: complete_order
        transition_to: end
      - name: revise_order
        transition_to: initial

  end:
    task_messages:
      - role: developer
        content: Thank the caller.
    post_actions:
      - type: tts_say
        text: Goodbye!
      - type: end_conversation

global_functions:
  - name: get_delivery_estimate
`

type dict = map[string]any

// minimalConfig is the smallest valid config document: one node, "a".
func minimalConfig() dict {
	return dict{
		"initial_node": "a",
		"nodes": dict{
			"a": dict{"task_messages": []any{dict{"role": "developer", "content": "hi"}}},
		},
	}
}

// promptNode is a node document with one task message and the functions given.
func promptNode(functions ...dict) dict {
	n := dict{"task_messages": []any{dict{"role": "developer", "content": "hi"}}}
	if len(functions) > 0 {
		list := make([]any, len(functions))
		for i, f := range functions {
			list[i] = f
		}
		n["functions"] = list
	}
	return n
}

// goTo is a transition_only function document that leads to target.
func goTo(name, target string) dict {
	return dict{"name": name, "transition_only": true, "description": "Go.", "transition_to": target}
}

// composedConfig is a flow whose verify node runs a verification subflow and
// comes back to book.
func composedConfig() dict {
	return dict{
		"initial_node": "start",
		"flows": dict{
			"verification": dict{
				"initial_node": "ask",
				"nodes": dict{
					"ask":     promptNode(dict{"name": "check", "transition_to": "confirm"}),
					"confirm": promptNode(goTo("verified", "return")),
				},
			},
		},
		"nodes": dict{
			"start":  promptNode(goTo("begin", "verify")),
			"verify": dict{"flow": "verification", "return_to": "book"},
			"book":   promptNode(),
		},
	}
}

// dig walks into nested documents by key.
func dig(t *testing.T, data dict, path ...string) dict {
	t.Helper()
	for _, key := range path {
		next, ok := data[key].(dict)
		if !ok {
			t.Fatalf("no mapping at %q of %v", key, path)
		}
		data = next
	}
	return data
}

func mustLoadMap(t *testing.T, data dict) *FlowConfig {
	t.Helper()
	config, err := FlowConfigFromMap(data)
	if err != nil {
		t.Fatalf("FlowConfigFromMap: %v", err)
	}
	return config
}

// assertInvalid checks that data is rejected with a FlowConfigError saying each
// of messages.
func assertInvalid(t *testing.T, data dict, messages ...string) {
	t.Helper()
	_, err := FlowConfigFromMap(data)
	if _, ok := errors.AsType[*FlowConfigError](err); !ok {
		t.Fatalf("got %v, want a FlowConfigError saying %q", err, messages)
	}
	for _, message := range messages {
		if !strings.Contains(err.Error(), message) {
			t.Errorf("error %q does not say %q", err, message)
		}
	}
}

// assertRoundTrip checks that a config written as JSON loads back as itself.
func assertRoundTrip(t *testing.T, config *FlowConfig) {
	t.Helper()
	raw, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("encoding the config: %v", err)
	}
	loaded, err := FlowConfigFromJSON(string(raw))
	if err != nil {
		t.Fatalf("loading the config back from %s: %v", raw, err)
	}
	if !reflect.DeepEqual(loaded, config) {
		t.Errorf("config loaded back from %s differs from the one written", raw)
	}
}

func functionNames(functions []FlowConfigFunction) []string {
	names := make([]string, len(functions))
	for i := range functions {
		names[i] = functions[i].Name
	}
	return names
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func assertEqual(t *testing.T, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

// Loading

func TestConfigLoadsFoodOrderingYAML(t *testing.T) {
	config, err := FlowConfigFromYAML(foodOrderingYAML, "")
	if err != nil {
		t.Fatalf("FlowConfigFromYAML: %v", err)
	}

	assertEqual(t, config.InitialNode, "initial")
	assertEqual(t, sortedKeys(config.Nodes), []string{"choose_pizza", "choose_sushi", "confirm", "end", "initial"})
	assertEqual(t, config.nodeNames(), []string{"initial", "choose_pizza", "choose_sushi", "confirm", "end"})

	initial := config.Nodes["initial"]
	if !strings.Contains(initial.RoleMessage, "{{ restaurant_name }}") {
		t.Errorf("role message %q lost its placeholder", initial.RoleMessage)
	}
	assertEqual(t, functionNames(initial.Functions), []string{"choose_pizza", "choose_sushi"})
	assertEqual(t, initial.Functions[0].TransitionTo, FlowConfigTransition{Node: "choose_pizza"})
	assertEqual(t, initial.PreActions[0].Type, "function")
	assertEqual(t, initial.PreActions[0].Handler, "check_kitchen_status")

	pizza := config.Nodes["choose_pizza"].Functions[0]
	if pizza.TransitionTo.Branch == nil {
		t.Fatal("select_pizza_order should transition on a branch")
	}
	assertEqual(t, pizza.TransitionTo.Branch.Field, "status")
	assertEqual(t, pizza.TransitionTo.Branch.Cases, map[string]string{"ok": "confirm", "unavailable": "choose_pizza"})
	assertEqual(t, pizza.TransitionTo.Branch.Default, "confirm")

	sushi := config.Nodes["choose_sushi"]
	assertEqual(t, sushi.ContextStrategy, ContextStrategyReset)
	if sushi.respondsImmediately() {
		t.Error("choose_sushi should not respond immediately")
	}

	end := config.Nodes["end"]
	assertEqual(t, end.PostActions[0].Type, "tts_say")
	assertEqual(t, end.PostActions[0].Extras, map[string]any{"text": "Goodbye!"})
	assertEqual(t, end.PostActions[1].Type, "end_conversation")

	assertEqual(t, functionNames(config.GlobalFunctions), []string{"get_delivery_estimate"})
	if !config.GlobalFunctions[0].TransitionTo.IsZero() {
		t.Error("get_delivery_estimate should not transition")
	}
}

func TestConfigDefaults(t *testing.T) {
	config := mustLoadMap(t, minimalConfig())
	node := config.Nodes["a"]
	assertEqual(t, node.RoleMessage, "")
	assertEqual(t, len(config.GlobalFunctions), 0)
	assertEqual(t, len(node.Functions), 0)
	assertEqual(t, len(node.PreActions), 0)
	assertEqual(t, len(node.PostActions), 0)
	assertEqual(t, node.ContextStrategy, ContextStrategy(""))
	if !node.respondsImmediately() {
		t.Error("a node responds immediately by default")
	}
}

func TestConfigJSONRoundTripMatchesYAML(t *testing.T) {
	fromYAML, err := FlowConfigFromYAML(foodOrderingYAML, "")
	if err != nil {
		t.Fatalf("FlowConfigFromYAML: %v", err)
	}
	assertRoundTrip(t, fromYAML)

	// json.Unmarshal loads a config the same way.
	raw, err := json.Marshal(fromYAML)
	if err != nil {
		t.Fatal(err)
	}
	var fromJSON FlowConfig
	if err := json.Unmarshal(raw, &fromJSON); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(&fromJSON, fromYAML) {
		t.Error("the config read with json.Unmarshal differs from the YAML one")
	}
}

func writeFiles(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestConfigFromFileYAMLWithInclude(t *testing.T) {
	dir := writeFiles(t, map[string]string{
		"greeting.yaml": "- role: developer\n  content: Greet the caller.\n",
		"flow.yaml":     "initial_node: a\nnodes:\n  a:\n    task_messages: !include greeting.yaml\n",
	})

	config, err := FlowConfigFromFile(filepath.Join(dir, "flow.yaml"))
	if err != nil {
		t.Fatalf("FlowConfigFromFile: %v", err)
	}
	assertEqual(t, config.Nodes["a"].TaskMessages[0].Content, "Greet the caller.")
}

func TestConfigFromFileJSON(t *testing.T) {
	raw, err := json.Marshal(minimalConfig())
	if err != nil {
		t.Fatal(err)
	}
	dir := writeFiles(t, map[string]string{"flow.json": string(raw)})

	config, err := FlowConfigFromFile(filepath.Join(dir, "flow.json"))
	if err != nil {
		t.Fatalf("FlowConfigFromFile: %v", err)
	}
	assertEqual(t, config.InitialNode, "a")
}

func TestConfigFromJSONRequiresAMapping(t *testing.T) {
	_, err := FlowConfigFromJSON(`["just", "a", "list"]`)
	if _, ok := errors.AsType[*FlowConfigError](err); !ok {
		t.Fatalf("got %v, want a FlowConfigError", err)
	}
}

func TestConfigFromYAMLWithoutBaseDirRejectsInclude(t *testing.T) {
	_, err := FlowConfigFromYAML("initial_node: a\nnodes: !include nodes.yaml\n", "")
	if !errors.Is(err, yamlutil.ErrIncludeUnavailable) {
		t.Fatalf("got %v, want ErrIncludeUnavailable", err)
	}
}

func TestConfigTopLevelMustBeAMapping(t *testing.T) {
	_, err := FlowConfigFromYAML("- just\n- a list\n", "")
	if err == nil || !strings.Contains(err.Error(), "top level must be a mapping") {
		t.Fatalf("got %v, want an error saying the top level must be a mapping", err)
	}
}

// The shipped schema

// shippedSchema reads flow_config.schema.json, which the repository keeps for
// editors and other tools to vendor.
func shippedSchema(t *testing.T) dict {
	t.Helper()
	raw, err := os.ReadFile("flow_config.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	var schema dict
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatalf("flow_config.schema.json is not JSON: %v", err)
	}
	return schema
}

func anyStrings(t *testing.T, value any) []string {
	t.Helper()
	list, ok := value.([]any)
	if !ok {
		t.Fatalf("%v is not a list", value)
	}
	out := make([]string, len(list))
	for i, item := range list {
		out[i], _ = item.(string)
	}
	return out
}

func TestShippedSchemaDescribesTheConfig(t *testing.T) {
	schema := shippedSchema(t)
	assertEqual(t, anyStrings(t, schema["required"]), []string{"initial_node", "nodes"})
	if _, has := dig(t, schema, "$defs")["Branch"]; !has {
		t.Error("the schema does not define Branch")
	}
}

func TestShippedSchemaHasOptionalMetadata(t *testing.T) {
	schema := shippedSchema(t)
	assertEqual(t, anyStrings(t, schema["required"]), []string{"initial_node", "nodes"})
	for _, model := range []string{"Node", "Function"} {
		if _, has := dig(t, schema, "$defs", model, "properties")["metadata"]; !has {
			t.Errorf("%s has no metadata property", model)
		}
		if required, has := dig(t, schema, "$defs", model)["required"]; has {
			if slices.Contains(anyStrings(t, required), "metadata") {
				t.Errorf("%s requires metadata", model)
			}
		}
	}
}

func TestShippedSchemaHasOptionalSubflows(t *testing.T) {
	schema := shippedSchema(t)
	// The root stays an inline object schema, not a reference.
	if _, has := schema["$ref"]; has {
		t.Error("the root schema is a reference")
	}
	assertEqual(t, schema["type"], "object")
	assertEqual(t, anyStrings(t, schema["required"]), []string{"initial_node", "nodes"})
	if _, has := dig(t, schema, "properties")["flows"]; !has {
		t.Error("the schema has no flows property")
	}
	assertEqual(t, anyStrings(t, dig(t, schema, "$defs", "Subflow")["required"]), []string{"initial_node", "nodes"})
	node := dig(t, schema, "$defs", "Node")
	for _, key := range []string{"flow", "return_to"} {
		if _, has := dig(t, node, "properties")[key]; !has {
			t.Errorf("Node has no %s property", key)
		}
	}
	// A node needs task_messages unless it runs a flow.
	if _, has := node["required"]; has {
		t.Error("Node requires fields unconditionally")
	}
	assertEqual(t, anyStrings(t, dig(t, node, "else")["required"]), []string{"task_messages"})
}

// jsonFields is the names a struct's fields are written under.
func jsonFields(value any) []string {
	typ := reflect.TypeOf(value)
	var names []string
	for field := range typ.Fields() {
		if tag := field.Tag.Get("json"); tag != "" {
			name, _, _ := strings.Cut(tag, ",")
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return names
}

// The schema is not generated from the Go types, so this is what keeps the two
// from drifting apart: each model takes exactly the properties the schema
// gives it.
func TestShippedSchemaPropertiesAreTheFieldsOfTheConfig(t *testing.T) {
	schema := shippedSchema(t)
	assertEqual(t, sortedKeys(dig(t, schema, "properties")), jsonFields(FlowConfig{}))
	for model, value := range map[string]any{
		"Message":  FlowConfigMessage{},
		"Branch":   FlowConfigBranch{},
		"Function": FlowConfigFunction{},
		"Node":     FlowConfigNode{},
		"Subflow":  FlowConfigSubflow{},
	} {
		assertEqual(t, sortedKeys(dig(t, schema, "$defs", model, "properties")), jsonFields(value))
	}
	assertEqual(t, sortedKeys(dig(t, schema, "$defs", "Action", "properties")), []string{"handler", "type"})
	assertEqual(t, dig(t, schema, "$defs", "Action")["additionalProperties"], true)
}

// Every example config these tests use loads, and a config that uses every
// property the schema has loads too.
func TestExampleConfigsLoad(t *testing.T) {
	if _, err := FlowConfigFromYAML(foodOrderingYAML, ""); err != nil {
		t.Errorf("the food ordering example: %v", err)
	}
	for name, data := range map[string]dict{
		"minimal":  minimalConfig(),
		"composed": composedConfig(),
		"flow":     flowConfigData(),
		"subflows": composedFlowData(),
	} {
		if _, err := FlowConfigFromMap(data); err != nil {
			t.Errorf("the %s example: %v", name, err)
		}
	}

	everything := `
initial_node: a
flows:
  sub:
    initial_node: s
    nodes:
      s:
        task_messages: []
nodes:
  a:
    task_messages:
      - {role: developer, content: hi}
    role_message: You are helpful.
    functions:
      - name: go
        transition_only: true
        description: Go on.
        transition_to: b
        metadata: {exit: go}
      - name: branch
        transition_to: {field: status, cases: {ok: b}, default: a}
    pre_actions:
      - {type: tts_say, text: One moment.}
    post_actions:
      - {type: notify, handler: notify, channel: general}
    context_strategy: append
    respond_immediately: true
    metadata: {step: 1}
  b:
    flow: sub
    return_to:
    metadata: {step: 2}
global_functions:
  - name: help
`
	if _, err := FlowConfigFromYAML(everything, ""); err != nil {
		t.Errorf("the config that uses every property: %v", err)
	}
}

// Validation

func TestConfigInitialNodeMustExist(t *testing.T) {
	data := minimalConfig()
	data["initial_node"] = "missing"
	assertInvalid(t, data, "initial_node 'missing'")
}

func TestConfigUnknownTopLevelKey(t *testing.T) {
	data := minimalConfig()
	data["extra_key"] = 1
	assertInvalid(t, data, "extra_key")
}

func TestConfigNodesRequiredAndNonEmpty(t *testing.T) {
	assertInvalid(t, dict{"initial_node": "a", "nodes": dict{}}, "nodes")
	assertInvalid(t, dict{"initial_node": "a"}, "nodes")
}

func TestConfigTaskMessagesRequired(t *testing.T) {
	assertInvalid(t, dict{"initial_node": "a", "nodes": dict{"a": dict{}}}, "task_messages")
}

func TestConfigTransitionToUnknownNode(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "go", "transition_to": "nowhere"}}
	assertInvalid(t, data, "node 'a' function 'go' transitions to unknown node 'nowhere'")
}

func TestConfigBranchCaseToUnknownNode(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{
		dict{"name": "go", "transition_to": dict{"field": "s", "cases": dict{"x": "nowhere"}}},
	}
	assertInvalid(t, data, "unknown node 'nowhere'")
}

func TestConfigBranchDefaultToUnknownNode(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{
		dict{"name": "go", "transition_to": dict{"field": "s", "cases": dict{"x": "a"}, "default": "zz"}},
	}
	assertInvalid(t, data, "unknown node 'zz'")
}

func TestConfigBranchKeysAreCanonicalStrings(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{
		dict{"name": "go", "transition_to": dict{
			"field": "s", "cases": map[any]any{true: "a", 3: "a", "x": "a"},
		}},
	}
	cases := mustLoadMap(t, data).Nodes["a"].Functions[0].TransitionTo.Branch.Cases
	assertEqual(t, cases, map[string]string{"true": "a", "3": "a", "x": "a"})
}

func TestConfigYAMLBooleanAndNumberKeys(t *testing.T) {
	config, err := FlowConfigFromYAML(
		"initial_node: a\nnodes:\n  a:\n    task_messages: [{role: developer, content: a}]\n"+
			"    functions:\n      - name: go\n        transition_to:\n          field: ok\n"+
			"          cases:\n            true: a\n            0: a\n", "")
	if err != nil {
		t.Fatalf("FlowConfigFromYAML: %v", err)
	}
	assertEqual(t, config.Nodes["a"].Functions[0].TransitionTo.Branch.Cases, map[string]string{"true": "a", "0": "a"})
}

func TestConfigBranchRequiresCases(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{
		dict{"name": "go", "transition_to": dict{"field": "s", "cases": dict{}}},
	}
	assertInvalid(t, data, "cases")
}

func TestConfigGlobalFunctionTransitionToUnknownNode(t *testing.T) {
	data := minimalConfig()
	data["global_functions"] = []any{dict{"name": "go", "transition_to": "nowhere"}}
	assertInvalid(t, data, "global_functions function 'go' transitions to unknown node")
}

func TestConfigDuplicateFunctionInNode(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f"}, dict{"name": "f"}}
	assertInvalid(t, data, "duplicate function 'f' in node")
}

func TestConfigDuplicateGlobalFunction(t *testing.T) {
	data := minimalConfig()
	data["global_functions"] = []any{dict{"name": "f"}, dict{"name": "f"}}
	assertInvalid(t, data, "duplicate function 'f' in global_functions")
}

func TestConfigNodeFunctionCollidesWithGlobal(t *testing.T) {
	data := minimalConfig()
	data["global_functions"] = []any{dict{"name": "f"}}
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f"}}
	assertInvalid(t, data, "node 'a' function 'f' is also a global function")
}

func TestConfigFunctionRejectsUnknownKeys(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "parameters": dict{}}}
	assertInvalid(t, data, "parameters")
}

func TestConfigDescriptionOnlyOnTransitionOnlyFunction(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "description": "d", "transition_to": "a"}}
	assertInvalid(t, data, "only a transition_only function takes")

	// An empty description is still one.
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "description": "", "transition_to": "a"}}
	assertInvalid(t, data, "only a transition_only function takes")
}

func TestConfigTransitionOnlyFunctionNeedsDescription(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "transition_only": true, "transition_to": "a"}}
	assertInvalid(t, data, "needs a description")
}

func TestConfigTransitionOnlyFunctionNeedsANodeName(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "transition_only": true, "description": "d"}}
	assertInvalid(t, data, "must name the node it transitions to")

	dig(t, data, "nodes", "a")["functions"] = []any{dict{
		"name": "f", "transition_only": true, "description": "d",
		"transition_to": dict{"field": "x", "cases": dict{"y": "a"}},
	}}
	assertInvalid(t, data, "must name the node it transitions to")
}

func TestConfigMetadataOnNodeAndFunction(t *testing.T) {
	data := minimalConfig()
	node := dig(t, data, "nodes", "a")
	metadata := dict{"graph_node": "n1", "position": dict{"x": 1, "y": 2.5}, "tags": []any{"a"}}
	node["metadata"] = metadata
	node["functions"] = []any{dict{"name": "f", "metadata": dict{"exit": "chosen"}}}

	config := mustLoadMap(t, data)
	assertEqual(t, config.Nodes["a"].Metadata, metadata)
	assertEqual(t, config.Nodes["a"].Functions[0].Metadata, map[string]any{"exit": "chosen"})
	assertRoundTrip(t, config)
}

func TestConfigMetadataDefaultsToNone(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f"}}
	config := mustLoadMap(t, data)
	if config.Nodes["a"].Metadata != nil || config.Nodes["a"].Functions[0].Metadata != nil {
		t.Error("metadata should be nil when the document has none")
	}
}

func TestConfigMetadataMustBeJSON(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["metadata"] = dict{"handle": make(chan int)}
	assertInvalid(t, data, "metadata")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "metadata": dict{"handle": make(chan int)}}}
	assertInvalid(t, data, "metadata")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["metadata"] = []any{"not", "a", "mapping"}
	assertInvalid(t, data, "metadata")

	// A config built in code is held to the same.
	config := mustLoadMap(t, minimalConfig())
	config.Nodes["a"].Metadata = map[string]any{"handle": make(chan int)}
	if err := config.Validate(); err == nil || !strings.Contains(err.Error(), "metadata") {
		t.Errorf("Validate: got %v, want an error about metadata", err)
	}
}

func TestConfigTransitionOnlyFunction(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{
		dict{"name": "f", "transition_only": true, "description": "d", "transition_to": "a"},
	}
	function := mustLoadMap(t, data).Nodes["a"].Functions[0]
	if !function.TransitionOnly {
		t.Error("the function should be transition_only")
	}
	assertEqual(t, function.Targets(), []string{"a"})
}

func TestConfigFunctionActionRequiresHandler(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["pre_actions"] = []any{dict{"type": "function"}}
	assertInvalid(t, data, "a 'function' action requires a 'handler' name")
}

func TestConfigBuiltInActionsRejectHandler(t *testing.T) {
	for _, builtIn := range []string{"tts_say", "end_conversation"} {
		data := minimalConfig()
		dig(t, data, "nodes", "a")["post_actions"] = []any{dict{"type": builtIn, "handler": "x"}}
		assertInvalid(t, data, "the built-in '"+builtIn+"' action does not take a 'handler'")
	}
}

func TestConfigCustomActionMayNameAHandler(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["post_actions"] = []any{
		dict{"type": "notify", "handler": "notify_team", "channel": "#x"},
		dict{"type": "audit"},
	}
	actions := mustLoadMap(t, data).Nodes["a"].PostActions
	assertEqual(t, actions[0].Handler, "notify_team")
	if actions[0].RegisteredInCode() {
		t.Error("an action that names its handler is not registered in code")
	}
	if !actions[1].RegisteredInCode() {
		t.Error("a custom action without a handler is registered in code")
	}
	assertEqual(t, actions[0].Extras, map[string]any{"channel": "#x"})
}

func TestConfigContextStrategyValues(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes", "a")["context_strategy"] = "reset_with_summary"
	assertInvalid(t, data, "context_strategy")
}

// Subflows

func TestSubflowComposedConfig(t *testing.T) {
	config := mustLoadMap(t, composedConfig())
	assertEqual(t, config.Nodes["verify"].Flow, "verification")
	assertEqual(t, config.Nodes["verify"].ReturnTo, "book")
	assertEqual(t, len(config.Nodes["verify"].TaskMessages), 0)
	assertEqual(t, sortedKeys(config.Flows["verification"].Nodes), []string{"ask", "confirm"})
	assertRoundTrip(t, config)
}

func TestConfigDefaultsToNoSubflows(t *testing.T) {
	config := mustLoadMap(t, minimalConfig())
	assertEqual(t, len(config.Flows), 0)
	assertEqual(t, config.Nodes["a"].Flow, "")
	assertEqual(t, config.Nodes["a"].ReturnTo, "")
}

func TestSubflowFromAnIncludedFile(t *testing.T) {
	subflow, err := yaml.Marshal(dig(t, composedConfig(), "flows", "verification"))
	if err != nil {
		t.Fatal(err)
	}
	data := composedConfig()
	delete(data, "flows")
	rest, err := yaml.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	dir := writeFiles(t, map[string]string{
		"verification.yaml": string(subflow),
		"flow.yaml":         string(rest) + "flows:\n  verification: !include verification.yaml\n",
	})

	config, err := FlowConfigFromFile(filepath.Join(dir, "flow.yaml"))
	if err != nil {
		t.Fatalf("FlowConfigFromFile: %v", err)
	}
	// The two are written in a different order, which is not a difference.
	want := mustLoadMap(t, composedConfig())
	config.nodeOrder, want.nodeOrder = nil, nil
	if !reflect.DeepEqual(config, want) {
		t.Error("the config with an included subflow differs from the one written in full")
	}
}

// A node prompts the LLM or runs a flow, not both.

func TestNodeThatRunsAFlowTakesNothingElse(t *testing.T) {
	extras := dict{
		"task_messages":       []any{dict{"role": "developer", "content": "hi"}},
		"role_message":        "You are helpful.",
		"functions":           []any{goTo("skip", "book")},
		"pre_actions":         []any{dict{"type": "tts_say", "text": "One moment."}},
		"post_actions":        []any{dict{"type": "end_conversation"}},
		"context_strategy":    "reset",
		"respond_immediately": false,
	}
	for key, value := range extras {
		data := composedConfig()
		dig(t, data, "nodes", "verify")[key] = value
		assertInvalid(t, data, "a node that runs flow 'verification' takes only 'flow', 'return_to'")
	}
}

func TestNodeThatRunsAFlowMayHaveMetadata(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes", "verify")["metadata"] = dict{"graph_node": "n7"}
	assertEqual(t, mustLoadMap(t, data).Nodes["verify"].Metadata, map[string]any{"graph_node": "n7"})
}

func TestReturnToNeedsAFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes", "book")["return_to"] = "start"
	assertInvalid(t, data, "'return_to' applies only to a node that runs a 'flow'")
}

func TestNodeNeedsTaskMessagesOrAFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["book"] = dict{}
	assertInvalid(t, data, "a node needs 'task_messages', or a 'flow' to run")
}

// A node runs a flow the config has.

func TestNodeRunsUnknownFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes", "verify")["flow"] = "nope"
	assertInvalid(t, data, "node 'verify' runs unknown flow 'nope'")

	plain := minimalConfig()
	dig(t, plain, "nodes")["b"] = dict{"flow": "verification"}
	assertInvalid(t, plain, "node 'b' runs unknown flow 'verification'")
}

// return_to names a node, and is needed only when the flow returns.

func TestReturnToUnknownNode(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes", "verify")["return_to"] = "nope"
	assertInvalid(t, data, "node 'verify' returns to unknown node 'nope'")
}

func TestReturnToANodeThatRunsAFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["verify_again"] = dict{"flow": "verification", "return_to": "book"}
	dig(t, data, "nodes", "verify")["return_to"] = "verify_again"
	mustLoadMap(t, data)
}

func TestNodesMayReturnToEachOther(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["verify_again"] = dict{"flow": "verification", "return_to": "verify"}
	dig(t, data, "nodes", "verify")["return_to"] = "verify_again"
	mustLoadMap(t, data)
}

func TestReturnToIsNeededWhenTheFlowReturns(t *testing.T) {
	data := composedConfig()
	delete(dig(t, data, "nodes", "verify"), "return_to")
	assertInvalid(t, data, "node 'verify' runs flow 'verification', which returns, and needs a 'return_to'")
}

func TestReturnToMayBeOmittedWhenTheFlowNeverReturns(t *testing.T) {
	data := composedConfig()
	dig(t, data, "flows")["closing"] = dict{
		"initial_node": "bye",
		"nodes":        dict{"bye": promptNode(goTo("again", "bye"))},
	}
	dig(t, data, "nodes")["goodbye"] = dict{"flow": "closing"}
	assertEqual(t, mustLoadMap(t, data).Nodes["goodbye"].ReturnTo, "")

	dig(t, data, "nodes", "goodbye")["return_to"] = "start"
	mustLoadMap(t, data)
}

// A node that runs a flow is a node: a target, or where the flow starts.

func TestInitialNodeMayRunAFlow(t *testing.T) {
	data := composedConfig()
	data["initial_node"] = "verify"
	assertEqual(t, mustLoadMap(t, data).InitialNode, "verify")
}

func TestInitialNodeCannotNameAFlow(t *testing.T) {
	data := composedConfig()
	data["initial_node"] = "verification"
	assertInvalid(t, data, "initial_node 'verification' is not a defined node")
}

func TestGlobalFunctionMayEnterAFlow(t *testing.T) {
	data := composedConfig()
	data["global_functions"] = []any{goTo("reverify", "verify")}
	mustLoadMap(t, data)
}

func TestBranchMayEnterAFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["book"] = promptNode(dict{
		"name": "book_slot",
		"transition_to": dict{
			"field":   "booked",
			"cases":   dict{"true": "verify"},
			"default": "book",
		},
	})
	mustLoadMap(t, data)
}

func TestTargetCannotNameAFlowOrOneOfItsNodes(t *testing.T) {
	for _, target := range []string{"verification", "ask", "verify.ask"} {
		data := composedConfig()
		dig(t, data, "nodes")["book"] = promptNode(goTo("again", target))
		assertInvalid(t, data, "transitions to unknown node '"+target+"'")
	}
}

func TestReturnIsAnOrdinaryNameOutsideASubflow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["book"] = promptNode(goTo("back", "return"))
	assertInvalid(t, data, "node 'book' function 'back' transitions to unknown node 'return'")

	dig(t, data, "nodes")["return"] = promptNode()
	mustLoadMap(t, data)

	plain := minimalConfig()
	dig(t, plain, "nodes", "a")["functions"] = []any{goTo("back", "return")}
	assertInvalid(t, plain, "transitions to unknown node 'return'")
	dig(t, plain, "nodes")["return"] = promptNode()
	mustLoadMap(t, plain)
}

// A target in a subflow names one of its nodes or is return.

func TestSubflowTargetMustBeItsOwnNodeOrReturn(t *testing.T) {
	for _, target := range []string{"book", "verify", "nope"} {
		data := composedConfig()
		dig(t, data, "flows", "verification", "nodes")["confirm"] = promptNode(goTo("verified", target))
		assertInvalid(t, data, "flows.verification",
			"node 'confirm' function 'verified' transitions to unknown node '"+target+"'")
	}
}

func TestSubflowBranchTargets(t *testing.T) {
	data := composedConfig()
	branch := dict{"field": "ok", "cases": dict{"true": "return", "false": "ask"}}
	dig(t, data, "flows", "verification", "nodes")["confirm"] = promptNode(
		dict{"name": "check_again", "transition_to": branch})
	mustLoadMap(t, data)

	branch["default"] = "book"
	assertInvalid(t, data, "transitions to unknown node 'book'")
}

func TestSubflowCannotHaveANodeNamedReturn(t *testing.T) {
	data := composedConfig()
	dig(t, data, "flows", "verification", "nodes")["return"] = promptNode()
	assertInvalid(t, data, "flows.verification", "cannot have a node named 'return'")
}

func TestSubflowInitialNodeMustExist(t *testing.T) {
	data := composedConfig()
	dig(t, data, "flows", "verification")["initial_node"] = "start"
	assertInvalid(t, data, "flows.verification", "initial_node 'start' is not a defined node")
}

func TestSubflowsDoNotNest(t *testing.T) {
	data := composedConfig()
	dig(t, data, "flows", "verification", "nodes")["inner"] = dict{"flow": "verification"}
	assertInvalid(t, data, "flows.verification", "node 'inner' runs a flow, which a subflow's nodes cannot do")

	for key, value := range (dict{"flows": dict{}, "global_functions": []any{}}) {
		data := composedConfig()
		dig(t, data, "flows", "verification")[key] = value
		assertInvalid(t, data, "flows.verification."+key)
	}
}

func TestSubflowFunctionCannotShareAGlobalFunctionName(t *testing.T) {
	data := composedConfig()
	data["global_functions"] = []any{dict{"name": "check"}}
	assertInvalid(t, data, "flow 'verification' node 'ask' function 'check' is also a global function")
}

// Node names have no dots in a config that has subflows.

func TestNoDotsInNodeNamesWithSubflows(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["book.slot"] = promptNode()
	assertInvalid(t, data, "node 'book.slot' has a '.' in its name")

	data = composedConfig()
	dig(t, data, "nodes")["verify.caller"] = dict{"flow": "verification", "return_to": "book"}
	assertInvalid(t, data, "node 'verify.caller' has a '.' in its name")

	data = composedConfig()
	dig(t, data, "flows", "verification", "nodes")["ask.again"] = promptNode()
	assertInvalid(t, data, "node 'ask.again' has a '.' in its name")
}

func TestDotsAreAllowedWithoutSubflows(t *testing.T) {
	data := minimalConfig()
	dig(t, data, "nodes")["scheduling.find_slot"] = promptNode(goTo("back", "a"))
	dig(t, data, "nodes", "a")["functions"] = []any{goTo("find", "scheduling.find_slot")}
	mustLoadMap(t, data)
}

// Every subflow is run.

func TestUnusedFlow(t *testing.T) {
	data := composedConfig()
	dig(t, data, "flows")["closing"] = dict{"initial_node": "bye", "nodes": dict{"bye": promptNode()}}
	assertInvalid(t, data, "flow 'closing' is not run by any node")

	dig(t, data, "nodes")["goodbye"] = dict{"flow": "closing"}
	mustLoadMap(t, data)
}

func TestOneFlowRunByTwoNodes(t *testing.T) {
	data := composedConfig()
	dig(t, data, "nodes")["verify_again"] = dict{"flow": "verification", "return_to": "start"}
	mustLoadMap(t, data)
}

// What a document may spell differently

func TestCaseKey(t *testing.T) {
	for _, tc := range []struct {
		value any
		want  string
	}{
		{true, "true"},
		{false, "false"},
		{"True", "true"},
		{"FALSE", "false"},
		{"ok", "ok"},
		{7, "7"},
		{json.Number("7"), "7"},
		{json.Number("7.0"), "7.0"},
		{json.Number("1e3"), "1000.0"},
		{0.5, "0.5"},
		{2.0, "2.0"},
		{1e16, "1e+16"},
		{nil, "None"},
	} {
		if got := caseKey(tc.value); got != tc.want {
			t.Errorf("caseKey(%#v) = %q, want %q", tc.value, got, tc.want)
		}
	}
}

func TestConfigRejectsFieldsOfTheWrongType(t *testing.T) {
	data := minimalConfig()
	data["initial_node"] = 3
	assertInvalid(t, data, "initial_node: Input should be a valid string")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["respond_immediately"] = "maybe"
	assertInvalid(t, data, "nodes.a.respond_immediately: Input should be a valid boolean")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = "f"
	assertInvalid(t, data, "nodes.a.functions: Input should be a valid list")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["functions"] = []any{dict{"name": "f", "transition_to": 3}}
	assertInvalid(t, data, "nodes.a.functions.0.transition_to")

	data = minimalConfig()
	dig(t, data, "nodes", "a")["task_messages"] = []any{dict{"role": "developer"}}
	assertInvalid(t, data, "nodes.a.task_messages.0.content: Field required")
}

// Every mistake in a document is reported, not only the first.
func TestConfigReportsEveryFieldError(t *testing.T) {
	data := minimalConfig()
	data["extra_key"] = 1
	dig(t, data, "nodes", "a")["surprise"] = true
	_, err := FlowConfigFromMap(data)
	var invalid *FlowConfigError
	if !errors.As(err, &invalid) {
		t.Fatalf("got %v, want a FlowConfigError", err)
	}
	assertEqual(t, invalid.Details, []FlowConfigErrorDetail{
		{Loc: "extra_key", Msg: "Extra inputs are not permitted"},
		{Loc: "nodes.a.surprise", Msg: "Extra inputs are not permitted"},
	})
	if !strings.Contains(err.Error(), "2 validation errors") {
		t.Errorf("error %q does not count its details", err)
	}
}

// A config remembers the order its nodes were written in, and a node added in
// code follows them.
func TestConfigKeepsDocumentOrder(t *testing.T) {
	config, err := FlowConfigFromJSON(`{"initial_node": "z", "nodes": {
		"z": {"task_messages": []}, "m": {"task_messages": []}, "a": {"task_messages": []}}}`)
	if err != nil {
		t.Fatalf("FlowConfigFromJSON: %v", err)
	}
	assertEqual(t, config.nodeNames(), []string{"z", "m", "a"})

	config.Nodes["b"] = &FlowConfigNode{TaskMessages: []FlowConfigMessage{}}
	assertEqual(t, config.nodeNames(), []string{"z", "m", "a", "b"})
	if err := config.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}
