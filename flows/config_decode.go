package flows

import (
	"encoding/json"
	"slices"
	"strings"
)

// The messages for a field of the wrong shape.
const (
	msgRequired   = "Field required"
	msgExtra      = "Extra inputs are not permitted"
	msgNotString  = "Input should be a valid string"
	msgNotBool    = "Input should be a valid boolean"
	msgNotList    = "Input should be a valid list"
	msgTransition = "Input should be a valid string or a branch mapping"
)

// configDecoder turns a parsed document into a FlowConfig, collecting every
// field that is missing, unknown or of the wrong type. What the fields say
// about each other is left to FlowConfig.Validate.
type configDecoder struct {
	details []FlowConfigErrorDetail
}

// fail records msg at loc.
func (d *configDecoder) fail(loc, msg string) {
	d.details = append(d.details, FlowConfigErrorDetail{Loc: loc, Msg: msg})
}

// model reads value as a mapping that takes only the fields named. It returns
// nil, having said why, when value is not a mapping with string keys.
func (d *configDecoder) model(loc string, value any, fields ...string) *object {
	obj := d.mapping(loc, value)
	if obj == nil {
		return nil
	}
	for _, key := range obj.keys {
		if name, _ := key.(string); !slices.Contains(fields, name) {
			d.fail(at(loc, name), msgExtra)
		}
	}
	return obj
}

// mapping reads value as a mapping with string keys.
func (d *configDecoder) mapping(loc string, value any) *object {
	obj, ok := value.(*object)
	if !ok {
		d.fail(loc, msgNotMapping)
		return nil
	}
	if !obj.stringKeys() {
		d.fail(loc, msgNotString)
		return nil
	}
	return obj
}

// requiredString reads a field that must be a string.
func (d *configDecoder) requiredString(loc string, obj *object, key string) string {
	value, has := obj.get(key)
	if !has {
		d.fail(at(loc, key), msgRequired)
		return ""
	}
	text, ok := value.(string)
	if !ok {
		d.fail(at(loc, key), msgNotString)
	}
	return text
}

// optionalString reads a field that is a string, null or absent. It reports
// whether a string was given.
func (d *configDecoder) optionalString(loc string, obj *object, key string) (string, bool) {
	value, has := obj.get(key)
	if !has || value == nil {
		return "", false
	}
	text, ok := value.(string)
	if !ok {
		d.fail(at(loc, key), msgNotString)
	}
	return text, ok
}

// boolean reads a field that is a boolean or absent, or nil when it is absent.
// The spellings of one a document may use are accepted: 0 and 1, and "true",
// "yes", "on", "y", "t", "1" with their opposites, in any case.
func (d *configDecoder) boolean(loc string, obj *object, key string) *bool {
	value, has := obj.get(key)
	if !has {
		return nil
	}
	var out bool
	switch v := value.(type) {
	case bool:
		out = v
	case int:
		if v != 0 && v != 1 {
			d.fail(at(loc, key), msgNotBool)
		}
		out = v == 1
	case float64:
		if v != 0 && v != 1 {
			d.fail(at(loc, key), msgNotBool)
		}
		out = v == 1
	case string:
		switch strings.ToLower(v) {
		case "1", "on", "t", "true", "y", "yes":
			out = true
		case "0", "off", "f", "false", "n", "no":
		default:
			d.fail(at(loc, key), msgNotBool)
		}
	default:
		d.fail(at(loc, key), msgNotBool)
	}
	return &out
}

// list reads a field that is a list or absent. It reports whether the field was
// there.
func (d *configDecoder) list(loc string, obj *object, key string) ([]any, bool) {
	value, has := obj.get(key)
	if !has {
		return nil, false
	}
	items, ok := value.([]any)
	if !ok {
		d.fail(at(loc, key), msgNotList)
		return nil, true
	}
	return items, true
}

// metadata reads a metadata field: a mapping of JSON values, null or absent.
func (d *configDecoder) metadata(loc string, obj *object) map[string]any {
	value, has := obj.get(locMetadata)
	if !has || value == nil {
		return nil
	}
	loc = at(loc, locMetadata)
	mapping := d.mapping(loc, value)
	if mapping == nil {
		return nil
	}
	out, _ := plain(mapping).(map[string]any)
	if _, err := json.Marshal(out); err != nil {
		d.fail(loc, msgNotJSON)
		return nil
	}
	return out
}

func (d *configDecoder) config(value any) *FlowConfig {
	config := &FlowConfig{}
	obj := d.model("", value, locInitialNode, locNodes, locGlobals, locFlows)
	if obj == nil {
		return config
	}
	config.InitialNode = d.requiredString("", obj, locInitialNode)
	config.Nodes, config.nodeOrder = d.nodes("", obj)
	config.GlobalFunctions = d.functions("", obj, locGlobals)

	if flows, has := obj.get(locFlows); has {
		if mapping := d.mapping(locFlows, flows); mapping != nil {
			config.Flows = make(map[string]*FlowConfigSubflow, len(mapping.keys))
			for i, key := range mapping.keys {
				id, _ := key.(string)
				config.Flows[id] = d.subflow(at(locFlows, id), mapping.values[i])
				config.flowOrder = append(config.flowOrder, id)
			}
		}
	}
	return config
}

func (d *configDecoder) subflow(loc string, value any) *FlowConfigSubflow {
	subflow := &FlowConfigSubflow{}
	obj := d.model(loc, value, locInitialNode, locNodes)
	if obj == nil {
		return subflow
	}
	subflow.InitialNode = d.requiredString(loc, obj, locInitialNode)
	subflow.Nodes, subflow.nodeOrder = d.nodes(loc, obj)
	return subflow
}

// nodes reads the required nodes field of a config or a subflow, and the order
// the nodes were written in.
func (d *configDecoder) nodes(loc string, obj *object) (map[string]*FlowConfigNode, []string) {
	value, has := obj.get(locNodes)
	if !has {
		d.fail(at(loc, locNodes), msgRequired)
		return nil, nil
	}
	loc = at(loc, locNodes)
	mapping := d.mapping(loc, value)
	if mapping == nil {
		return nil, nil
	}
	nodes := make(map[string]*FlowConfigNode, len(mapping.keys))
	order := make([]string, 0, len(mapping.keys))
	for i, key := range mapping.keys {
		name, _ := key.(string)
		nodes[name] = d.node(at(loc, name), mapping.values[i])
		order = append(order, name)
	}
	return nodes, order
}

func (d *configDecoder) node(loc string, value any) *FlowConfigNode {
	node := &FlowConfigNode{}
	obj := d.model(loc, value, locTaskMessages, locRoleMessage, locFunctions, locPreActions,
		locPostActions, locStrategy, locRespond, locMetadata, locFlow, locReturnTo)
	if obj == nil {
		return node
	}
	if items, has := d.list(loc, obj, locTaskMessages); has {
		node.TaskMessages = make([]FlowConfigMessage, 0, len(items))
		for i, item := range items {
			node.TaskMessages = append(node.TaskMessages, d.message(at(loc, locTaskMessages, i), item))
		}
	}
	node.RoleMessage, _ = d.optionalString(loc, obj, locRoleMessage)
	node.Functions = d.functions(loc, obj, locFunctions)
	node.PreActions = d.actions(loc, obj, locPreActions)
	node.PostActions = d.actions(loc, obj, locPostActions)
	strategy, _ := d.optionalString(loc, obj, locStrategy)
	node.ContextStrategy = ContextStrategy(strategy)
	node.RespondImmediately = d.boolean(loc, obj, locRespond)
	node.Metadata = d.metadata(loc, obj)
	node.Flow, _ = d.optionalString(loc, obj, locFlow)
	node.ReturnTo, _ = d.optionalString(loc, obj, locReturnTo)
	return node
}

func (d *configDecoder) message(loc string, value any) FlowConfigMessage {
	obj := d.model(loc, value, "role", "content")
	if obj == nil {
		return FlowConfigMessage{}
	}
	return FlowConfigMessage{
		Role:    d.requiredString(loc, obj, "role"),
		Content: d.requiredString(loc, obj, "content"),
	}
}

// functions reads a list of functions, or nil when the field is absent or
// empty.
func (d *configDecoder) functions(loc string, obj *object, key string) []FlowConfigFunction {
	items, _ := d.list(loc, obj, key)
	if len(items) == 0 {
		return nil
	}
	functions := make([]FlowConfigFunction, 0, len(items))
	for i, item := range items {
		functions = append(functions, d.function(at(loc, key, i), item))
	}
	return functions
}

func (d *configDecoder) function(loc string, value any) FlowConfigFunction {
	obj := d.model(loc, value, locName, locTransitionOnly, locDescription, locTransitionTo, locMetadata)
	if obj == nil {
		return FlowConfigFunction{}
	}
	function := FlowConfigFunction{
		Name:         d.requiredString(loc, obj, locName),
		TransitionTo: d.transition(at(loc, locTransitionTo), obj),
		Metadata:     d.metadata(loc, obj),
	}
	if only := d.boolean(loc, obj, locTransitionOnly); only != nil {
		function.TransitionOnly = *only
	}
	description, given := d.optionalString(loc, obj, locDescription)
	function.Description = description
	if given && description == "" && !function.TransitionOnly {
		// An empty description is still a description, which the field cannot
		// tell from none once it is read.
		d.fail(loc, descriptionNotTaken(function.Name))
	}
	return function
}

// transition reads a transition_to field: a node name, a branch mapping, null
// or absent.
func (d *configDecoder) transition(loc string, obj *object) FlowConfigTransition {
	value, _ := obj.get(locTransitionTo)
	switch v := value.(type) {
	case nil:
		return FlowConfigTransition{}
	case string:
		return FlowConfigTransition{Node: v}
	case *object:
		return FlowConfigTransition{Branch: d.branch(loc, v)}
	default:
		d.fail(loc, msgTransition)
		return FlowConfigTransition{}
	}
}

// branch reads a branch mapping. The keys of its cases may be strings,
// booleans, or numbers, and are kept as their canonical strings.
func (d *configDecoder) branch(loc string, obj *object) *FlowConfigBranch {
	branch := &FlowConfigBranch{}
	for _, key := range obj.keys {
		if name, ok := key.(string); !ok || !slices.Contains([]string{"field", "cases", "default"}, name) {
			d.fail(at(loc, key), msgExtra)
		}
	}
	branch.Field = d.requiredString(loc, obj, "field")
	branch.Default, _ = d.optionalString(loc, obj, "default")

	value, has := obj.get("cases")
	if !has {
		d.fail(at(loc, "cases"), msgRequired)
		return branch
	}
	cases, ok := value.(*object)
	if !ok {
		d.fail(at(loc, "cases"), msgNotMapping)
		return branch
	}
	branch.Cases = make(map[string]string, len(cases.keys))
	for i, key := range cases.keys {
		target, ok := cases.values[i].(string)
		if !ok {
			d.fail(at(loc, "cases", caseKey(key)), msgNotString)
			continue
		}
		branch.Cases[caseKey(key)] = target
	}
	return branch
}

// actions reads a list of actions, or nil when the field is absent or empty.
func (d *configDecoder) actions(loc string, obj *object, key string) []FlowConfigAction {
	items, _ := d.list(loc, obj, key)
	if len(items) == 0 {
		return nil
	}
	actions := make([]FlowConfigAction, 0, len(items))
	for i, item := range items {
		actions = append(actions, d.action(at(loc, key, i), item))
	}
	return actions
}

// action reads an action. Keys beyond type and handler pass through as they
// are.
func (d *configDecoder) action(loc string, value any) FlowConfigAction {
	obj := d.mapping(loc, value)
	if obj == nil {
		return FlowConfigAction{}
	}
	action := FlowConfigAction{Type: d.requiredString(loc, obj, locType)}
	action.Handler, _ = d.optionalString(loc, obj, locHandler)
	for i, key := range obj.keys {
		name, _ := key.(string)
		if name == locType || name == locHandler {
			continue
		}
		if action.Extras == nil {
			action.Extras = make(map[string]any)
		}
		action.Extras[name] = plain(obj.values[i])
	}
	return action
}
