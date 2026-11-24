package flows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/gojargo/jargo/frames"
)

// Handlers is the code behind a flow config, by the name the config refers to
// it with: a NodeFunction for a tool, an ActionHandler for an action handler.
// Only the names the config references are looked up.
type Handlers map[string]any

// The kinds of thing a config looks up in the handlers.
const (
	kindTool          = "tool"
	kindActionHandler = "action handler"
)

// Flow is a FlowConfig joined to the code it names: the config's nodes turned
// into runnable NodeConfigs, with every tool reference resolved to a
// NodeFunction, every transition_only function built from the config alone,
// and every transition wired to the node the config names.
//
// The flow hands ready-made nodes to a FlowManager:
//
//	config, err := flows.FlowConfigFromFile("flow.yaml")
//	flow, err := flows.NewFlow(config, handlers)
//	fm, err := flows.New(flows.Config{…, GlobalFunctions: flow.GlobalFunctions()})
//	err = fm.Initialize(ctx, flow.InitialNode())
//
// Building a flow validates the references the config could not: each tool
// exists and can be offered, each action handler the config names is one. Tool
// return values are checked at call time against the configured-flow contract:
// a tool returns TransitionInYAML in its next-node slot and the config decides
// the next node. {{ key }} placeholders in prompts are left in place; the
// FlowManager fills them from its state when it enters the node.
//
// A config's subflows are flattened before anything is built: a subflow's nodes
// are copied in once for each node that runs it, under the name
// <node>.<subflow node>, with every transition pointing at the node it leads
// to.
type Flow struct {
	config   *FlowConfig
	handlers []Handlers
	problems []FlowProblem

	// nodes are the built nodes, by the name each is built under.
	nodes map[string]*NodeConfig
	// entries is, for every node name in the config, the name of the built node
	// it leads to.
	entries         map[string]string
	globalFunctions []NodeFunction
}

// NewFlow joins a config to the handlers it refers to.
//
// handlers is the code behind the config: the NodeFunctions that implement its
// tools and the ActionHandlers that implement its actions. Several may be
// given, which lets tools and action handlers live apart; a name found in more
// than one of them is an error rather than a silent choice. The config's name
// for a tool is the tool's identity: it is what the LLM sees whatever the
// NodeFunction's own Name says, so several entries may share one NodeFunction.
//
// It returns a *FlowReferenceError if a referenced tool or handler is missing
// or ambiguous, or a tool cannot be offered. Every such problem is collected
// and reported together.
func NewFlow(config *FlowConfig, handlers ...Handlers) (*Flow, error) {
	f := &Flow{config: config, handlers: handlers}

	flat := flatten(config)
	f.entries = flat.entries
	f.nodes = make(map[string]*NodeConfig, len(flat.nodes))
	for _, name := range flat.names {
		f.nodes[name] = f.buildNode(name, flat.nodes[name])
	}
	for name, parent := range flat.parents {
		f.nodes[name].Parent = parent
	}
	for i := range flat.globalFunctions {
		if fn, ok := f.buildFunction(&flat.globalFunctions[i], locGlobals, ""); ok {
			f.globalFunctions = append(f.globalFunctions, fn)
		}
	}
	// Building continues past each problem so all of them are found; the
	// partially built state never escapes because this return comes last.
	if len(f.problems) > 0 {
		return nil, &FlowReferenceError{Problems: f.problems}
	}
	return f, nil
}

// Config returns the config this flow was built from.
func (f *Flow) Config() *FlowConfig { return f.config }

// InitialNode returns the node the flow starts in.
func (f *Flow) InitialNode() *NodeConfig {
	return f.nodes[f.entries[f.config.InitialNode]]
}

// GlobalFunctions returns the tools available at every node, for
// Config.GlobalFunctions. It is a fresh slice each time, so the caller may
// extend it.
func (f *Flow) GlobalFunctions() []NodeFunction {
	return append([]NodeFunction(nil), f.globalFunctions...)
}

// Node returns the node for name: a node name from the config, or
// <node>.<subflow node> for a node of a subflow. For a node that runs a
// subflow, this is the subflow's first node. It returns an ErrFlow if the
// config has no such node.
func (f *Flow) Node(name string) (*NodeConfig, error) {
	built := name
	if entry, has := f.entries[name]; has {
		built = entry
	}
	node, has := f.nodes[built]
	if !has {
		return nil, errf(ErrFlow, fmt.Sprintf("flow has no node '%s'", name))
	}
	return node, nil
}

// Building

func (f *Flow) buildNode(name string, node *FlowConfigNode) *NodeConfig {
	where := fmt.Sprintf("node '%s'", name)
	respond := node.respondsImmediately()
	config := &NodeConfig{
		Name:               name,
		TaskMessages:       make([]frames.Message, 0, len(node.TaskMessages)),
		RespondImmediately: &respond,
		RoleMessage:        node.RoleMessage,
		Metadata:           node.Metadata,
	}
	for _, m := range node.TaskMessages {
		config.TaskMessages = append(config.TaskMessages, frames.Message{Role: frames.Role(m.Role), Text: m.Content})
	}
	for i := range node.Functions {
		if fn, ok := f.buildFunction(&node.Functions[i], where, name); ok {
			config.Functions = append(config.Functions, fn)
		}
	}
	for i := range node.PreActions {
		config.PreActions = append(config.PreActions,
			f.buildAction(&node.PreActions[i], where+" pre_actions", name))
	}
	for i := range node.PostActions {
		config.PostActions = append(config.PostActions,
			f.buildAction(&node.PostActions[i], where+" post_actions", name))
	}
	if node.ContextStrategy != "" {
		config.ContextStrategy = &ContextStrategyConfig{Strategy: node.ContextStrategy}
	}
	return config
}

// buildFunction builds the NodeFunction for one function entry, or records why
// it cannot be built.
func (f *Flow) buildFunction(ref *FlowConfigFunction, where, node string) (NodeFunction, bool) {
	if ref.TransitionOnly {
		return NodeFunction{
			Name:        ref.Name,
			Description: ref.Description,
			Properties:  map[string]any{},
			Required:    []string{},
			Handler:     f.makeTransitionHandler(ref),
			Metadata:    ref.Metadata,
		}, true
	}
	target := f.lookup(ref.Name, kindTool, where, node)
	if target == nil {
		return NodeFunction{}, false
	}
	var tool NodeFunction
	switch t := target.(type) {
	case NodeFunction:
		tool = t
	case *NodeFunction:
		tool = *t
	}
	if tool.Handler == nil {
		f.problem(ProblemInvalidTool,
			fmt.Sprintf("%s tool '%s' is not a valid NodeFunction: it has no handler", where, ref.Name),
			node, ref.Name)
		return NodeFunction{}, false
	}

	// The config's name is the tool's identity: it is what the LLM sees, what
	// the config's other entries refer to, and it may map several names to one
	// function.
	return NodeFunction{
		Name:                 ref.Name,
		Description:          tool.Description,
		Properties:           tool.Properties,
		Required:             tool.Required,
		Handler:              f.makeHandler(ref, tool.Handler, where),
		CancelOnInterruption: tool.CancelOnInterruption,
		TimeoutSecs:          tool.TimeoutSecs,
		Metadata:             ref.Metadata,
	}, true
}

func (f *Flow) buildAction(action *FlowConfigAction, where, node string) ActionConfig {
	config := ActionConfig{Type: action.Type}
	if action.Handler != "" {
		switch h := f.lookup(action.Handler, kindActionHandler, where, node).(type) {
		case ActionHandler:
			config.Handler = h
		case func(context.Context, ActionConfig, *FlowManager) error:
			config.Handler = h
		}
	}
	for key, value := range action.Extras {
		text, isText := value.(string)
		flag, isFlag := value.(bool)
		switch {
		case key == "text" && isText:
			config.Text = text
		case key == "append_text_to_context" && isFlag:
			config.AppendTextToContext = &flag
		default:
			if config.Params == nil {
				config.Params = make(map[string]any)
			}
			config.Params[key] = value
		}
	}
	return config
}

// lookup resolves name in the handlers, or records why it cannot be resolved
// and returns nil.
func (f *Flow) lookup(name, kind, where, node string) any {
	isTool := kind == kindTool
	function := ""
	if isTool {
		function = name
	}

	var (
		target any
		found  []string
	)
	for index, namespace := range f.handlers {
		if candidate, has := namespace[name]; has && !isNilHandler(candidate) {
			target = candidate
			found = append(found, fmt.Sprintf("handlers[%d]", index))
		}
	}
	switch {
	case len(found) > 1:
		f.problem(pick(isTool, ProblemAmbiguousTool, ProblemAmbiguousHandler),
			fmt.Sprintf("%s %s '%s' is defined in more than one of the handlers (%s)",
				where, kind, name, strings.Join(found, ", ")),
			node, function)
		return nil
	case target == nil:
		f.problem(pick(isTool, ProblemMissingTool, ProblemMissingHandler),
			fmt.Sprintf("%s references %s '%s', which is not in the handlers", where, kind, name),
			node, function)
		return nil
	}

	want := "an ActionHandler"
	ok := false
	switch target.(type) {
	case NodeFunction, *NodeFunction:
		ok = isTool
	case ActionHandler, func(context.Context, ActionConfig, *FlowManager) error:
		ok = !isTool
	}
	if isTool {
		want = "a NodeFunction"
	}
	if !ok {
		f.problem(pick(isTool, ProblemInvalidTool, ProblemMissingHandler),
			fmt.Sprintf("%s %s '%s' is not %s", where, kind, name, want),
			node, function)
		return nil
	}
	return target
}

// isNilHandler reports whether a value in the handlers is there in name only.
func isNilHandler(value any) bool {
	switch v := value.(type) {
	case nil:
		return true
	case *NodeFunction:
		return v == nil
	case ActionHandler:
		return v == nil
	case func(context.Context, ActionConfig, *FlowManager) error:
		return v == nil
	default:
		return false
	}
}

// pick returns a when cond holds, and b otherwise.
func pick(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

func (f *Flow) problem(code, message, node, function string) {
	f.problems = append(f.problems, FlowProblem{Code: code, Message: message, Node: node, Function: function})
}

// Calling

func (f *Flow) makeHandler(ref *FlowConfigFunction, tool Handler, where string) Handler {
	return func(ctx context.Context, args json.RawMessage, fm *FlowManager) (string, *NodeConfig, error) {
		result, next, err := tool(ctx, args, fm)
		if err != nil {
			return "", nil, err
		}
		if err = checkContract(next, ref.Name, where); err != nil {
			return "", nil, err
		}
		if next == NoResponse {
			return result, NoResponse, nil
		}
		destination, err := f.destination(ctx, ref, result, where)
		if err != nil {
			return "", nil, err
		}
		return result, destination, nil
	}
}

func (f *Flow) makeTransitionHandler(ref *FlowConfigFunction) Handler {
	return func(context.Context, json.RawMessage, *FlowManager) (string, *NodeConfig, error) {
		node, err := f.builtNode(ref.TransitionTo.Node)
		return "", node, err
	}
}

// destination is the node a tool's transition leads to given its result, or nil
// to stay on the current node.
func (f *Flow) destination(
	ctx context.Context, ref *FlowConfigFunction, result, where string,
) (*NodeConfig, error) {
	target := ref.TransitionTo
	switch {
	case target.Branch != nil:
	case target.Node != "":
		return f.builtNode(target.Node)
	default:
		return nil, nil //nolint:nilnil // no transition is not a failure: the flow stays on its node
	}

	branch := target.Branch
	value, has := resultField(result, branch.Field)
	if !has {
		return nil, errf(ErrFlow, fmt.Sprintf(
			"%s tool '%s' branches on result field '%s', but its result has no such field",
			where, ref.Name, branch.Field))
	}
	name, matched := branch.Cases[caseKey(value)]
	if !matched {
		name = branch.Default
	}
	if name == "" {
		slog.DebugContext(ctx, "flows: no branch case matched, staying", "where", where, "tool", ref.Name)
		return nil, nil //nolint:nilnil // an unmatched branch is not a failure: the flow stays on its node
	}
	return f.builtNode(name)
}

// builtNode is the built node a transition names.
func (f *Flow) builtNode(name string) (*NodeConfig, error) {
	node, has := f.nodes[name]
	if !has {
		return nil, errf(ErrFlow, fmt.Sprintf("flow has no node '%s'", name))
	}
	return node, nil
}

// resultField reads a field of a tool's result, which a branch can do only when
// the result is a JSON object.
func resultField(result, field string) (any, bool) {
	dec := json.NewDecoder(bytes.NewReader([]byte(result)))
	dec.UseNumber()
	var fields map[string]any
	if err := dec.Decode(&fields); err != nil || fields == nil {
		return nil, false
	}
	value, has := fields[field]
	return value, has
}

// flattened is a config with each subflow copied in under the node that runs
// it. Every transition in its nodes and functions names a built node.
type flattened struct {
	// nodes are the nodes to build, keyed by the name each is built under, and
	// names is those names in the order to build them.
	nodes map[string]*FlowConfigNode
	names []string
	// globalFunctions are the config's global functions.
	globalFunctions []FlowConfigFunction
	// entries is, for every node name in the config, the name of the built node
	// it leads to.
	entries map[string]string
	// parents is, for every node copied from a subflow, the name of the node
	// that runs the subflow.
	parents map[string]string
}

// flatten copies each subflow in under the node that runs it.
func flatten(config *FlowConfig) flattened {
	entry := func(name string) string {
		node := config.Nodes[name]
		if node == nil || node.Flow == "" || config.Flows[node.Flow] == nil {
			return name
		}
		return name + "." + config.Flows[node.Flow].InitialNode
	}

	flat := flattened{
		nodes:   make(map[string]*FlowConfigNode),
		entries: make(map[string]string, len(config.Nodes)),
		parents: make(map[string]string),
	}
	add := func(name string, node *FlowConfigNode) {
		flat.nodes[name] = node
		flat.names = append(flat.names, name)
	}
	for _, name := range config.nodeNames() {
		node := config.Nodes[name]
		if node == nil {
			continue
		}
		flat.entries[name] = entry(name)
		subflow := config.Flows[node.Flow]
		if node.Flow == "" || subflow == nil {
			add(name, retargetNode(node, entry))
			continue
		}

		inside := func(target string) string {
			if target == returnTarget {
				return entry(node.ReturnTo)
			}
			return name + "." + target
		}
		for _, innerName := range subflow.nodeNames() {
			if subflow.Nodes[innerName] == nil {
				continue
			}
			add(name+"."+innerName, retargetNode(subflow.Nodes[innerName], inside))
			flat.parents[name+"."+innerName] = name
		}
	}

	for i := range config.GlobalFunctions {
		flat.globalFunctions = append(flat.globalFunctions, retarget(config.GlobalFunctions[i], entry))
	}
	return flat
}

// retargetNode is a copy of node with each function's transition targets passed
// through resolve.
func retargetNode(node *FlowConfigNode, resolve func(string) string) *FlowConfigNode {
	out := *node
	out.Functions = make([]FlowConfigFunction, 0, len(node.Functions))
	for i := range node.Functions {
		out.Functions = append(out.Functions, retarget(node.Functions[i], resolve))
	}
	return &out
}

// retarget is a copy of function with its transition targets passed through
// resolve.
func retarget(function FlowConfigFunction, resolve func(string) string) FlowConfigFunction {
	target := function.TransitionTo
	switch {
	case target.Branch != nil:
		branch := &FlowConfigBranch{
			Field: target.Branch.Field,
			Cases: make(map[string]string, len(target.Branch.Cases)),
		}
		for key, value := range target.Branch.Cases {
			branch.Cases[key] = resolve(value)
		}
		if target.Branch.Default != "" {
			branch.Default = resolve(target.Branch.Default)
		}
		function.TransitionTo = FlowConfigTransition{Branch: branch}
	case target.Node != "":
		function.TransitionTo = FlowConfigTransition{Node: resolve(target.Node)}
	}
	return function
}

// checkContract checks a tool's next node against the configured-flow contract:
// a tool returns TransitionInYAML, or NoResponse to stay on the node without a
// bot turn.
func checkContract(next *NodeConfig, name, where string) error {
	switch next {
	case NoResponse, TransitionInYAML:
		return nil
	case nil:
		return errf(ErrFlow, fmt.Sprintf(
			"%s tool '%s' returned nil as the next node; in a configured flow "+
				"return (result, TransitionInYAML) to let the config decide", where, name))
	default:
		return errf(ErrFlow, fmt.Sprintf(
			"%s tool '%s' returned a next node; in a configured flow the config "+
				"owns transitions, so return (result, TransitionInYAML)", where, name))
	}
}
