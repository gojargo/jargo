package flows

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// returnTarget is the transition target that, inside a subflow, leads to the
// ReturnTo of the node running it.
const returnTarget = "return"

// builtInActionWithoutHandler reports whether actionType is a built-in action
// whose behavior is fixed, so a handler is not allowed.
func builtInActionWithoutHandler(actionType string) bool {
	return actionType == actionTTSSay || actionType == actionEndConversation
}

// builtInAction reports whether actionType is one the runtime provides without
// registration.
func builtInAction(actionType string) bool {
	return builtInActionWithoutHandler(actionType) || actionType == actionFunction
}

// FlowConfig is a conversation flow described as data: the nodes, what each one
// says, which tools each node offers, and where each tool leads. It contains no
// code. Every tool a node references is a NodeFunction that lives in the
// application's code and is resolved by name when the config is joined to the
// application's handlers with NewFlow. The exception is a transition_only
// function, which does nothing but move the conversation to another node: the
// config supplies its description and it needs no code.
//
// Load one with FlowConfigFromFile for a YAML or JSON file, FlowConfigFromYAML
// for YAML text, FlowConfigFromJSON for JSON text, or FlowConfigFromMap for a
// document that is already parsed. A config is validated structurally on load:
// the initial node exists, every transition names a node, tool names are unique
// within a node, and every action is well-formed. NewFlow validates the
// references to code. A config built in code is checked with Validate.
//
//	initial_node: greet
//
//	nodes:
//	  greet:
//	    role_message: You are a friendly order-taking assistant.
//	    task_messages:
//	      - role: developer
//	        content: Greet the caller and ask whether they want pizza or sushi.
//	    functions:
//	      - name: choose_pizza
//	        transition_only: true
//	        description: The caller wants to order pizza.
//	        transition_to: pizza
//	      - name: choose_sushi
//	        transition_only: true
//	        description: The caller wants to order sushi.
//	        transition_to: sushi
//
//	  pizza:
//	    task_messages:
//	      - role: developer
//	        content: Take a pizza order.
//	    functions:
//	      - name: select_pizza_order
//	        transition_to:
//	          field: status
//	          cases:
//	            ok: confirm
//	            unavailable: pizza
//	          default: confirm
//
//	global_functions:
//	  - name: get_delivery_estimate
//
// `!include` resolves paths on the application's file system and can read any
// file the process can read, so it belongs only in configs the application
// ships. Load a config from another source, such as a request body, with
// FlowConfigFromYAML and no base directory, which rejects `!include`, or with
// FlowConfigFromJSON.
//
// Prompt text may refer to the manager's state with {{ key }} placeholders: a
// node's role_message, the content of its task_messages, and the text of a
// tts_say action. The FlowManager fills them from its state each time it enters
// the node, so a value stored by a handler earlier in the conversation can
// appear in a later prompt. {{ order.size }} walks into a stored map. A key
// that is not in state is an ErrFlow when the node is entered. To show the LLM
// a literal {{ key }}, escape it as \{{ key }}.
//
// Metadata: a node and a function each take an optional metadata mapping of
// JSON values, for whatever the application wants to keep with them, such as
// the id of the element a node stands for in an editor. The flow carries it and
// never acts on it, and it is never sent to the LLM. A Flow copies it onto the
// nodes and functions it builds, and the FlowManager passes it to
// EventNodeEntered handlers and to an RTVI client as the flow moves. Metadata
// is sent to the client, so keep anything the end user should not see out of
// it.
//
// Subflows: a set of nodes used by several flows, or several times in one, is
// written once as a subflow under flows. A node runs it by naming it as its
// flow, in place of prompts and functions of its own, and says where the
// subflow's return leads:
//
//	initial_node: verify_caller
//	flows:
//	  verification: !include verification.yaml
//	nodes:
//	  verify_caller:
//	    flow: verification
//	    return_to: book
//	  book:
//	    ...
//
// Entering such a node enters the subflow at its first node. Inside a subflow a
// transition target names one of its own nodes, or is return, which leads to
// the return_to of the node running it; to return to two places, run the
// subflow from two nodes. return_to may be omitted for a subflow that never
// returns. A Flow builds a subflow's nodes once for each node that runs it,
// named <node>.<subflow node>, so everything before the dot is the node that
// runs the subflow; that is the name FlowManager.CurrentNode reports, and
// EventNodeEntered gives the running node separately as Parent. A config that
// has subflows therefore cannot have a dot in a node name. A subflow's nodes
// cannot run subflows, and global_functions are offered at them too.
//
// A config remembers the order its nodes and subflows were written in, which is
// the order problems are reported in; those added in code afterwards follow, by
// name.
type FlowConfig struct {
	// InitialNode is the name of the node the flow starts in.
	InitialNode string `json:"initial_node"`
	// Nodes are the flow's nodes, keyed by name.
	Nodes map[string]*FlowConfigNode `json:"nodes"`
	// GlobalFunctions are the tools offered at every node.
	GlobalFunctions []FlowConfigFunction `json:"global_functions,omitempty"`
	// Flows are the subflows the flow's nodes can run, keyed by flow id.
	Flows map[string]*FlowConfigSubflow `json:"flows,omitempty"`

	// nodeOrder and flowOrder are the names of Nodes and Flows in the order the
	// document wrote them.
	nodeOrder []string
	flowOrder []string
}

// FlowConfigMessage is one message in a node's task_messages.
type FlowConfigMessage struct {
	// Role is the message role, for example "developer" or "system".
	Role string `json:"role"`
	// Content is the message text. It may contain {{ key }} placeholders; see
	// FlowConfig.
	Content string `json:"content"`
}

// FlowConfigBranch is a transition chosen by a field of the tool's result.
type FlowConfigBranch struct {
	// Field is the key of the tool's result whose value selects the case.
	Field string `json:"field"`
	// Cases maps a result value to a node name. In a document the keys may be
	// written as strings, booleans, or numbers; they match the result value by
	// its canonical string, so `true:` matches a result of true.
	Cases map[string]string `json:"cases"`
	// Default is the node to transition to when the value matches no case.
	// When empty, an unmatched value stays on the current node.
	Default string `json:"default,omitempty"`
}

// Targets returns every node name this branch can transition to.
func (b *FlowConfigBranch) Targets() []string {
	keys := make([]string, 0, len(b.Cases))
	for key := range b.Cases {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	targets := make([]string, 0, len(keys)+1)
	for _, key := range keys {
		targets = append(targets, b.Cases[key])
	}
	if b.Default != "" {
		targets = append(targets, b.Default)
	}
	return targets
}

// FlowConfigTransition is where a function leads: nowhere (the zero value), a
// node, or a branch on the tool's result. In a document it is a node name or a
// branch mapping.
type FlowConfigTransition struct {
	// Node is the node to transition to. Inside a subflow it may also be
	// "return"; see "Subflows" in FlowConfig.
	Node string
	// Branch chooses the node from the tool's result. It replaces Node when it
	// is set.
	Branch *FlowConfigBranch
}

// IsZero reports whether the transition leads nowhere.
func (t FlowConfigTransition) IsZero() bool { return t.Node == "" && t.Branch == nil }

// MarshalJSON writes the transition as a document has it: a node name, a branch
// mapping, or null.
func (t FlowConfigTransition) MarshalJSON() ([]byte, error) {
	switch {
	case t.Branch != nil:
		return json.Marshal(t.Branch)
	case t.Node != "":
		return json.Marshal(t.Node)
	default:
		return []byte("null"), nil
	}
}

// FlowConfigFunction is a tool offered at a node.
//
// Ordinarily the entry names a NodeFunction in the handlers a Flow is built
// with, and the tool's description and parameters come from that function. A
// transition_only entry is defined entirely here instead: it takes no
// parameters, runs no code, and moves the conversation to TransitionTo when the
// LLM calls it.
type FlowConfigFunction struct {
	// Name is the tool's name, as the LLM sees it. For an ordinary entry, it is
	// also the name of the NodeFunction in the handlers.
	Name string `json:"name"`
	// TransitionOnly says the tool is defined here rather than in code. It
	// requires a Description and a TransitionTo that names a node.
	TransitionOnly bool `json:"transition_only,omitempty"`
	// Description is what the tool is for, for the LLM. Only a transition_only
	// entry has one; a NodeFunction carries its own.
	Description string `json:"description,omitempty"`
	// TransitionTo is the node to transition to after the tool completes, or a
	// branch. It is left zero for tools that stay on the current node.
	TransitionTo FlowConfigTransition `json:"transition_to,omitzero"`
	// Metadata is JSON data the application attaches to the tool. See
	// "Metadata" in FlowConfig.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Targets returns every node name this function can transition to.
func (f *FlowConfigFunction) Targets() []string {
	switch {
	case f.TransitionTo.Branch != nil:
		return f.TransitionTo.Branch.Targets()
	case f.TransitionTo.Node != "":
		return []string{f.TransitionTo.Node}
	default:
		return nil
	}
}

// checkShape checks the fields a function takes against its kind, returning
// what is wrong or "".
func (f *FlowConfigFunction) checkShape() string {
	switch {
	case f.TransitionOnly && f.Description == "":
		return fmt.Sprintf("function '%s' is transition_only and needs a description", f.Name)
	case f.TransitionOnly && (f.TransitionTo.Branch != nil || f.TransitionTo.Node == ""):
		return fmt.Sprintf("function '%s' is transition_only and must name the node it transitions to", f.Name)
	case !f.TransitionOnly && f.Description != "":
		return descriptionNotTaken(f.Name)
	default:
		return ""
	}
}

// descriptionNotTaken is what is wrong with an ordinary function that has a
// description.
func descriptionNotTaken(name string) string {
	return fmt.Sprintf("function '%s' has a description, which only a transition_only function takes; "+
		"a function in the handlers carries its own", name)
}

// FlowConfigAction is a pre- or post-action on a node.
//
// The built-in tts_say and end_conversation types take no handler. The built-in
// function type requires one: the handler runs inline in the pipeline, queued
// behind the bot's turn. A custom type may name a handler too, which then runs
// immediately when the node's actions execute; a custom type without one must
// be registered in code with FlowManager.RegisterAction. Any additional keys
// pass through to the handler. The text of a tts_say action may contain
// {{ key }} placeholders; see FlowConfig.
type FlowConfigAction struct {
	// Type is the action type identifier.
	Type string
	// Handler is the name of the handler in the handlers a Flow is built with.
	// It is required for function, optional for custom types, and not allowed
	// on tts_say or end_conversation.
	Handler string
	// Extras are the pass-through keys beyond type and handler.
	Extras map[string]any
}

// RegisteredInCode reports whether this is a custom type whose handler the
// config does not name.
func (a *FlowConfigAction) RegisteredInCode() bool {
	return !builtInAction(a.Type) && a.Handler == ""
}

// MarshalJSON writes the action as a document has it: its type, its handler and
// the pass-through keys side by side.
func (a FlowConfigAction) MarshalJSON() ([]byte, error) {
	out := make(map[string]any, len(a.Extras)+2)
	maps.Copy(out, a.Extras)
	out["type"] = a.Type
	if a.Handler != "" {
		out["handler"] = a.Handler
	}
	return json.Marshal(out)
}

// checkHandler checks an action's handler against its type, returning what is
// wrong or "".
func (a *FlowConfigAction) checkHandler() string {
	switch {
	case a.Type == actionFunction && a.Handler == "":
		return "a 'function' action requires a 'handler' name"
	case builtInActionWithoutHandler(a.Type) && a.Handler != "":
		return fmt.Sprintf("the built-in '%s' action does not take a 'handler'", a.Type)
	default:
		return ""
	}
}

// FlowConfigNode is one node of the flow.
//
// A node either prompts the LLM, with TaskMessages and the fields that go with
// them, or runs a subflow, with Flow and ReturnTo in their place. See
// "Subflows" in FlowConfig.
type FlowConfigNode struct {
	// TaskMessages are what the LLM should do at this node. They are required
	// unless the node runs a subflow: a nil slice is a node that never said
	// what it is for, an empty non-nil one a node that deliberately adds
	// nothing.
	TaskMessages []FlowConfigMessage `json:"task_messages,omitzero"`
	// RoleMessage is the bot's role or personality, sent as the LLM's system
	// instruction on entering this node. It persists across transitions until
	// another node sets its own. It may contain {{ key }} placeholders; see
	// FlowConfig.
	RoleMessage string `json:"role_message,omitempty"`
	// Functions are the tools offered at this node, in addition to the config's
	// GlobalFunctions.
	Functions []FlowConfigFunction `json:"functions,omitempty"`
	// PreActions are the actions run before the LLM responds at this node.
	PreActions []FlowConfigAction `json:"pre_actions,omitempty"`
	// PostActions are the actions run after the LLM responds at this node.
	PostActions []FlowConfigAction `json:"post_actions,omitempty"`
	// ContextStrategy is how the LLM context is updated on entering this node:
	// ContextStrategyAppend or ContextStrategyReset. Empty uses the
	// FlowManager's strategy.
	ContextStrategy ContextStrategy `json:"context_strategy,omitempty"`
	// RespondImmediately says whether the LLM responds as soon as the node is
	// entered. Nil defaults to true.
	RespondImmediately *bool `json:"respond_immediately,omitempty"`
	// Metadata is JSON data the application attaches to the node. See
	// "Metadata" in FlowConfig.
	Metadata map[string]any `json:"metadata,omitempty"`
	// Flow is the id of the subflow this node runs, a key of the config's
	// Flows. A node that runs a subflow takes only ReturnTo and Metadata
	// besides.
	Flow string `json:"flow,omitempty"`
	// ReturnTo is the node that return in the subflow leads to. It may be
	// omitted when the subflow never returns.
	ReturnTo string `json:"return_to,omitempty"`
}

// respondsImmediately reports whether the node responds on entry. The zero
// value (nil) means it does.
func (n *FlowConfigNode) respondsImmediately() bool {
	return n.RespondImmediately == nil || *n.RespondImmediately
}

// checkShape checks that a node prompts the LLM or runs a flow, not both, and
// that its function names are unique. It returns what is wrong or "".
func (n *FlowConfigNode) checkShape() string {
	switch {
	case n.Flow == "" && n.TaskMessages == nil:
		return "a node needs 'task_messages', or a 'flow' to run"
	case n.Flow == "" && n.ReturnTo != "":
		return "'return_to' applies only to a node that runs a 'flow'"
	case n.Flow != "" && n.promptsTheLLM():
		return fmt.Sprintf("a node that runs flow '%s' takes only 'flow', 'return_to' and 'metadata'", n.Flow)
	default:
		return checkUnique(n.Functions, "node")
	}
}

// promptsTheLLM reports whether the node sets any of the fields that belong to
// a node with prompts of its own.
func (n *FlowConfigNode) promptsTheLLM() bool {
	return len(n.TaskMessages) > 0 ||
		n.RoleMessage != "" ||
		len(n.Functions) > 0 ||
		len(n.PreActions) > 0 ||
		len(n.PostActions) > 0 ||
		n.ContextStrategy != "" ||
		!n.respondsImmediately()
}

// FlowConfigSubflow is a set of nodes that a flow's nodes can run.
//
// Inside a subflow a function transitions to one of the subflow's own nodes, or
// to return. See "Subflows" in FlowConfig.
type FlowConfigSubflow struct {
	// InitialNode is the name of the node the subflow starts in.
	InitialNode string `json:"initial_node"`
	// Nodes are the subflow's nodes, keyed by name.
	Nodes map[string]*FlowConfigNode `json:"nodes"`

	// nodeOrder is the names of Nodes in the order the document wrote them.
	nodeOrder []string
}

// Returns reports whether any function in the subflow transitions to return.
func (s *FlowConfigSubflow) Returns() bool {
	for _, node := range s.Nodes {
		if node == nil {
			continue
		}
		for i := range node.Functions {
			if slices.Contains(node.Functions[i].Targets(), returnTarget) {
				return true
			}
		}
	}
	return false
}

// nodeNames is the names of the subflow's nodes, in document order.
func (s *FlowConfigSubflow) nodeNames() []string { return orderedKeys(s.Nodes, s.nodeOrder) }

// checkGraph checks the subflow's nodes against each other, returning what is
// wrong or "".
func (s *FlowConfigSubflow) checkGraph() string {
	if _, has := s.Nodes[returnTarget]; has {
		return fmt.Sprintf("a subflow cannot have a node named '%s', which is how its functions leave it",
			returnTarget)
	}
	names := s.nodeNames()
	if msg := checkNoDots(names); msg != "" {
		return msg
	}
	if _, has := s.Nodes[s.InitialNode]; !has {
		return fmt.Sprintf("initial_node '%s' is not a defined node", s.InitialNode)
	}
	targets := map[string]bool{returnTarget: true}
	for _, name := range names {
		targets[name] = true
	}
	for _, name := range names {
		node := s.Nodes[name]
		if node.Flow != "" {
			return fmt.Sprintf("node '%s' runs a flow, which a subflow's nodes cannot do", name)
		}
		for i := range node.Functions {
			if msg := checkTargets(&node.Functions[i], targets, fmt.Sprintf("node '%s'", name)); msg != "" {
				return msg
			}
		}
	}
	return ""
}

// nodeNames is the names of the config's nodes, in document order.
func (c *FlowConfig) nodeNames() []string { return orderedKeys(c.Nodes, c.nodeOrder) }

// flowNames is the ids of the config's subflows, in document order.
func (c *FlowConfig) flowNames() []string { return orderedKeys(c.Flows, c.flowOrder) }

// checkGraph checks the config's nodes, subflows and global functions against
// each other, returning what is wrong or "".
func (c *FlowConfig) checkGraph() string {
	names := c.nodeNames()
	if len(c.Flows) > 0 {
		// A subflow's nodes are built as <node>.<subflow node>.
		if msg := checkNoDots(names); msg != "" {
			return msg
		}
	}
	if _, has := c.Nodes[c.InitialNode]; !has {
		return fmt.Sprintf("initial_node '%s' is not a defined node", c.InitialNode)
	}
	if msg := c.checkSubflowsRun(names); msg != "" {
		return msg
	}
	if msg := checkUnique(c.GlobalFunctions, "global_functions"); msg != "" {
		return msg
	}
	if msg := c.checkGlobalNames(names); msg != "" {
		return msg
	}

	targets := make(map[string]bool, len(names))
	for _, name := range names {
		targets[name] = true
	}
	for _, name := range names {
		node := c.Nodes[name]
		for i := range node.Functions {
			if msg := checkTargets(&node.Functions[i], targets, fmt.Sprintf("node '%s'", name)); msg != "" {
				return msg
			}
		}
	}
	for i := range c.GlobalFunctions {
		if msg := checkTargets(&c.GlobalFunctions[i], targets, "global_functions"); msg != "" {
			return msg
		}
	}
	return ""
}

// checkSubflowsRun checks that each node that runs a subflow names one the
// config has and says where it returns when it does, and that every subflow is
// run.
func (c *FlowConfig) checkSubflowsRun(names []string) string {
	run := make(map[string]bool)
	for _, name := range names {
		node := c.Nodes[name]
		if node.Flow == "" {
			continue
		}
		subflow := c.Flows[node.Flow]
		if subflow == nil {
			return fmt.Sprintf("node '%s' runs unknown flow '%s'", name, node.Flow)
		}
		run[node.Flow] = true
		if node.ReturnTo == "" {
			if subflow.Returns() {
				return fmt.Sprintf("node '%s' runs flow '%s', which returns, and needs a 'return_to'",
					name, node.Flow)
			}
		} else if _, has := c.Nodes[node.ReturnTo]; !has {
			return fmt.Sprintf("node '%s' returns to unknown node '%s'", name, node.ReturnTo)
		}
	}
	for _, id := range c.flowNames() {
		if !run[id] {
			return fmt.Sprintf("flow '%s' is not run by any node", id)
		}
	}
	return ""
}

// checkGlobalNames checks that no node, of the flow or of a subflow, has a
// function named as a global function is.
func (c *FlowConfig) checkGlobalNames(names []string) string {
	global := make(map[string]bool, len(c.GlobalFunctions))
	for i := range c.GlobalFunctions {
		global[c.GlobalFunctions[i].Name] = true
	}
	check := func(scope string, nodes map[string]*FlowConfigNode, names []string) string {
		for _, name := range names {
			for i := range nodes[name].Functions {
				if fn := nodes[name].Functions[i].Name; global[fn] {
					return fmt.Sprintf("%snode '%s' function '%s' is also a global function", scope, name, fn)
				}
			}
		}
		return ""
	}
	if msg := check("", c.Nodes, names); msg != "" {
		return msg
	}
	for _, id := range c.flowNames() {
		if msg := check(fmt.Sprintf("flow '%s' ", id), c.Flows[id].Nodes, c.Flows[id].nodeNames()); msg != "" {
			return msg
		}
	}
	return ""
}

// checkUnique reports the first function name used twice.
func checkUnique(functions []FlowConfigFunction, where string) string {
	seen := make(map[string]bool, len(functions))
	for i := range functions {
		name := functions[i].Name
		if seen[name] {
			return fmt.Sprintf("duplicate function '%s' in %s", name, where)
		}
		seen[name] = true
	}
	return ""
}

// checkNoDots reports the first node name with a dot in it.
func checkNoDots(names []string) string {
	for _, name := range names {
		if strings.Contains(name, ".") {
			return fmt.Sprintf(
				"node '%s' has a '.' in its name, which a config that has subflows cannot have", name)
		}
	}
	return ""
}

// checkTargets reports the first of a function's targets that is not a node.
func checkTargets(function *FlowConfigFunction, targets map[string]bool, where string) string {
	for _, target := range function.Targets() {
		if !targets[target] {
			return fmt.Sprintf("%s function '%s' transitions to unknown node '%s'", where, function.Name, target)
		}
	}
	return ""
}

// orderedKeys is the keys of m: those order names first, as it has them, then
// the rest by name.
func orderedKeys[V any](m map[string]V, order []string) []string {
	out := make([]string, 0, len(m))
	seen := make(map[string]bool, len(m))
	for _, key := range order {
		if _, has := m[key]; has && !seen[key] {
			seen[key] = true
			out = append(out, key)
		}
	}
	rest := make([]string, 0, len(m)-len(out))
	for key := range m {
		if !seen[key] {
			rest = append(rest, key)
		}
	}
	slices.Sort(rest)
	return append(out, rest...)
}

// FlowConfigErrorDetail is one thing wrong with a flow config.
type FlowConfigErrorDetail struct {
	// Loc is where in the document the error is, as a dotted path such as
	// "nodes.greet.functions.0", or "" for the document as a whole.
	Loc string
	// Msg says what is wrong.
	Msg string
}

// FlowConfigError reports a flow config that is not valid: a document of the
// wrong shape, or one whose nodes do not fit together.
type FlowConfigError struct {
	// Details is everything found wrong, in the order found.
	Details []FlowConfigErrorDetail
}

// Error implements error.
func (e *FlowConfigError) Error() string {
	var b strings.Builder
	plural := "s"
	if len(e.Details) == 1 {
		plural = ""
	}
	fmt.Fprintf(&b, "flow config has %d validation error%s:", len(e.Details), plural)
	for _, d := range e.Details {
		b.WriteString("\n- ")
		if d.Loc != "" {
			b.WriteString(d.Loc + ": ")
		}
		b.WriteString(d.Msg)
	}
	return b.String()
}

// The messages of a FlowConfigError that are about one field.
const (
	msgTooFewItems = "Dictionary should have at least 1 item after validation, not 0"
	msgNotJSON     = "Input should be a valid JSON object"
	msgNotMapping  = "Input should be a valid dictionary"
	msgNotStrategy = "Input should be 'append' or 'reset'"
	msgTopLevel    = "top level must be a mapping"
)

// The fields of a config document, as a FlowConfigErrorDetail locates them.
const (
	locInitialNode    = "initial_node"
	locNodes          = "nodes"
	locGlobals        = "global_functions"
	locFlows          = "flows"
	locTaskMessages   = "task_messages"
	locRoleMessage    = "role_message"
	locFunctions      = "functions"
	locPreActions     = "pre_actions"
	locPostActions    = "post_actions"
	locStrategy       = "context_strategy"
	locRespond        = "respond_immediately"
	locMetadata       = "metadata"
	locFlow           = "flow"
	locReturnTo       = "return_to"
	locName           = "name"
	locTransitionOnly = "transition_only"
	locDescription    = "description"
	locTransitionTo   = "transition_to"
	locCases          = "transition_to.cases"
	locType           = "type"
	locHandler        = "handler"
)

// at is the location of part under loc.
func at(loc string, parts ...any) string {
	for _, part := range parts {
		if loc != "" {
			loc += "."
		}
		loc += fmt.Sprint(part)
	}
	return loc
}

// Validate checks a config built or changed in code the way a loaded one is
// checked: every function and action is well-formed, a node prompts the LLM or
// runs a subflow, and the nodes, subflows and transitions fit together. It
// returns a *FlowConfigError holding everything found wrong, or nil.
func (c *FlowConfig) Validate() error {
	v := &configValidator{}
	v.config(c)
	if len(v.details) > 0 {
		return &FlowConfigError{Details: v.details}
	}
	return nil
}

// configValidator collects what is wrong with a config. A model's own check
// runs only when everything inside it is valid, so one mistake is reported
// once, where it is.
type configValidator struct {
	details []FlowConfigErrorDetail
}

// fail records msg at loc and returns false.
func (v *configValidator) fail(loc, msg string) bool {
	v.details = append(v.details, FlowConfigErrorDetail{Loc: loc, Msg: msg})
	return false
}

// check records a model's own check when it found something.
func (v *configValidator) check(loc, msg string) bool {
	if msg != "" {
		return v.fail(loc, msg)
	}
	return true
}

// metadata checks that metadata can be written as JSON.
func (v *configValidator) metadata(loc string, metadata map[string]any) bool {
	if metadata == nil {
		return true
	}
	if _, err := json.Marshal(metadata); err != nil {
		return v.fail(at(loc, locMetadata), msgNotJSON)
	}
	return true
}

func (v *configValidator) function(loc string, f *FlowConfigFunction) bool {
	ok := v.metadata(loc, f.Metadata)
	if f.TransitionTo.Branch != nil && len(f.TransitionTo.Branch.Cases) == 0 {
		ok = v.fail(at(loc, locCases), msgTooFewItems)
	}
	return ok && v.check(loc, f.checkShape())
}

func (v *configValidator) functions(loc string, functions []FlowConfigFunction) bool {
	ok := true
	for i := range functions {
		ok = v.function(at(loc, i), &functions[i]) && ok
	}
	return ok
}

func (v *configValidator) actions(loc string, actions []FlowConfigAction) bool {
	ok := true
	for i := range actions {
		ok = v.check(at(loc, i), actions[i].checkHandler()) && ok
	}
	return ok
}

func (v *configValidator) node(loc string, n *FlowConfigNode) bool {
	if n == nil {
		return v.fail(loc, msgNotMapping)
	}
	ok := v.functions(at(loc, locFunctions), n.Functions)
	ok = v.actions(at(loc, locPreActions), n.PreActions) && ok
	ok = v.actions(at(loc, locPostActions), n.PostActions) && ok
	ok = v.metadata(loc, n.Metadata) && ok
	switch n.ContextStrategy {
	case "", ContextStrategyAppend, ContextStrategyReset:
	default:
		ok = v.fail(at(loc, locStrategy), msgNotStrategy)
	}
	return ok && v.check(loc, n.checkShape())
}

func (v *configValidator) nodes(loc string, nodes map[string]*FlowConfigNode, names []string) bool {
	if len(nodes) == 0 {
		return v.fail(loc, msgTooFewItems)
	}
	ok := true
	for _, name := range names {
		ok = v.node(at(loc, name), nodes[name]) && ok
	}
	return ok
}

func (v *configValidator) subflow(loc string, s *FlowConfigSubflow) bool {
	if s == nil {
		return v.fail(loc, msgNotMapping)
	}
	return v.nodes(at(loc, locNodes), s.Nodes, s.nodeNames()) && v.check(loc, s.checkGraph())
}

func (v *configValidator) config(c *FlowConfig) {
	ok := v.nodes(locNodes, c.Nodes, c.nodeNames())
	ok = v.functions(locGlobals, c.GlobalFunctions) && ok
	for _, id := range c.flowNames() {
		ok = v.subflow(at(locFlows, id), c.Flows[id]) && ok
	}
	if ok {
		v.check("", c.checkGraph())
	}
}
