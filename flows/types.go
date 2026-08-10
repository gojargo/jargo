package flows

import (
	"context"
	"encoding/json"

	"github.com/gojargo/jargo/frames"
)

// Handler runs when the model calls a node function.
//
// args carries the raw JSON arguments the model produced, and fm is the manager
// driving the flow. It returns the result to feed back to the model as the tool
// result and, optionally, the next node to move to: a non-nil next transitions
// the flow, a nil next leaves it on the current node and answers from the
// result, and NoResponse leaves it there without the assistant being asked to
// say anything.
//
// Returning an empty result marks the function as transition-only: the manager
// substitutes an acknowledgement, since the model called the function and is
// owed an answer whether or not the function had one of its own.
type Handler func(ctx context.Context, args json.RawMessage, fm *FlowManager) (string, *NodeConfig, error)

// NoResponse is the sentinel a Handler returns in its next-node slot to finish
// the call without transitioning and without generation re-running.
//
// It is for a tool that has already said its piece: one that plays audio of its
// own, or that ends the conversation. A result ordinarily goes back to the model
// for it to answer from, and that answer would be spoken over whatever the tool
// started. The result still reaches the conversation, so the model knows what
// happened when the user speaks next.
//
//nolint:gochecknoglobals // sentinel node, compared by identity
var NoResponse = &NodeConfig{Name: "no-response"}

// ActionHandler runs one action. It is given the action that named it and the
// manager driving the flow, so an action can read the flow's state or move it.
type ActionHandler func(ctx context.Context, action ActionConfig, fm *FlowManager) error

// ActionConfig configures one action to run on entering or leaving a node.
//
// Type names the action, and is what decides which handler runs: one of the
// built-ins ("tts_say", "end_conversation", "function") or a type registered
// with FlowManager.RegisterAction. A config carrying a Handler registers it
// under Type the first time the action is seen, so an action can bring its own
// implementation rather than being registered up front.
type ActionConfig struct {
	// Type identifies the action. It is required.
	Type string
	// Handler runs the action. It is required for the "function" action, which is
	// nothing but a handler to run in pipeline order, and optional otherwise: for
	// any other type it registers the handler under Type when the action is first
	// seen.
	Handler ActionHandler
	// Text is what "tts_say" speaks, or the optional goodbye "end_conversation"
	// speaks before ending.
	Text string
	// AppendTextToContext controls whether the text the built-in speaking actions
	// say is written to the conversation. Nil defaults to true.
	AppendTextToContext *bool
	// Params carries whatever else the action needs, for a custom handler to
	// read. The built-in actions do not look at it.
	Params map[string]any
}

// appendTextToContext reports whether a speaking action's text is written to the
// conversation. The zero value (nil) means it is.
func (a ActionConfig) appendTextToContext() bool {
	return a.AppendTextToContext == nil || *a.AppendTextToContext
}

// ContextStrategy says what happens to the conversation when a node is entered.
type ContextStrategy string

const (
	// ContextStrategyAppend adds the node's task messages to the conversation,
	// keeping everything said so far. It is the default.
	ContextStrategyAppend ContextStrategy = "append"
	// ContextStrategyReset replaces the conversation with the node's task
	// messages, so the node starts from a clean slate.
	ContextStrategyReset ContextStrategy = "reset"
	// ContextStrategyResetWithSummary replaces the conversation with a summary of
	// what was said, followed by the node's task messages. It keeps the substance
	// of a long conversation without its length.
	ContextStrategyResetWithSummary ContextStrategy = "reset_with_summary"
)

// ContextStrategyConfig configures how the conversation is updated when a node
// is entered. Set it on the manager for every node, or on a node to override it
// for that one.
type ContextStrategyConfig struct {
	// Strategy is what happens to the conversation.
	Strategy ContextStrategy
	// SummaryPrompt guides the summary, and is required by
	// ContextStrategyResetWithSummary.
	SummaryPrompt string
}

// Validate reports whether the configuration is usable.
func (c ContextStrategyConfig) Validate() error {
	if c.Strategy == ContextStrategyResetWithSummary && c.SummaryPrompt == "" {
		return errf(ErrFlow, "SummaryPrompt is required when using the reset-with-summary strategy")
	}
	return nil
}

// NodeFunction is a tool offered to the model at a node. It pairs the schema the
// model sees with the handler that runs when the model calls it, and with the
// options the call runs under.
type NodeFunction struct {
	// Name is the tool name the model calls. It must be unique within a node and
	// is required.
	Name string
	// Description tells the model when to call the tool.
	Description string
	// Properties describes the tool's arguments, as the properties of a JSON
	// Schema object: a map of argument name to its schema.
	Properties map[string]any
	// Required names the arguments the model must supply.
	Required []string
	// Handler runs the call and may return the next node. It is required.
	Handler Handler
	// CancelOnInterruption sets whether a call to this tool is canceled when the
	// user interrupts. Nil leaves the flow default, which does not cancel: a flow
	// function is usually doing work the conversation still needs whether or not
	// the user spoke over it.
	CancelOnInterruption *bool
	// TimeoutSecs bounds how long a call to this tool may take, overriding the
	// service-wide bound. Nil leaves the service-wide one.
	TimeoutSecs *float64
}

// parameters renders the function's arguments as the JSON Schema object a tool
// advertises. A function with no properties advertises an empty object, which
// is how a tool that takes no arguments is described.
func (f NodeFunction) parameters() (json.RawMessage, error) {
	props := f.Properties
	if props == nil {
		props = map[string]any{}
	}
	required := f.Required
	if required == nil {
		required = []string{}
	}
	return json.Marshal(map[string]any{
		"type":       "object",
		"properties": props,
		"required":   required,
	})
}

// cancelOnInterruption returns the call option this function registers with. A
// flow function defaults to surviving an interruption, which is not the LLM
// service's own default, so the flow default is stated rather than left out.
func (f NodeFunction) cancelOnInterruption() *bool {
	if f.CancelOnInterruption != nil {
		return f.CancelOnInterruption
	}
	stay := false
	return &stay
}

// NodeConfig defines one state of a conversation: the assistant's task at this
// point, the tools it may call, what runs on the way in and out, and whether it
// speaks on entry.
type NodeConfig struct {
	// Name labels the node in logs. It is optional; an unnamed node is given a
	// generated name so two of them are told apart.
	Name string
	// RoleMessage is the assistant's persona. It is sent as the LLM service's
	// system instruction on entry and, being sticky, persists across later
	// transitions until another node sets its own. Leave it empty to keep the
	// current persona.
	RoleMessage string
	// TaskMessages state the assistant's objective at this node. They are added to
	// the conversation on entry, as the node's context strategy says.
	//
	// It is required: a nil slice is a node that never said what it is for, and is
	// rejected. An empty non-nil slice is a node that deliberately adds nothing,
	// which is allowed.
	TaskMessages []frames.Message
	// Functions are the tools available at this node. A function whose handler
	// returns a next node is an edge that transitions the flow; one that returns
	// no next node only gathers data.
	Functions []NodeFunction
	// PreActions run before the node's context is applied, so what they say is
	// spoken ahead of anything the node generates.
	PreActions []ActionConfig
	// PostActions run after the node's context is applied. On a node that speaks
	// on entry they run immediately; on one that waits for the user they are held
	// back until the assistant's first turn at this node is over.
	PostActions []ActionConfig
	// ContextStrategy overrides the manager's strategy for this node. Nil uses the
	// manager's.
	ContextStrategy *ContextStrategyConfig
	// RespondImmediately controls whether the assistant generates a response as
	// soon as the node is entered. It defaults to true; point it at false for a
	// node that should wait for the user to speak first, such as the opening node
	// of a call the user initiates.
	RespondImmediately *bool
}

// respondsImmediately reports whether the node should generate a response on
// entry. The zero value (nil) means it should.
func (n *NodeConfig) respondsImmediately() bool {
	return n.RespondImmediately == nil || *n.RespondImmediately
}
