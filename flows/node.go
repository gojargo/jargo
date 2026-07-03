package flows

import (
	"context"
	"encoding/json"

	"github.com/gojargo/jargo/frames"
)

// Handler runs when the model calls a node function. args carries the raw JSON
// arguments the model produced, and fm is the manager driving the flow. It
// returns the result to feed back to the model as the tool result and,
// optionally, the next node to move to: a non-nil next transitions the flow, a
// nil next leaves it on the current node.
type Handler func(ctx context.Context, args json.RawMessage, fm *FlowManager) (string, *NodeConfig, error)

// NodeFunction is a tool offered to the model at a node. It pairs the schema the
// model sees with the handler that runs when the model calls it.
type NodeFunction struct {
	// Name is the tool name the model calls. It must be unique within a node and
	// is required.
	Name string
	// Description tells the model when to call the tool.
	Description string
	// Parameters is the JSON-Schema object describing the tool's arguments, for
	// example `{"type":"object","properties":{…},"required":[…]}`. An empty value
	// declares a tool that takes no arguments.
	Parameters json.RawMessage
	// Handler runs the call and may return the next node. It is required.
	Handler Handler
}

// NodeConfig defines one state of a conversation: the assistant's task at this
// point, the tools it may call, and whether it speaks on entry.
type NodeConfig struct {
	// Name labels the node in logs. It is optional.
	Name string
	// RoleMessage is the assistant's persona. It is applied to the system prompt
	// on entry and, being sticky, persists across later transitions until another
	// node sets its own. Leave it empty to keep the current persona.
	RoleMessage string
	// TaskMessages state the assistant's objective at this node. They are appended
	// to the conversation on entry, so the model pursues the new goal while
	// keeping the prior history.
	TaskMessages []frames.Message
	// Functions are the tools available at this node. A function whose handler
	// returns a next node is an edge that transitions the flow; one that returns
	// no next node only gathers data.
	Functions []NodeFunction
	// RespondImmediately controls whether the assistant generates a response as
	// soon as the node is entered. It defaults to true; point it at false for a
	// node that should wait for the user to speak first, such as the opening node
	// of a call the user initiates.
	RespondImmediately *bool
}
