// Package flows builds structured, multi-step conversations on top of jargo's
// LLM service.
//
// A conversation is a graph of nodes. Each node sets the assistant's persona and
// objective, the tools it may call, and what runs on the way in and out. A tool
// whose handler returns the next node is an edge that moves the conversation on;
// one that returns no next node only gathers data. A caller outside the graph
// moves it with SetNode, for a transition the conversation never asked for.
//
// A FlowManager sits beside the pipeline rather than in it. It steers the
// conversation by queueing frames as nodes are entered: the persona as the LLM
// service's system instruction, the node's objective as messages, and its
// toolset. Queueing rather than writing to the conversation directly is what
// orders each change against whatever else is in flight, and what lets a
// realtime service learn the toolset changed. The LLM service's tool loop then
// carries out each transition.
//
// # Transitions
//
// An edge function does not move the flow where it stands. It records the
// transition and reports its result with a context-updated callback, so the move
// happens once the result has reached the conversation and every other tool call
// of that turn has reported. A turn that called three tools therefore enters the
// next node once, against a finished turn.
//
// # Context strategies
//
// Entering a node appends its objective to the conversation by default, so
// everything said so far is kept. A node or the whole flow may instead reset the
// conversation, or reset it to a generated summary of what was said, which keeps
// the substance of a long call without its length.
//
// # Actions
//
// Actions are how a flow does something other than talk. Three are built in:
// "tts_say" speaks a fixed line, "end_conversation" ends the call, and
// "function" runs a handler at the point in the pipeline it reaches, once the
// speech queued ahead of it has been said. An application registers whatever
// else it needs with RegisterAction. A node's pre-actions run before its context
// is applied; its post-actions run after, or, on a node that waits for the user,
// once the assistant's first turn there is over.
package flows
