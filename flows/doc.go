// Package flows builds structured, multi-step conversations on top of jargo's
// LLM service. A conversation is a graph of nodes; each node sets the
// assistant's task and the tools it may call, and a tool whose handler returns
// the next node moves the conversation on. A caller outside the graph moves it
// with SetNode, for a transition the conversation never asked for.
//
// A FlowManager sits beside the pipeline rather than in it. It shares the LLM
// service and the conversation context with the pipeline and steers the
// conversation by swapping the system prompt, task messages and toolset as nodes
// are entered; the LLM service's existing tool loop then carries out each
// transition, regenerating with the new node's tools.
//
// This is the initial, transition-focused version: nodes, edge and
// data-gathering functions, and per-node "respond on entry" control. Node
// actions (speaking fixed lines on entry or exit), context-reset strategies and
// per-call timeouts are not implemented yet.
package flows
