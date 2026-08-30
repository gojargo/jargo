// Package llmworker is a pipeline worker built around a language model, with
// the tool handling that makes one usable from the bus.
//
// A worker's tools run while the pipeline that called them is still running,
// and what a tool does often needs to reach that pipeline: appending to the
// conversation, ending the session, handing over to another worker. Doing any of
// it while the call is still in flight puts it out of order with the result the
// model is waiting for. This package holds it back until the call is done.
package llmworker

import "github.com/gojargo/jargo/workers"

// ActivationArgs are what an LLM worker understands when it is activated: the
// messages to put into the conversation, and whether to answer them.
type ActivationArgs struct {
	workers.BaseActivationArgs
	// Messages are appended to the conversation when the worker activates.
	Messages []map[string]any
	// RunLLM says whether the model answers them. Nil answers, which is what a
	// caller sending messages almost always means.
	RunLLM *bool
}

// ToMap implements workers.ActivationArgs.
func (a ActivationArgs) ToMap() map[string]any {
	m := a.BaseActivationArgs.ToMap()
	if len(a.Messages) > 0 {
		msgs := make([]any, 0, len(a.Messages))
		for _, msg := range a.Messages {
			msgs = append(msgs, msg)
		}
		m["messages"] = msgs
	}
	if a.RunLLM != nil {
		m["run_llm"] = *a.RunLLM
	}
	return m
}

// ActivationArgsFrom reads what an LLM worker understands out of the arguments
// it was activated with, ignoring anything else in them.
func ActivationArgsFrom(args map[string]any) ActivationArgs {
	a := ActivationArgs{BaseActivationArgs: workers.BaseActivationArgsFrom(args)}
	if listed, ok := args["messages"].([]any); ok {
		for _, item := range listed {
			if msg, ok := item.(map[string]any); ok {
				a.Messages = append(a.Messages, msg)
			}
		}
	}
	if run, ok := args["run_llm"].(bool); ok {
		a.RunLLM = &run
	}
	return a
}

// runLLM reports whether the model answers the messages, which it does unless
// the caller said otherwise.
func (a ActivationArgs) runLLM() bool { return a.RunLLM == nil || *a.RunLLM }
