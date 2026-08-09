// Package realtime converts a universal conversation into what OpenAI's
// Realtime API takes on a session.
//
// A realtime session is not a request carrying the conversation: the model
// listens to the audio and keeps the history itself, so what is converted here
// is the part of a session the conversation decides, which is the toolset and
// whether the model may or must call one.
package realtime

import (
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The keys a session payload's function-calling block is built from.
const (
	keyType        = "type"
	keyName        = "name"
	keyDescription = "description"
	keyParameters  = "parameters"
	keyTools       = "tools"
	keyToolChoice  = "tool_choice"
)

// toolTypeFunction is the only tool type the Realtime API defines.
const toolTypeFunction = "function"

// Params is the part of a Realtime session the conversation decides: the tools
// to advertise and whether the model may or must call one.
type Params struct {
	Tools      []map[string]any
	ToolChoice string
}

// Session renders the params as the session payload block the API takes, or nil
// when there are no tools to advertise.
func (p Params) Session() map[string]any {
	if len(p.Tools) == 0 {
		return nil
	}
	return map[string]any{keyTools: p.Tools, keyToolChoice: p.ToolChoice}
}

// Adapter converts a universal conversation into Realtime session parameters.
// The zero value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "openai_realtime" }

// LLMInvocationParams converts the conversation into what a session takes from
// it.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, _ adapter.Options,
) (Params, error) {
	return a.SessionParams(convo.Tools(), convo.ToolChoice()), nil
}

// SessionParams renders a toolset and choice for a session. It is what the
// service calls when the toolset changes mid-conversation, which reaches a
// realtime model as a session update rather than on the next request.
func (a *Adapter) SessionParams(tools []frames.Tool, choice frames.ToolChoice) Params {
	if choice == "" {
		choice = frames.ToolChoiceAuto
	}
	return Params{Tools: a.ToProviderToolsFormat(a.WithBuiltins(tools)), ToolChoice: string(choice)}
}

// ToProviderToolsFormat implements adapter.LLMAdapter. The Realtime API
// flattens the function fields onto the tool rather than nesting them.
func (*Adapter) ToProviderToolsFormat(tools []frames.Tool) []map[string]any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		spec := map[string]any{keyType: toolTypeFunction, keyName: t.Name}
		if t.Description != "" {
			spec[keyDescription] = t.Description
		}
		if len(t.Parameters) > 0 {
			spec[keyParameters] = t.Parameters
		}
		out = append(out, spec)
	}
	return out
}

// MessagesForLogging implements adapter.LLMAdapter. A realtime session carries
// no message list: the model hears the conversation rather than being sent it,
// so there is nothing to render.
func (*Adapter) MessagesForLogging(*frames.LLMContext) []map[string]any { return nil }
