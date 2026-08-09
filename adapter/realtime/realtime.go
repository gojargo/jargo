// Package realtime converts a universal conversation into what OpenAI's
// Realtime API takes on a session.
//
// A realtime session is not a request carrying the conversation: the model
// listens to the audio and keeps the history itself, so what is converted here
// is the part of a session the conversation decides, which is the toolset and
// whether the model may or must call one.
package realtime

import (
	"log/slog"

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
	return a.SessionParams(convo.ToolsSchema(), convo.ToolChoice()), nil
}

// SessionParams renders a toolset and choice for a session. It is what the
// service calls when the toolset changes mid-conversation, which reaches a
// realtime model as a session update rather than on the next request.
func (a *Adapter) SessionParams(schema frames.ToolsSchema, choice frames.ToolChoice) Params {
	if choice == "" {
		choice = frames.ToolChoiceAuto
	}
	return Params{
		Tools:      a.ToProviderToolsFormat(a.WithBuiltins(schema)),
		ToolChoice: string(choice),
	}
}

// ToProviderToolsFormat implements adapter.LLMAdapter. The Realtime API
// flattens the function fields onto the tool rather than nesting them, and
// reads the custom tools written for OpenAI like the other OpenAI APIs.
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	var out []map[string]any
	for _, t := range schema.Standard {
		spec := map[string]any{keyType: toolTypeFunction, keyName: t.Name}
		if t.Description != "" {
			spec[keyDescription] = t.Description
		}
		if len(t.Parameters) > 0 {
			spec[keyParameters] = t.Parameters
		}
		out = append(out, spec)
	}
	custom, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeOpenAI)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return out
	}
	return append(out, custom...)
}

// MessagesForLogging implements adapter.LLMAdapter. A realtime session carries
// no message list: the model hears the conversation rather than being sent it,
// so there is nothing to render.
func (*Adapter) MessagesForLogging(*frames.LLMContext) []map[string]any { return nil }
