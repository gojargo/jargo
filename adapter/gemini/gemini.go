// Package gemini converts a universal conversation into the request Gemini
// takes, and with it Vertex AI, which serves the same models.
package gemini

import (
	"encoding/json"
	"log/slog"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The keys Gemini's generateContent body is built from.
const (
	keyRole  = "role"
	keyParts = "parts"
	keyName  = "name"
	keyText  = "text"
	keyID    = "id"
)

// The roles Gemini's contents take. It calls the assistant the model, and has
// no system or developer input role: a message in either enters as the user.
const (
	roleUser  = "user"
	roleModel = "model"
)

// unnamedToolResult names a tool result whose call cannot be found. Gemini
// requires a name on every functionResponse, and a result can outlive the call
// it answers: an asynchronous tool's messages carry only the call's id.
const unnamedToolResult = "tool_call_result"

// Params is what one generateContent call takes from the conversation: the
// system instruction Gemini carries beside the conversation, the contents
// themselves, and the tools to advertise.
type Params struct {
	SystemInstruction string
	Contents          []map[string]any
	Tools             []map[string]any
}

// Adapter converts a universal conversation into a Gemini request. The zero
// value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "google" }

// LLMInvocationParams converts the conversation into a generateContent request.
//
// Gemini carries the system instruction beside the conversation rather than in
// it, so it has one field to put it in: an instruction given for this call
// replaces the conversation's own prompt rather than supplementing it.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	fromContext, msgs := a.ExtractInitialSystem(
		a.SystemWithBuiltins(convo.System()), opts.SystemInstruction,
		convo.MessagesFor(a.IDForLLMSpecificMessages()),
	)
	system := a.ResolveSystemInstruction(fromContext, opts.SystemInstruction, true)

	contents, err := ToContents(msgs)
	if err != nil {
		return Params{}, err
	}
	// A conversation of nothing but tool turns gives the model no prose to answer,
	// and Gemini reads the system instruction as framing rather than as something
	// to act on. Saying it again as a user message is what prompts a reply.
	if system != "" && !hasProse(contents) {
		contents = append(contents, textContent(roleUser, system))
	}
	contents = dropEmpty(contents)

	params := Params{SystemInstruction: system, Contents: contents}
	if tools := a.WithBuiltins(convo.ToolsSchema()); len(tools.Standard) > 0 ||
		len(tools.Custom) > 0 {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ToContents converts the conversation's messages into Gemini contents. The
// assistant role maps to "model", and a system or developer message enters as
// the user, Gemini having neither input role. Tool turns become functionCall
// parts (model) and functionResponse parts (user), paired by the call id
// carried on both.
func ToContents(msgs []frames.Message) ([]map[string]any, error) {
	names := toolCallNames(msgs)
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case m.IsLLMSpecific():
			native, err := adapter.NativeMessage[map[string]any](m)
			if err != nil {
				return nil, err
			}
			out = append(out, native)
		case len(m.ToolResults) > 0:
			out = append(out, map[string]any{
				keyRole: roleUser, keyParts: toolResultParts(m.ToolResults, names),
			})
		case len(m.ToolCalls) > 0:
			out = append(out, map[string]any{
				keyRole: roleModel, keyParts: toolCallParts(m.Text, m.ToolCalls),
			})
		default:
			role := roleUser
			if m.Role == frames.RoleAssistant {
				role = roleModel
			}
			out = append(out, textContent(role, m.Text))
		}
	}
	return out, nil
}

// textContent builds a content carrying one text part.
func textContent(role, text string) map[string]any {
	return map[string]any{keyRole: role, keyParts: []map[string]any{{keyText: text}}}
}

// hasProse reports whether any content is something the model said or was told
// in words, as opposed to a tool call or its result.
func hasProse(contents []map[string]any) bool {
	for _, c := range contents {
		parts, ok := c[keyParts].([]map[string]any)
		if !ok || len(parts) != 1 {
			continue
		}
		if text, ok := parts[0][keyText].(string); ok && text != "" {
			return true
		}
	}
	return false
}

// dropEmpty removes contents carrying no parts, which Gemini rejects.
func dropEmpty(contents []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(contents))
	for _, c := range contents {
		if parts, ok := c[keyParts].([]map[string]any); ok && len(parts) > 0 {
			out = append(out, c)
		}
	}
	return out
}

// toolCallNames maps each tool call in the conversation to the name it was made
// with, so a result can be named after the call it answers rather than after
// whatever it happens to carry itself.
func toolCallNames(msgs []frames.Message) map[string]string {
	names := make(map[string]string)
	for _, m := range msgs {
		for _, c := range m.ToolCalls {
			names[c.ID] = c.Name
		}
	}
	return names
}

// toolCallParts renders an assistant turn's optional preamble and tool calls as
// Gemini parts.
func toolCallParts(text string, calls []frames.ToolCall) []map[string]any {
	parts := make([]map[string]any, 0, len(calls)+1)
	if text != "" {
		parts = append(parts, map[string]any{keyText: text})
	}
	for _, c := range calls {
		args := c.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{keyID: c.ID, keyName: c.Name, "args": args},
		})
	}
	return parts
}

// toolResultParts renders tool outputs as Gemini functionResponse parts. Each
// carries the id of the call it answers, which is what tells apart the results
// of a batch of parallel calls to the same tool.
func toolResultParts(results []frames.ToolResult, names map[string]string) []map[string]any {
	parts := make([]map[string]any, 0, len(results))
	for _, r := range results {
		parts = append(parts, map[string]any{
			"functionResponse": map[string]any{
				keyID:      r.ID,
				keyName:    toolResultName(r, names),
				"response": FunctionResponseDict(r.Content),
			},
		})
	}
	return parts
}

// toolResultName names a result after the call it answers, falling back to the
// name recorded on the result itself and then to a placeholder.
func toolResultName(r frames.ToolResult, names map[string]string) string {
	if name, ok := names[r.ID]; ok && name != "" {
		return name
	}
	if r.Name != "" {
		return r.Name
	}
	return unnamedToolResult
}

// FunctionResponseDict shapes a tool result for Gemini's functionResponse,
// which requires an object: a JSON object passes through, anything else is
// wrapped as {"value": content}.
func FunctionResponseDict(content string) any {
	var obj map[string]any
	if json.Unmarshal([]byte(content), &obj) == nil {
		return json.RawMessage(content)
	}
	return map[string]any{"value": content}
}

// ToProviderToolsFormat implements adapter.LLMAdapter, rendering the standard
// tools as Gemini functionDeclarations and appending the custom tools written
// for Gemini beside them.
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	var out []map[string]any
	if len(schema.Standard) > 0 {
		decls := make([]map[string]any, 0, len(schema.Standard))
		for _, t := range schema.Standard {
			d := map[string]any{keyName: t.Name}
			if t.Description != "" {
				d["description"] = t.Description
			}
			if params := geminiParameters(t.Parameters); params != nil {
				d["parameters"] = params
			}
			decls = append(decls, d)
		}
		out = append(out, map[string]any{"functionDeclarations": decls})
	}
	// A custom tool that cannot be read is left out rather than failing the whole
	// conversion: the tools are advertised alongside a conversation that is
	// otherwise sendable.
	custom, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeGemini)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return out
	}
	return append(out, custom...)
}

// geminiParameters returns the tool's JSON-Schema parameters with
// "additionalProperties" stripped (Gemini rejects it). On a parse error the raw
// schema passes through unchanged.
func geminiParameters(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var schema map[string]any
	if json.Unmarshal(raw, &schema) != nil {
		return raw
	}
	stripAdditionalProperties(schema)
	return schema
}

// stripAdditionalProperties recursively removes "additionalProperties" keys.
func stripAdditionalProperties(v any) {
	switch t := v.(type) {
	case map[string]any:
		delete(t, "additionalProperties")
		for _, val := range t {
			stripAdditionalProperties(val)
		}
	case []any:
		for _, val := range t {
			stripAdditionalProperties(val)
		}
	}
}

// MessagesForLogging implements adapter.LLMAdapter.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	return params.Contents
}
