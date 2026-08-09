// Package responses converts a universal conversation into the request
// OpenAI's Responses API takes, over HTTP and over the WebSocket alike: the two
// transports carry the same request.
package responses

import (
	"encoding/json"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The kinds of entry a Responses request's input list holds.
const (
	ItemMessage    = "message"
	ItemFuncCall   = "function_call"
	ItemFuncOutput = "function_call_output"
)

// The roles the Responses API takes on an input message. It has no system role:
// what would be a system message enters as a developer message, which is the
// role it reserves for instructions given out of band.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleDeveloper = "developer"
)

// toolTypeFunction is the only tool type the Responses API defines.
const toolTypeFunction = "function"

// InputItem is one entry of a Responses request's input list: a message, a
// function call the model made, or the result of one.
type InputItem struct {
	Type string `json:"type"`
	// Role and Content carry a message.
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	// CallID, Name and Arguments carry a function call.
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	// Output carries a function call's result.
	Output string `json:"output,omitempty"`
}

// Tool is a function tool advertised on the request. The Responses API flattens
// the function fields onto the tool rather than nesting them.
type Tool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Params is what one Responses call takes from the conversation: the input list,
// the instructions carried beside it, and the tools to advertise.
type Params struct {
	Input        []InputItem
	Instructions string
	Tools        []Tool
}

// Adapter converts a universal conversation into a Responses request. The zero
// value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, Tool] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "openai" }

// LLMInvocationParams converts the conversation into a Responses request.
//
// The Responses API takes the system prompt as its own field beside the input
// rather than as a leading message, so it has one place to put it: an
// instruction given for this call replaces the conversation's own prompt.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	instructions := a.ResolveSystemInstruction(
		a.SystemWithBuiltins(convo.System()), opts.SystemInstruction, true,
	)
	params := Params{
		Input:        ToInput(convo.Messages()),
		Instructions: instructions,
	}
	// The API requires at least one input item when instructions are given, so a
	// conversation with nothing said yet carries the instruction as a developer
	// message instead of beside an empty list.
	if instructions != "" && len(params.Input) == 0 {
		params.Input = []InputItem{
			{Type: ItemMessage, Role: RoleDeveloper, Content: instructions},
		}
		params.Instructions = ""
	}
	if tools := a.WithBuiltins(convo.Tools()); len(tools) > 0 {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ToInput converts the conversation's messages into the Responses input list. A
// tool turn becomes a function_call item and a function_call_output item
// answering it, paired by the call id.
func ToInput(msgs []frames.Message) []InputItem {
	items := make([]InputItem, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case len(m.ToolResults) > 0:
			for _, r := range m.ToolResults {
				items = append(items, InputItem{
					Type: ItemFuncOutput, CallID: r.ID, Output: r.Content,
				})
			}
		case len(m.ToolCalls) > 0:
			items = append(items, toolCallItems(m)...)
		default:
			items = append(items, InputItem{
				Type: ItemMessage, Role: inputRole(m.Role), Content: m.Text,
			})
		}
	}
	return items
}

// inputRole maps a conversation role to the one the Responses API takes. It has
// no system role, so a system message enters as a developer message: that is
// the role it reserves for instructions given out of band, which is what a
// system message is.
func inputRole(role frames.Role) string {
	if role == frames.RoleSystem || role == frames.RoleDeveloper {
		return RoleDeveloper
	}
	if role == frames.RoleAssistant {
		return RoleAssistant
	}
	return RoleUser
}

// toolCallItems renders an assistant turn's optional preamble and tool calls.
func toolCallItems(m frames.Message) []InputItem {
	items := make([]InputItem, 0, len(m.ToolCalls)+1)
	if m.Text != "" {
		items = append(items, InputItem{
			Type: ItemMessage, Role: RoleAssistant, Content: m.Text,
		})
	}
	for _, call := range m.ToolCalls {
		args := string(call.Args)
		if args == "" {
			args = "{}"
		}
		items = append(items, InputItem{
			Type: ItemFuncCall, CallID: call.ID, Name: call.Name, Arguments: args,
		})
	}
	return items
}

// ToProviderToolsFormat implements adapter.LLMAdapter.
func (*Adapter) ToProviderToolsFormat(tools []frames.Tool) []Tool {
	out := make([]Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, Tool{
			Type:        toolTypeFunction,
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	return out
}

// MessagesForLogging implements adapter.LLMAdapter.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(params.Input))
	for _, item := range params.Input {
		raw, err := json.Marshal(item)
		if err != nil {
			continue
		}
		var got map[string]any
		if err := json.Unmarshal(raw, &got); err != nil {
			continue
		}
		out = append(out, got)
	}
	return out
}
