// Package responses converts a universal conversation into the request
// OpenAI's Responses API takes, over HTTP and over the WebSocket alike: the two
// transports carry the same request.
package responses

import (
	"encoding/json"
	"log/slog"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The kinds of entry a Responses request's input list holds.
const (
	ItemMessage    = "message"
	ItemFuncCall   = "function_call"
	ItemFuncOutput = "function_call_output"
	ItemReasoning  = "reasoning"
)

// SummaryTypeText is the type of a reasoning summary's text part.
const SummaryTypeText = "summary_text"

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
// function call the model made, the result of one, or the model's reasoning
// before a reply.
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
	// ID, Summary and EncryptedContent carry a reasoning item: the id the
	// server gave it, the summary of the reasoning, and the reasoning itself,
	// encrypted, which is what carries it across turns.
	ID               string        `json:"id,omitempty"`
	Summary          []SummaryPart `json:"summary,omitempty"`
	EncryptedContent string        `json:"encrypted_content,omitempty"`
}

// SummaryPart is one part of a reasoning item's summary.
type SummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// MarshalJSON encodes the item. A reasoning item always carries its summary,
// empty or not, since the API requires one; its encrypted content is left off
// when there is none rather than sent empty.
func (i InputItem) MarshalJSON() ([]byte, error) {
	// plain drops the method set, so marshaling it does not recurse.
	type plain InputItem
	if i.Type != ItemReasoning {
		return json.Marshal(plain(i))
	}
	summary := i.Summary
	if summary == nil {
		summary = []SummaryPart{}
	}
	return json.Marshal(struct {
		Type             string        `json:"type"`
		ID               string        `json:"id"`
		Summary          []SummaryPart `json:"summary"`
		EncryptedContent string        `json:"encrypted_content,omitempty"`
	}{i.Type, i.ID, summary, i.EncryptedContent})
}

// NewReasoningMessage is a reasoning item the model produced, as a message of
// this adapter's own kind: appended to the conversation, it goes back to the
// model as a reasoning input item placed before the reply it led to, which is
// how the model keeps its reasoning across turns.
func NewReasoningMessage(id string, summary []SummaryPart, encryptedContent string) frames.Message {
	return frames.NewLLMSpecificMessage(LLMSpecificID, InputItem{
		Type: ItemReasoning, ID: id, Summary: summary, EncryptedContent: encryptedContent,
	})
}

// LLMSpecificID is the id the adapter's own messages are tagged with.
const LLMSpecificID = "openai_responses"

// Tool is a tool advertised on the request. The Responses API flattens the
// function fields onto the tool rather than nesting them. A tool the provider
// implements itself, which this schema has no place for, is carried whole in
// Raw.
type Tool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	// Raw is the tool exactly as the provider takes it, and replaces the modeled
	// fields entirely when set. It is how a provider-native tool (a hosted search
	// tool, say, which the model runs itself rather than calling back for) is
	// advertised without being forced into the function shape.
	Raw map[string]any `json:"-"`
}

// MarshalJSON encodes the tool, sending Raw in place of the modeled fields when
// it is set.
func (t Tool) MarshalJSON() ([]byte, error) {
	if len(t.Raw) > 0 {
		return json.Marshal(t.Raw)
	}
	// plain drops the method set, so marshaling it does not recurse.
	type plain Tool
	return json.Marshal(plain(t))
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

// IDForLLMSpecificMessages implements adapter.LLMAdapter. It is not the chat
// adapter's id: the two OpenAI APIs take different native messages, so a
// conversation shared by both services keeps each one's messages to itself.
func (*Adapter) IDForLLMSpecificMessages() string { return LLMSpecificID }

// LLMInvocationParams converts the conversation into a Responses request.
//
// The conversation's own system prompt stays in the input, as a developer
// message at its head. The instruction given for the call (the service's
// composed system instruction) goes in the request's instructions field beside
// it, so both reach the model. The only exception is a conversation with
// nothing in it at all: the API requires at least one input item when
// instructions are given, so the instruction becomes that item instead.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	system := convo.System()
	// Only the conflict is reported here: both are sent either way.
	a.ResolveSystemInstruction(system, opts.SystemInstruction, false)

	msgs := convo.MessagesFor(a.IDForLLMSpecificMessages())
	converted, err := ToInput(msgs)
	if err != nil {
		return Params{}, err
	}
	input := make([]InputItem, 0, len(converted)+1)
	if system != "" {
		input = append(input, InputItem{Type: ItemMessage, Role: RoleDeveloper, Content: system})
	}
	params := Params{Input: append(input, converted...)}
	if instruction := opts.SystemInstruction; instruction != "" {
		if len(params.Input) == 0 {
			params.Input = []InputItem{
				{Type: ItemMessage, Role: RoleDeveloper, Content: instruction},
			}
		} else {
			params.Instructions = instruction
		}
	}
	if tools := a.WithBuiltins(convo.ToolsSchema()); len(tools.Standard) > 0 ||
		len(tools.Custom) > 0 {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ToInput converts the conversation's messages into the Responses input list. A
// tool turn becomes a function_call item and a function_call_output item
// answering it, paired by the call id.
func ToInput(msgs []frames.Message) ([]InputItem, error) {
	items := make([]InputItem, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case m.IsLLMSpecific():
			native, err := adapter.NativeMessage[InputItem](m)
			if err != nil {
				return nil, err
			}
			items = append(items, native)
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
	return items, nil
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
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []Tool {
	out := make([]Tool, 0, len(schema.Standard))
	for _, t := range schema.Standard {
		out = append(out, Tool{
			Type:        toolTypeFunction,
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	custom, err := adapter.CustomToolsFor[Tool](schema, frames.AdapterTypeOpenAI)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return out
	}
	return append(out, custom...)
}

// MessagesForLogging implements adapter.LLMAdapter. A reasoning item's
// encrypted content is opaque noise in a log, so it is elided.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(params.Input))
	for _, item := range params.Input {
		if item.Type == ItemReasoning && item.EncryptedContent != "" {
			item.EncryptedContent = "..."
		}
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
