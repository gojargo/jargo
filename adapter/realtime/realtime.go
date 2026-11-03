// Package realtime converts a universal conversation into what OpenAI's
// Realtime API takes on a session: the system instruction, the conversation
// items to seed it with, and the tools it carries.
package realtime

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// toolTypeFunction is the type of a function tool.
const toolTypeFunction = "function"

// The text wrapped around a conversation packed into one message, telling the
// model what it is reading and what to do once it has.
const (
	packedIntro = `
        This is a previously saved conversation. Please treat this conversation history as a
        starting point for the current conversation.`
	packedTrailer = `
        This is the end of the previously saved conversation. Please continue the conversation
        from here. If the last message is a user instruction or question, act on that instruction
        or answer the question. If the last message is an assistant response, simple say that you
        are ready to continue the conversation.`
)

// Params is what a session takes from the conversation: its system
// instruction, the items to seed it with, and its tools.
type Params struct {
	// SystemInstruction is the instruction in effect, empty when there is none.
	SystemInstruction string
	// Messages are the items to seed the session with.
	Messages []events.ConversationItem
	// Tools are the tools the session carries, in the API's form.
	Tools []map[string]any
}

// Adapter converts a universal conversation into Realtime session parameters.
// The zero value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "openai-realtime" }

// LLMInvocationParams converts the conversation into what a session takes from
// it.
//
// The system instruction given in opts wins over the conversation's own prompt,
// the API having one place for it. The tools are the conversation's own when it
// advertises any, else the ones in opts.ServiceTools; the built-in tools ride
// along with whichever set is in use, and are the set when there is neither.
func (a *Adapter) LLMInvocationParams(convo *frames.LLMContext, opts adapter.Options) (Params, error) {
	items, fromContext, err := a.fromMessages(convo.System(), convo.MessagesFor(a.IDForLLMSpecificMessages()))
	if err != nil {
		return Params{}, err
	}
	return Params{
		SystemInstruction: a.ResolveSystemInstruction(fromContext, opts.SystemInstruction, true),
		Messages:          items,
		Tools:             a.ToProviderToolsFormat(a.RealtimeSessionTools(convo.ToolsSchema(), opts.ServiceTools)),
	}, nil
}

// fromMessages converts the conversation's prompt and messages into the items a
// session is seeded with, returning the prompt as the instruction it carries.
//
// A long history cannot be loaded into the API item by item: given a series of
// items, the model forgets it can speak. So a single user message goes in as it
// stands, and anything more is packed into one user message holding the whole
// history, with a word to the model on what it is reading. Messages written for
// another provider, or for this one, are left out: a packed history cannot carry
// them.
func (a *Adapter) fromMessages(system string, all []frames.Message) ([]events.ConversationItem, string, error) {
	msgs := make([]frames.Message, 0, len(all))
	for _, m := range all {
		if m.IsLLMSpecific() {
			continue
		}
		// The API takes no system or developer messages, so they are the user's.
		if m.Role == frames.RoleSystem || m.Role == frames.RoleDeveloper {
			m.Role = frames.RoleUser
		}
		msgs = append(msgs, m)
	}
	if len(msgs) == 0 {
		return nil, system, nil
	}
	if len(msgs) == 1 && msgs[0].Role == frames.RoleUser && len(msgs[0].ToolResults) == 0 {
		item, err := fromMessage(msgs[0])
		if err != nil {
			return nil, "", err
		}
		return []events.ConversationItem{item}, system, nil
	}

	history, err := openai.ToMessages(msgs, false)
	if err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(history); err != nil {
		return nil, "", err
	}
	item := events.NewConversationItem(events.ItemMessage)
	item.Role = string(frames.RoleUser)
	item.Content = []events.ItemContent{{
		Type: "input_text",
		Text: strings.Join([]string{packedIntro, strings.TrimSuffix(buf.String(), "\n"), packedTrailer}, "\n\n"),
	}}
	return []events.ConversationItem{item}, system, nil
}

// fromMessage converts one message into an item: a user message, or the first
// call of an assistant message calling functions.
func fromMessage(m frames.Message) (events.ConversationItem, error) {
	switch {
	case m.Role == frames.RoleUser:
		item := events.NewConversationItem(events.ItemMessage)
		item.Role = string(frames.RoleUser)
		item.Content = []events.ItemContent{{Type: "input_text", Text: m.Text}}
		return item, nil
	case m.Role == frames.RoleAssistant && len(m.ToolCalls) > 0:
		call := m.ToolCalls[0]
		item := events.NewConversationItem(events.ItemFunctionCall)
		item.CallID, item.Name, item.Arguments = call.ID, call.Name, string(call.Args)
		return item, nil
	default:
		return events.ConversationItem{}, &adapter.ConversionError{Cause: &unhandledMessageError{role: string(m.Role)}}
	}
}

// unhandledMessageError reports a message the API has no item for.
type unhandledMessageError struct{ role string }

// Error implements error.
func (e *unhandledMessageError) Error() string {
	return "realtime: unhandled message with role " + e.role
}

// ToProviderToolsFormat implements adapter.LLMAdapter. The Realtime API
// flattens a function's fields onto the tool, and reads the custom tools written
// for OpenAI like the other OpenAI APIs.
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	out := make([]map[string]any, 0, len(schema.Standard))
	for _, t := range schema.Standard {
		out = append(out, functionFormat(t))
	}
	custom, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeOpenAI)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return out
	}
	return append(out, custom...)
}

// functionFormat renders a tool as a Realtime function: its name, description,
// and the object schema of its parameters.
func functionFormat(t frames.Tool) map[string]any {
	var params struct {
		Properties map[string]any `json:"properties"`
		Required   []string       `json:"required"`
	}
	if len(t.Parameters) > 0 {
		if err := json.Unmarshal(t.Parameters, &params); err != nil {
			slog.Error("a tool's parameters are not a JSON object schema", "tool", t.Name, "err", err)
		}
	}
	if params.Properties == nil {
		params.Properties = map[string]any{}
	}
	if params.Required == nil {
		params.Required = []string{}
	}
	return map[string]any{
		"type":        toolTypeFunction,
		"name":        t.Name,
		"description": t.Description,
		"parameters": map[string]any{
			"type":       "object",
			"properties": params.Properties,
			"required":   params.Required,
		},
	}
}

// MessagesForLogging implements adapter.LLMAdapter. It renders the
// conversation's messages in the universal form, as the session receives them
// before they are packed.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	msgs, err := openai.ToMessages(convo.MessagesFor(a.IDForLLMSpecificMessages()), false)
	if err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		raw, err := json.Marshal(m)
		if err != nil {
			continue
		}
		var got map[string]any
		if json.Unmarshal(raw, &got) == nil {
			out = append(out, got)
		}
	}
	return out
}
