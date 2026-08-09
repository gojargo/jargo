// Package openai converts a universal conversation into the chat-completions
// request OpenAI takes, and with it every endpoint that speaks OpenAI's API.
//
// The types here are the wire format itself, so a provider that departs from
// OpenAI in some detail can embed this adapter and rewrite what it produces
// rather than convert the conversation a second time.
package openai

import (
	"encoding/json"
	"maps"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The message roles the chat-completions API defines.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	RoleDeveloper = "developer"
)

// toolTypeFunction is the only tool type OpenAI's chat API defines.
const toolTypeFunction = "function"

// ContentPartText is the only content part type this format needs today.
const ContentPartText = "text"

// ContentPart is one part of a message's content. The API takes content either
// as a plain string or as a list of parts.
type ContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// TextPart builds a text content part.
func TextPart(text string) ContentPart {
	return ContentPart{Type: ContentPartText, Text: text}
}

// Message is one message of the conversation as the chat-completions API takes
// it.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	// ContentParts carries the content as a list of parts instead of a plain
	// string, and replaces Content when it is set. It is what lets an endpoint
	// that takes two same-role messages as one keep what each said distinct,
	// rather than running their text together.
	ContentParts []ContentPart `json:"-"`
	// Extra sets fields on this message that OpenAI's schema has no place for,
	// merged over the modeled ones when the message is encoded. It is how an
	// endpoint that reads a field of its own is served without that field
	// reaching the endpoints that would reject it.
	Extra map[string]any `json:"-"`
}

// MarshalJSON encodes the message, sending the content as parts when it has
// them and merging Extra over the modeled fields.
func (m Message) MarshalJSON() ([]byte, error) {
	// plain drops the method set, so marshaling it does not recurse.
	type plain Message
	if len(m.Extra) == 0 && len(m.ContentParts) == 0 {
		return json.Marshal(plain(m))
	}
	over := map[string]any{}
	if len(m.ContentParts) > 0 {
		over["content"] = m.ContentParts
	}
	// Extra is the explicit override, so it wins over the parts.
	maps.Copy(over, m.Extra)
	return MergeExtra(plain(m), over)
}

// Parts returns the message's content as parts, whichever way it carries it. A
// plain string becomes the single text part it stands for.
func (m Message) Parts() []ContentPart {
	if len(m.ContentParts) > 0 {
		return m.ContentParts
	}
	return []ContentPart{TextPart(m.Content)}
}

// DemoteLateSystem sends every system message past the leading run as a user
// message. Several endpoints that otherwise speak this API accept a system
// message only at the start of a conversation and reject one that follows any
// other message.
func DemoteLateSystem(msgs []Message) {
	for i := range msgs {
		if msgs[i].Role == RoleSystem {
			continue
		}
		for j := i; j < len(msgs); j++ {
			if msgs[j].Role == RoleSystem {
				msgs[j].Role = RoleUser
			}
		}
		return
	}
}

// ToolCall is an assistant tool-call entry on a message.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the function a tool call invokes, with its arguments as
// the raw JSON string the model produced.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool is a function tool advertised on the request.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function describes the tool a model may call.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Params is what one chat-completions call takes from the conversation: the
// messages to send, the tools to advertise, and whether the model may or must
// call one. Everything else on the request is the service's own configuration.
type Params struct {
	Messages   []Message
	Tools      []Tool
	ToolChoice string
}

// MergeExtra encodes v and merges extra over the fields it produced, so a
// caller-supplied field wins over the modeled one of the same name. It is how
// both a message and a whole request carry fields OpenAI's schema has no place
// for.
func MergeExtra(v any, extra map[string]any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	maps.Copy(m, extra)
	return json.Marshal(m)
}

// Adapter converts a universal conversation into an OpenAI chat-completions
// request. The zero value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, Tool] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "openai" }

// LLMInvocationParams converts the conversation into a chat-completions
// request.
//
// OpenAI takes the system prompt as a leading message rather than beside the
// conversation, so an instruction given for this call and the conversation's own
// prompt can both be sent, and both are: an instruction supplements the
// conversation's prompt rather than silently replacing it.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	system := a.SystemWithBuiltins(convo.System())
	// Only the conflict is resolved here, not the prompt's position: OpenAI
	// carries the conversation's prompt in the messages either way.
	instruction := a.ResolveSystemInstruction(system, opts.SystemInstruction, false)

	msgs := convo.Messages()
	out := make([]Message, 0, len(msgs)+2)
	if instruction != "" {
		out = append(out, Message{Role: RoleSystem, Content: instruction})
	}
	if system != "" {
		out = append(out, Message{Role: RoleSystem, Content: system})
	}
	out = append(out, ToMessages(msgs, opts.ConvertDeveloperToUser)...)

	params := Params{Messages: out}
	if tools := a.WithBuiltins(convo.Tools()); len(tools) > 0 {
		params.Tools = a.ToProviderToolsFormat(tools)
		params.ToolChoice = string(convo.ToolChoice())
	}
	return params, nil
}

// ToMessages converts the conversation's messages into chat-completions
// messages. A tool turn becomes an assistant message carrying tool_calls and one
// "tool" message per result.
//
// convertDeveloperToUser sends a developer message as a user message, for an
// endpoint that has no developer role.
//
// It is exported so an adapter that embeds this one can convert the conversation
// once and then rewrite what came out.
func ToMessages(msgs []frames.Message, convertDeveloperToUser bool) []Message {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case len(m.ToolResults) > 0:
			for _, r := range m.ToolResults {
				out = append(out, Message{Role: RoleTool, ToolCallID: r.ID, Content: r.Content})
			}
		case len(m.ToolCalls) > 0:
			out = append(out, assistantToolCalls(m))
		default:
			role := string(m.Role)
			if convertDeveloperToUser && m.Role == frames.RoleDeveloper {
				role = RoleUser
			}
			out = append(out, Message{Role: role, Content: m.Text})
		}
	}
	return out
}

// assistantToolCalls renders an assistant turn that requested tool calls.
func assistantToolCalls(m frames.Message) Message {
	msg := Message{Role: RoleAssistant, Content: m.Text}
	for _, c := range m.ToolCalls {
		args := string(c.Args)
		if args == "" {
			args = "{}"
		}
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:       c.ID,
			Type:     toolTypeFunction,
			Function: ToolCallFunction{Name: c.Name, Arguments: args},
		})
	}
	return msg
}

// ToProviderToolsFormat implements adapter.LLMAdapter.
func (*Adapter) ToProviderToolsFormat(tools []frames.Tool) []Tool {
	out := make([]Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, Tool{
			Type: toolTypeFunction,
			Function: Function{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return out
}

// MessagesForLogging implements adapter.LLMAdapter. It renders the conversation
// as the endpoint will see it, so a trace carries the prompt that was actually
// sent rather than the universal form it was converted from.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	return asMaps(params.Messages)
}

// asMaps renders messages through their own encoding, so what is logged is the
// shape that goes on the wire rather than a second rendering that could drift
// from it.
func asMaps(msgs []Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		raw, err := json.Marshal(m)
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
