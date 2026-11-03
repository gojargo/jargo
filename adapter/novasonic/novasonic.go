// Package novasonic converts a universal conversation into what Amazon Nova
// Sonic takes on a session: the system instruction, the conversation history to
// seed it with, and the tools it carries.
package novasonic

import (
	"encoding/json"
	"log/slog"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// Role is who a Nova Sonic content block comes from.
type Role string

// The roles Nova Sonic takes. A conversation's history only carries USER and
// ASSISTANT; SYSTEM is the system instruction and TOOL a tool result.
const (
	RoleSystem    Role = "SYSTEM"
	RoleUser      Role = "USER"
	RoleAssistant Role = "ASSISTANT"
	RoleTool      Role = "TOOL"
)

// HistoryMessage is one message of the history a session is seeded with.
type HistoryMessage struct {
	// Role is USER or ASSISTANT.
	Role Role `json:"role"`
	// Text is what was said.
	Text string `json:"text"`
}

// Params is what a session takes from the conversation: its system
// instruction, the history to seed it with, and its tools.
type Params struct {
	// SystemInstruction is the instruction in effect, empty when there is none.
	SystemInstruction string
	// Messages are the history to seed the session with.
	Messages []HistoryMessage
	// Tools are the tools the session carries, in Nova Sonic's form.
	Tools []map[string]any
}

// Adapter converts a universal conversation into Nova Sonic session
// parameters. The zero value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "aws-nova-sonic" }

// LLMInvocationParams converts the conversation into what a session takes from
// it.
//
// The system instruction given in opts wins over the conversation's own prompt.
// The tools are the conversation's own when it advertises any, else the ones in
// opts.ServiceTools; the built-in tools ride along with whichever set is in
// use, and are the set when there is neither.
func (a *Adapter) LLMInvocationParams(convo *frames.LLMContext, opts adapter.Options) (Params, error) {
	history := a.fromMessages(convo.MessagesFor(a.IDForLLMSpecificMessages()))
	tools := a.ToProviderToolsFormat(a.RealtimeSessionTools(convo.ToolsSchema(), opts.ServiceTools))
	if tools == nil {
		tools = []map[string]any{}
	}
	return Params{
		SystemInstruction: a.ResolveSystemInstruction(convo.System(), opts.SystemInstruction, true),
		Messages:          history,
		Tools:             tools,
	}, nil
}

// fromMessages converts the conversation's messages into the history a session
// is seeded with.
//
// Nova Sonic's history only takes the user's and the assistant's words, so
// system and developer messages become the user's, and what has no words to
// carry is left out: an assistant message that only calls tools, a tool result.
// Messages written for a provider, this one included, are left out too.
func (a *Adapter) fromMessages(all []frames.Message) []HistoryMessage {
	var out []HistoryMessage
	for _, m := range all {
		if m.IsLLMSpecific() {
			continue
		}
		role := m.Role
		if role == frames.RoleSystem || role == frames.RoleDeveloper {
			role = frames.RoleUser
		}
		if msg, ok := fromMessage(role, m); ok {
			out = append(out, msg)
		}
	}
	return out
}

// fromMessage converts one message into a history message, reporting false for
// one the history has no place for.
func fromMessage(role frames.Role, m frames.Message) (HistoryMessage, bool) {
	if role != frames.RoleUser && role != frames.RoleAssistant {
		return HistoryMessage{}, false
	}
	if m.Text == "" {
		return HistoryMessage{}, false
	}
	r := RoleUser
	if role == frames.RoleAssistant {
		r = RoleAssistant
	}
	return HistoryMessage{Role: r, Text: m.Text}, true
}

// MessagesForLogging implements adapter.LLMAdapter. It renders the history a
// session would be seeded with.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	history := a.fromMessages(convo.MessagesFor(a.IDForLLMSpecificMessages()))
	out := make([]map[string]any, 0, len(history))
	for _, m := range history {
		out = append(out, map[string]any{"role": string(m.Role), "text": m.Text})
	}
	return out
}

// ToProviderToolsFormat implements adapter.LLMAdapter. Nova Sonic declares a
// function as a toolSpec whose input schema is the JSON text of an object
// schema rather than the schema itself. Only standard tools are converted.
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	out := make([]map[string]any, 0, len(schema.Standard))
	for _, t := range schema.Standard {
		out = append(out, functionFormat(t))
	}
	return out
}

// functionFormat renders a tool as a Nova Sonic toolSpec.
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
	schema, err := json.Marshal(map[string]any{
		"type":       "object",
		"properties": params.Properties,
		"required":   params.Required,
	})
	if err != nil {
		slog.Error("a tool's parameters cannot be encoded", "tool", t.Name, "err", err)
	}
	return map[string]any{
		"toolSpec": map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"inputSchema": map[string]any{"json": string(schema)},
		},
	}
}
