package realtime

import (
	"encoding/json"
	"strings"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// prismAdapter converts the conversation and its tools to what a Prism session
// takes. It submits no live turn and runs no tool.
type prismAdapter struct {
	adapter.Base
}

// messageText is the text of a message, which has to be all its content.
func messageText(m frames.Message) (string, error) {
	if len(m.Content) == 0 {
		return m.Text, nil
	}
	var text strings.Builder
	for _, part := range m.Content {
		if part.Type != frames.ContentPartText {
			return "", settingsError("Prism history supports text content only")
		}
		text.WriteString(part.Text)
	}
	return text.String(), nil
}

// historyBuilder validates a whole history as it converts it, completed and
// ordered tool pairs included.
type historyBuilder struct {
	items   []historyItem
	calls   map[string]bool
	pending map[string]bool
}

func (b *historyBuilder) addCalls(m frames.Message, role string) error {
	for _, call := range m.ToolCalls {
		if role != string(frames.RoleAssistant) {
			return settingsError("Prism supports assistant function calls only")
		}
		if call.ID == "" || b.calls[call.ID] {
			return settingsError("Imported calls require unique call IDs")
		}
		if call.Name == "" {
			return settingsError("Imported calls require a name and string arguments")
		}
		arguments := string(call.Args)
		if arguments == "" {
			arguments = "{}"
		}
		b.items = append(b.items, historyItem{
			kind: itemFunctionCall, callID: call.ID, name: call.Name, arguments: arguments,
		})
		b.calls[call.ID], b.pending[call.ID] = true, true
	}
	return nil
}

func (b *historyBuilder) addResults(m frames.Message) error {
	for _, result := range m.ToolResults {
		if !b.pending[result.ID] {
			return settingsError("Imported results must follow their calls exactly once")
		}
		delete(b.pending, result.ID)
		b.items = append(b.items, historyItem{kind: itemFunctionOut, callID: result.ID, output: result.Content})
	}
	return nil
}

func (b *historyBuilder) add(m frames.Message) error {
	if m.IsLLMSpecific() {
		return settingsError("Prism does not support opaque provider history")
	}
	if len(m.ToolResults) > 0 {
		return b.addResults(m)
	}
	role := string(m.Role)
	if m.Role == frames.RoleDeveloper {
		role = string(frames.RoleSystem)
	}
	if role != "user" && role != "assistant" && role != "system" {
		return settingsError("Prism does not support history role %q", role)
	}
	text, err := messageText(m)
	if err != nil {
		return err
	}
	if text != "" || len(m.ToolCalls) == 0 {
		b.items = append(b.items, historyItem{kind: itemMessage, role: role, text: text})
	}
	return b.addCalls(m, role)
}

// historyItems converts messages to history items, refusing a history Prism
// cannot hold: one with content that is not text, or a tool call with no
// result.
func historyItems(messages []frames.Message) ([]historyItem, error) {
	b := &historyBuilder{calls: map[string]bool{}, pending: map[string]bool{}}
	for _, m := range messages {
		if err := b.add(m); err != nil {
			return nil, err
		}
	}
	if len(b.pending) > 0 {
		return nil, settingsError("Cannot import pending tool calls into Prism")
	}
	return b.items, nil
}

// toolDefinitionOf converts a standard tool.
func toolDefinitionOf(t frames.Tool) (toolDefinition, error) {
	parameters := map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}}
	if len(t.Parameters) > 0 {
		parameters = nil
		if err := json.Unmarshal(t.Parameters, &parameters); err != nil || parameters == nil {
			return toolDefinition{}, settingsError("Prism tool parameters must be a JSON object")
		}
	}
	return toolDefinition{name: t.Name, description: t.Description, parameters: parameters}, nil
}

// tools converts a toolset, with the tools the service implements itself
// added.
func (a *prismAdapter) tools(schema frames.ToolsSchema) ([]toolDefinition, error) {
	if len(schema.Custom) > 0 {
		return nil, settingsError("Prism does not support provider-native tool overrides")
	}
	names := map[string]bool{}
	values := make([]toolDefinition, 0, len(schema.Standard))
	for _, t := range schema.Standard {
		if names[t.Name] {
			return nil, settingsError("Prism tool names must be unique")
		}
		definition, err := toolDefinitionOf(t)
		if err != nil {
			return nil, err
		}
		names[t.Name] = true
		values = append(values, definition)
	}
	for _, t := range a.WithBuiltins(frames.ToolsSchema{}).Standard {
		if names[t.Name] {
			continue
		}
		definition, err := toolDefinitionOf(t)
		if err != nil {
			return nil, err
		}
		names[t.Name] = true
		values = append(values, definition)
	}
	return values, nil
}
