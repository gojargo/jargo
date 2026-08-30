package bus

import (
	"fmt"

	"github.com/gojargo/jargo/frames"
)

// The adapters for the types the serializer cannot take apart itself.
//
// A conversation keeps its state private behind a lock, so its fields are
// unreachable by reflection and it is rebuilt through its own methods instead.
// Its toolset travels with it: the tools written in one provider's own format
// are values only that provider's adapter understands, so they cross as the
// JSON they already are.

// frameContextSample names the conversation type for registration. The adapter
// is registered against the type, and this is how the type is named.
type frameContextSample = frames.LLMContext

// contextAdapter converts a conversation to a map and back.
type contextAdapter struct{}

// Serialize writes the parts of a conversation that make it up: what was said,
// what the model may call, and whether it must.
//
// The system prompt is written separately from the messages, because a
// conversation holds it apart from them: rebuilding one from its messages alone
// would lose it.
func (contextAdapter) Serialize(obj any, serialize SerializeFunc) (map[string]any, error) {
	c, ok := obj.(*frames.LLMContext)
	if !ok {
		return nil, fmt.Errorf("%w: the conversation adapter was given a %T", ErrWrongAdapter, obj)
	}

	messages, err := serialize(c.Messages())
	if err != nil {
		return nil, err
	}
	out := map[string]any{"messages": messages}

	if system := c.System(); system != "" {
		out["system"] = system
	}

	schema := c.ToolsSchema()
	if len(schema.Standard) > 0 || len(schema.Custom) > 0 {
		tools, err := toolsAdapter{}.Serialize(&schema, serialize)
		if err != nil {
			return nil, err
		}
		out["tools"] = tools
	}
	if choice := c.ToolChoice(); choice != "" {
		out["tool_choice"] = string(choice)
	}
	return out, nil
}

// Deserialize rebuilds a conversation through its own methods, which is the
// only way in: its state is private.
//
// A key the sender left out is left alone rather than being written as a zero,
// so a conversation that advertised no tools comes back advertising none rather
// than advertising an empty set.
func (contextAdapter) Deserialize(data map[string]any, deserialize DeserializeFunc) (any, error) {
	system, _ := data["system"].(string)
	c := frames.NewLLMContext(system)

	if raw, present := data["messages"]; present {
		msgs, err := decodeInto[[]frames.Message](raw, deserialize)
		if err != nil {
			return nil, fmt.Errorf("bus: the conversation's messages: %w", err)
		}
		c.SetMessages(msgs)
	}
	if raw, present := data["tools"]; present {
		fields, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: the conversation's tools carried %T, want an object", ErrWrongShape, raw)
		}
		schema, err := toolsAdapter{}.Deserialize(fields, deserialize)
		if err != nil {
			return nil, err
		}
		built, ok := schema.(*frames.ToolsSchema)
		if !ok {
			return nil, fmt.Errorf("%w: the toolset adapter built a %T", ErrWrongAdapter, schema)
		}
		c.SetToolsSchema(*built)
	}
	if choice, ok := data["tool_choice"].(string); ok {
		c.SetToolChoice(frames.ToolChoice(choice))
	}
	return c, nil
}

// toolsAdapter converts a toolset to a map and back.
//
// A tool's handler and the resource it works through are deliberately not
// written: they are running code, which the far end could not have used, and a
// tool arriving over a bus is advertise-only there.
type toolsAdapter struct{}

// Serialize writes the standard tools and, when there are any, the ones written
// in a provider's own format.
func (toolsAdapter) Serialize(obj any, serialize SerializeFunc) (map[string]any, error) {
	s, ok := obj.(*frames.ToolsSchema)
	if !ok {
		return nil, fmt.Errorf("%w: the toolset adapter was given a %T", ErrWrongAdapter, obj)
	}

	standard := make([]any, 0, len(s.Standard))
	for _, t := range s.Standard {
		standard = append(standard, map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"parameters":  string(t.Parameters),
		})
	}
	out := map[string]any{"standard_tools": standard}

	if len(s.Custom) > 0 {
		custom := map[string]any{}
		for kind, tools := range s.Custom {
			written, err := serialize(tools)
			if err != nil {
				return nil, err
			}
			custom[string(kind)] = written
		}
		out["custom_tools"] = custom
	}
	return out, nil
}

// Deserialize rebuilds a toolset. The tools come back advertise-only, without
// the handler the sending side may have had.
func (toolsAdapter) Deserialize(data map[string]any, deserialize DeserializeFunc) (any, error) {
	out := &frames.ToolsSchema{}

	if raw, present := data["standard_tools"]; present {
		listed, ok := raw.([]any)
		if !ok {
			return nil, fmt.Errorf("%w: standard_tools carried %T, want a list", ErrWrongShape, raw)
		}
		for _, item := range listed {
			fields, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("%w: a tool carried %T, want an object", ErrWrongShape, item)
			}
			tool := frames.Tool{}
			tool.Name, _ = fields["name"].(string)
			tool.Description, _ = fields["description"].(string)
			if params, ok := fields["parameters"].(string); ok && params != "" {
				tool.Parameters = []byte(params)
			}
			out.Standard = append(out.Standard, tool)
		}
	}

	if raw, present := data["custom_tools"]; present {
		fields, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: custom_tools carried %T, want an object", ErrWrongShape, raw)
		}
		out.Custom = map[frames.AdapterType][]any{}
		for kind, tools := range fields {
			listed, err := decodeInto[[]any](tools, deserialize)
			if err != nil {
				return nil, err
			}
			out.Custom[frames.AdapterType(kind)] = listed
		}
	}
	return out, nil
}

// decodeInto rebuilds raw and converts it to T, which is what an adapter needs
// when the value it is filling is a concrete Go type rather than a map.
func decodeInto[T any](raw any, deserialize DeserializeFunc) (T, error) {
	var zero T
	v, err := deserialize(raw)
	if err != nil {
		return zero, err
	}
	if v == nil {
		return zero, nil
	}
	out, err := reencode[T](v)
	if err != nil {
		return zero, err
	}
	return out, nil
}
