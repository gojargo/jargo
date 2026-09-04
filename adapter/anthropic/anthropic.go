// Package anthropic converts a universal conversation into the request
// Anthropic takes, and with it Bedrock, which serves the same models.
package anthropic

import (
	"encoding/json"
	"errors"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// emptyText stands in for a message whose content came out empty. Anthropic
// rejects a message with no content, and a turn that produced none is still
// part of the conversation.
const emptyText = "(empty)"

// noOpUserTurn is the minimal user message appended to a conversation ending on
// an assistant message. A full stop says nothing in any language, which is the
// point: it is there to satisfy the shape, not to be read.
const noOpUserTurn = "."

// Params is what one Anthropic call takes from the conversation: the system
// prompt it carries beside the messages, the messages themselves, and the tools
// to advertise.
type Params struct {
	System   []sdk.TextBlockParam
	Messages []sdk.MessageParam
	Tools    []sdk.ToolUnionParam
}

// Adapter converts a universal conversation into an Anthropic request. The zero
// value is ready to use.
type Adapter struct {
	adapter.Base
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, sdk.ToolUnionParam] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "anthropic" }

// LLMInvocationParams converts the conversation into an Anthropic request.
//
// Anthropic carries the system prompt beside the conversation rather than in
// it, so it has one field to put it in: an instruction given for this call
// replaces the conversation's own prompt rather than supplementing it. It has
// neither a system nor a developer input role either, so a message in one of
// those roles enters as the user.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	fromContext, msgs := a.ExtractInitialSystem(
		convo.System(), opts.SystemInstruction,
		convo.MessagesFor(a.IDForLLMSpecificMessages()),
	)

	converted, err := ToMessages(msgs)
	if err != nil {
		return Params{}, err
	}
	if opts.EnsureLastMessageIsUser {
		converted = EnsureLastMessageIsUser(converted)
	}
	if opts.EnablePromptCaching {
		converted = WithCacheControlMarkers(converted)
	}

	params := Params{
		Messages: converted,
		System:   a.systemBlocks(convo, fromContext, opts),
	}
	if tools := a.WithBuiltins(convo.ToolsSchema()); len(tools.Standard) > 0 ||
		len(tools.Custom) > 0 {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// systemBlocks renders the system prompt Anthropic is sent beside the
// conversation.
//
// When the prompt comes from the conversation and caching is on, the cache
// breakpoint goes on the part of it that survives between turns. A cached
// prefix is only reused while it stays byte-identical, and everything after the
// breakpoint is free to vary without disturbing it. The recalled context a
// memory service refreshes every turn is exactly that: putting it inside the
// breakpoint would rewrite the cache on every request and never read one back,
// which costs more than not caching at all.
func (a *Adapter) systemBlocks(
	convo *frames.LLMContext, fromContext string, opts adapter.Options,
) []sdk.TextBlockParam {
	resolved := a.ResolveSystemInstruction(fromContext, opts.SystemInstruction, true)
	if resolved == "" {
		return nil
	}
	// An instruction given for one call is not the conversation's prompt and has
	// no part that survives between turns, so there is nothing to cache.
	if !opts.EnablePromptCaching || resolved != fromContext {
		return []sdk.TextBlockParam{{Text: resolved}}
	}

	stable, volatile := convo.SystemParts()
	blocks := make([]sdk.TextBlockParam, 0, 2)
	if stable != "" {
		blocks = append(blocks, sdk.TextBlockParam{
			Text:         stable,
			CacheControl: sdk.NewCacheControlEphemeralParam(),
		})
	}
	if volatile != "" {
		blocks = append(blocks, sdk.TextBlockParam{Text: volatile})
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

// ToMessages converts the conversation's messages into Anthropic messages.
//
// A system or developer message enters as the user, which is the only input
// role Anthropic offers besides the assistant. Consecutive messages of one role
// are merged, because Anthropic requires the roles to alternate.
func ToMessages(msgs []frames.Message) ([]sdk.MessageParam, error) {
	out := make([]sdk.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		converted, send, err := toMessage(m)
		if err != nil {
			if _, ok := errors.AsType[*adapter.ConversionError](err); ok {
				return nil, err
			}
			return nil, &adapter.ConversionError{Cause: err}
		}
		if !send {
			continue
		}
		out = append(out, converted)
	}
	return mergeSameRole(out), nil
}

// toMessage converts one message, reporting whether there is one to send: a
// thought Anthropic cannot read is left out rather than sent as something else.
func toMessage(m frames.Message) (sdk.MessageParam, bool, error) {
	switch {
	case m.IsLLMSpecific():
		return nativeMessage(m)
	case len(m.ToolResults) > 0:
		blocks := make([]sdk.ContentBlockParamUnion, 0, len(m.ToolResults))
		for _, r := range m.ToolResults {
			blocks = append(blocks, sdk.NewToolResultBlock(r.ID, r.Content, false))
		}
		return sdk.NewUserMessage(blocks...), true, nil
	case len(m.ToolCalls) > 0:
		msg, err := toolCallMessage(m)
		return msg, err == nil, err
	case m.Role == frames.RoleAssistant:
		return sdk.NewAssistantMessage(sdk.NewTextBlock(text(m.Text))), true, nil
	default:
		// A user message, and a system or developer message too: Anthropic has
		// neither a system nor a developer input role, so all three enter as the
		// user rather than being dropped.
		return sdk.NewUserMessage(sdk.NewTextBlock(text(m.Text))), true, nil
	}
}

// Thought is a reasoning block the model produced, kept in the conversation so
// it can be handed back on the next turn.
//
// Signature is the encrypted reasoning Anthropic decrypts when the block is
// passed back. A thinking block is only valid with one, so a thought without a
// signature cannot be sent and is left out of the conversation entirely. The
// text may legitimately be empty: a model set to omit its reasoning returns
// thinking blocks with none.
type Thought struct {
	Text      string
	Signature string
}

// NewThought builds a conversation message carrying a reasoning block, written
// for Anthropic and sent to no other provider.
func NewThought(t Thought) frames.Message {
	return adapter.CreateLLMSpecificMessage(&Adapter{}, t)
}

// nativeMessage reads a message written for Anthropic. It is either a thought,
// which becomes a thinking block, or a message already in Anthropic's format.
func nativeMessage(m frames.Message) (sdk.MessageParam, bool, error) {
	if t, ok := m.Native.(Thought); ok {
		if t.Signature == "" {
			// Nothing to send: the block cannot be round-tripped without the
			// signature the API decrypts it by.
			return sdk.MessageParam{}, false, nil
		}
		return sdk.NewAssistantMessage(
			sdk.NewThinkingBlock(t.Signature, t.Text),
		), true, nil
	}
	native, err := adapter.NativeMessage[sdk.MessageParam](m)
	return native, err == nil, err
}

// toolCallMessage renders an assistant turn that requested tool calls.
//
// Any text the model spoke alongside the calls is dropped. It was already
// pushed downstream to be spoken and is committed as an assistant message of
// its own when the turn ends, so keeping it here too would say it twice.
func toolCallMessage(m frames.Message) (sdk.MessageParam, error) {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(m.ToolCalls))
	for _, c := range m.ToolCalls {
		input := any(c.Args)
		if len(c.Args) == 0 {
			input = json.RawMessage("{}")
		} else if !json.Valid(c.Args) {
			return sdk.MessageParam{}, &argumentsError{name: c.Name, id: c.ID}
		}
		blocks = append(blocks, sdk.NewToolUseBlock(c.ID, input, c.Name))
	}
	return sdk.NewAssistantMessage(blocks...), nil
}

// argumentsError reports a tool call whose arguments are not JSON, which
// Anthropic has no representation for: the input of a tool-use block is an
// object, not a string.
type argumentsError struct {
	name string
	id   string
}

// Error implements error.
func (e *argumentsError) Error() string {
	return "tool call " + e.id + " to " + e.name + " has arguments that are not valid JSON"
}

// text substitutes a placeholder for content that came out empty.
func text(s string) string {
	if s == "" {
		return emptyText
	}
	return s
}

// mergeSameRole folds each run of same-role messages into one carrying every
// content block in turn, because Anthropic requires the roles to alternate.
func mergeSameRole(msgs []sdk.MessageParam) []sdk.MessageParam {
	out := make([]sdk.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		if n := len(out); n > 0 && out[n-1].Role == m.Role {
			out[n-1].Content = append(out[n-1].Content, m.Content...)
			continue
		}
		out = append(out, m)
	}
	for i := range out {
		if len(out[i].Content) == 0 {
			out[i].Content = []sdk.ContentBlockParamUnion{sdk.NewTextBlock(emptyText)}
		}
	}
	return out
}

// EnsureLastMessageIsUser appends a minimal user message when the conversation
// ends on an assistant message, which a model without assistant-prefill support
// rejects.
func EnsureLastMessageIsUser(msgs []sdk.MessageParam) []sdk.MessageParam {
	if n := len(msgs); n > 0 && msgs[n-1].Role == sdk.MessageParamRoleAssistant {
		return append(msgs, sdk.NewUserMessage(sdk.NewTextBlock(noOpUserTurn)))
	}
	return msgs
}

// WithCacheControlMarkers marks the two most recent user messages so the prompt
// up to each is cached.
//
// The marker on the most recent user message tells Anthropic to cache the
// prompt up to that point. The one on the message before it tells Anthropic to
// look up the cache written on the previous turn, which ended there. Marking
// only the last would write a cache on every turn and never read one back.
//
// User messages are the ones marked because a turn is generated as soon as the
// user finishes speaking, and Anthropic's roles strictly alternate, so the last
// user message is where the reusable prefix ends.
func WithCacheControlMarkers(msgs []sdk.MessageParam) []sdk.MessageParam {
	out := make([]sdk.MessageParam, len(msgs))
	copy(out, msgs)
	marked := 0
	for i := len(out) - 1; i >= 0 && marked < 2; i-- {
		if out[i].Role != sdk.MessageParamRoleUser || len(out[i].Content) == 0 {
			continue
		}
		// The blocks are shared with the caller's messages, so the run is copied
		// before its last block is given a marker.
		blocks := make([]sdk.ContentBlockParamUnion, len(out[i].Content))
		copy(blocks, out[i].Content)
		last := &blocks[len(blocks)-1]
		if !markCacheControl(last) {
			continue
		}
		out[i].Content = blocks
		marked++
	}
	return out
}

// markCacheControl puts an ephemeral cache breakpoint on a content block,
// reporting whether the block is one that carries it.
//
// A block union holds its block by pointer, so the marked block is replaced by
// a copy rather than written through: the original is shared with the
// conversation this request was converted from.
func markCacheControl(b *sdk.ContentBlockParamUnion) bool {
	switch {
	case b.OfText != nil:
		blk := *b.OfText
		blk.CacheControl = sdk.NewCacheControlEphemeralParam()
		b.OfText = &blk
	case b.OfToolResult != nil:
		blk := *b.OfToolResult
		blk.CacheControl = sdk.NewCacheControlEphemeralParam()
		b.OfToolResult = &blk
	case b.OfImage != nil:
		blk := *b.OfImage
		blk.CacheControl = sdk.NewCacheControlEphemeralParam()
		b.OfImage = &blk
	default:
		return false
	}
	return true
}

// ToProviderToolsFormat implements adapter.LLMAdapter.
//
// Anthropic takes no custom tools: it has no tool this schema cannot describe,
// so anything under a custom key is not for it and is left out.
func (*Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []sdk.ToolUnionParam {
	out := make([]sdk.ToolUnionParam, 0, len(schema.Standard))
	for _, t := range schema.Standard {
		var schema struct {
			Properties json.RawMessage `json:"properties"`
			Required   []string        `json:"required"`
		}
		if len(t.Parameters) > 0 {
			_ = json.Unmarshal(t.Parameters, &schema)
		}
		tool := &sdk.ToolParam{
			Name:        t.Name,
			InputSchema: sdk.ToolInputSchemaParam{Required: schema.Required},
		}
		if t.Description != "" {
			tool.Description = param.NewOpt(t.Description)
		}
		if len(schema.Properties) > 0 {
			tool.InputSchema.Properties = schema.Properties
		}
		out = append(out, sdk.ToolUnionParam{OfTool: tool})
	}
	return out
}

// MessagesForLogging implements adapter.LLMAdapter.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(params.Messages))
	for _, m := range params.Messages {
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
