// Package deepseek converts a universal conversation into the request DeepSeek
// takes.
//
// DeepSeek's API is OpenAI-compatible, but in thinking mode it requires every
// assistant message of the current turn to carry a reasoning_content field once
// a tool call is involved, and rejects the request with a 400 otherwise. The
// conversation does not keep a model's reasoning, so this adapter supplies an
// empty reasoning_content on assistant messages that lack one. DeepSeek accepts
// the empty field on any assistant message and ignores it when thinking is
// disabled, so it is applied uniformly rather than per turn.
package deepseek

import (
	"maps"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// reasoningContent is the assistant message field DeepSeek reads a model's
// reasoning back from.
const reasoningContent = "reasoning_content"

// Adapter converts a universal conversation into a DeepSeek chat-completions
// request. The zero value is ready to use.
//
// It embeds the OpenAI adapter and adds an empty reasoning_content to every
// assistant message that has none, so requests are accepted in thinking mode
// after a tool call. Messages travel under the OpenAI identifier, which the
// embedded adapter supplies: DeepSeek reads the same message format.
type Adapter struct {
	openai.Adapter
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[openai.Params, openai.Tool] = (*Adapter)(nil)

// LLMInvocationParams converts the conversation and then adds the
// reasoning_content DeepSeek requires on every assistant message.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (openai.Params, error) {
	p, err := a.Adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return openai.Params{}, err
	}
	p.Messages = AddReasoningContent(p.Messages)
	return p, nil
}

// AddReasoningContent returns the messages with reasoning_content on every
// assistant message.
//
// Assistant messages that already carry the field are left as they are. A
// message written in the provider's own format shares its Extra map with the
// conversation, so a stamped message gets a copy of it rather than a write.
func AddReasoningContent(msgs []openai.Message) []openai.Message {
	out := make([]openai.Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == openai.RoleAssistant {
			if _, ok := m.Extra[reasoningContent]; !ok {
				extra := maps.Clone(m.Extra)
				if extra == nil {
					extra = map[string]any{}
				}
				extra[reasoningContent] = ""
				m.Extra = extra
			}
		}
		out = append(out, m)
	}
	return out
}
