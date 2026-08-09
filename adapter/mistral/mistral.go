// Package mistral converts a universal conversation into the request Mistral
// takes.
//
// Mistral speaks the OpenAI chat-completions schema but constrains the shape of
// a conversation beyond what that schema says, so this adapter embeds the
// OpenAI one and rewrites what it produced rather than converting the
// conversation a second time.
package mistral

import (
	"maps"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// Adapter converts a universal conversation into a Mistral chat-completions
// request. The zero value is ready to use.
//
// Messages travel under the OpenAI identifier, which the embedded adapter
// supplies: Mistral reads the same message format, so a message written for one
// is readable by the other.
type Adapter struct {
	openai.Adapter
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[openai.Params, openai.Tool] = (*Adapter)(nil)

// LLMInvocationParams converts the conversation and then rewrites it to satisfy
// the constraints Mistral puts on a message history.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (openai.Params, error) {
	p, err := a.Adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return openai.Params{}, err
	}
	p.Messages = TransformMessages(p.Messages)
	return p, nil
}

// TransformMessages rewrites the conversation to satisfy the three constraints
// Mistral puts on a message history that the OpenAI schema does not:
//
//  1. A tool result must be followed by an assistant message. One that is not
//     is rejected, so a minimal assistant message is inserted after it.
//  2. Only the leading run of system messages is accepted. A system message
//     after any other message is sent as a user message instead.
//  3. A conversation ending on an assistant message is a partial reply Mistral
//     is being asked to continue, which it only does when the message is marked
//     as a prefix.
func TransformMessages(msgs []openai.Message) []openai.Message {
	if len(msgs) == 0 {
		return msgs
	}

	out := make([]openai.Message, 0, len(msgs)+1)
	for i, m := range msgs {
		out = append(out, m)
		if m.Role != openai.RoleTool {
			continue
		}
		if i == len(msgs)-1 || msgs[i+1].Role != openai.RoleAssistant {
			out = append(out, openai.Message{Role: openai.RoleAssistant, Content: " "})
		}
	}

	demoteLateSystem(out)

	if last := &out[len(out)-1]; last.Role == openai.RoleAssistant {
		if _, ok := last.Extra["prefix"]; !ok {
			extra := maps.Clone(last.Extra)
			if extra == nil {
				extra = map[string]any{}
			}
			extra["prefix"] = true
			last.Extra = extra
		}
	}
	return out
}

// demoteLateSystem sends every system message past the leading run as a user
// message, which is the only place Mistral accepts one.
func demoteLateSystem(msgs []openai.Message) {
	for i := range msgs {
		if msgs[i].Role == openai.RoleSystem {
			continue
		}
		for j := i; j < len(msgs); j++ {
			if msgs[j].Role == openai.RoleSystem {
				msgs[j].Role = openai.RoleUser
			}
		}
		return
	}
}
