// Package perplexity converts a universal conversation into the request
// Perplexity takes.
//
// Perplexity speaks the OpenAI chat-completions schema but is stricter than
// OpenAI about the shape of a conversation, so this adapter embeds the OpenAI
// one and rewrites what it produced rather than converting the conversation a
// second time.
package perplexity

import (
	"slices"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// Adapter converts a universal conversation into a Perplexity chat-completions
// request. The zero value is ready to use.
//
// Messages travel under the OpenAI identifier, which the embedded adapter
// supplies: Perplexity reads the same message format, so a message written for
// one is readable by the other.
type Adapter struct {
	openai.Adapter
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[openai.Params, openai.Tool] = (*Adapter)(nil)

// LLMInvocationParams converts the conversation and then rewrites it to satisfy
// the constraints Perplexity puts on a message history.
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
// Perplexity puts on a message history that the OpenAI schema does not:
//
//  1. Only the leading run of system messages is accepted. A system message
//     after any other message is sent as a user message instead.
//  2. Roles must strictly alternate. Adjacent messages of the same role are
//     merged into one carrying both their contents, which is also what the
//     demotion above tends to produce: a system message sent as a user message
//     next to a user message.
//  3. The conversation must end on a user or tool message. A trailing assistant
//     message is dropped, which is what OpenAI does with one server-side, so the
//     turn reads the same either way.
//
// A trailing system message is deliberately left as it is. Demoting it would
// make the rewrite depend on how much of the conversation has happened so far,
// and Perplexity keeps state within a conversation: a message sent as a user
// message on one turn and as a system message on the next, once more messages
// follow it, is rejected. A conversation of nothing but system messages is
// refused by the API, which reports the mistake straight away.
func TransformMessages(msgs []openai.Message) []openai.Message {
	if len(msgs) == 0 {
		return msgs
	}
	// The demotion writes to the messages, which the caller still holds.
	out := slices.Clone(msgs)
	openai.DemoteLateSystem(out)
	out = mergeSameRole(out)
	return dropTrailingAssistant(out)
}

// mergeSameRole folds each run of same-role messages into one carrying every
// content in turn, so the roles strictly alternate.
//
// Two system messages are never merged. Perplexity accepts several at the
// start, and by this point the demotion above has left system messages nowhere
// else, so a run of them is the opening block and merging it would rewrite a
// conversation the endpoint already reads correctly.
func mergeSameRole(msgs []openai.Message) []openai.Message {
	out := make([]openai.Message, 0, len(msgs))
	for _, m := range msgs {
		if len(out) == 0 {
			out = append(out, m)
			continue
		}
		last := &out[len(out)-1]
		if last.Role != m.Role || m.Role == openai.RoleSystem {
			out = append(out, m)
			continue
		}
		last.ContentParts = append(last.Parts(), m.Parts()...)
		last.Content = ""
	}
	return out
}

// dropTrailingAssistant removes the assistant messages the conversation ends on,
// which Perplexity refuses as a final message.
func dropTrailingAssistant(msgs []openai.Message) []openai.Message {
	for len(msgs) > 0 && msgs[len(msgs)-1].Role == openai.RoleAssistant {
		msgs = msgs[:len(msgs)-1]
	}
	return msgs
}
