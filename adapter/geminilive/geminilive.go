// Package geminilive converts a universal conversation into what Gemini Live
// takes on a session.
//
// It is the Gemini adapter with two differences. Tool calls and their results
// are converted to text, which is all the Live API accepts when a session is
// seeded with history. And the tools are those a realtime session carries: the
// conversation's own when it advertises any, else the ones the service was
// configured with.
package geminilive

import (
	"fmt"
	"strings"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/frames"
)

// unnamedToolResult names a result whose call is not in the conversation. It is
// the same fallback the Gemini adapter uses.
const unnamedToolResult = "tool_call_result"

// Adapter converts a universal conversation into Gemini Live session
// parameters. The zero value is ready to use.
type Adapter struct {
	gemini.Adapter
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[gemini.Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "gemini-live" }

// LLMInvocationParams converts the conversation into what a Live session takes
// from it: the system instruction, the contents to seed it with, and the tools
// it carries.
//
// The tools are the conversation's own when it has any, else the ones given in
// opts.ServiceTools; the built-in tools ride along with whichever set is in use,
// and are the set when there is neither.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (gemini.Params, error) {
	params, err := a.ParamsFromMessages(
		convo.System(), ConvertToolCallsToText(convo.MessagesFor(a.IDForLLMSpecificMessages())), opts,
	)
	if err != nil {
		return gemini.Params{}, err
	}
	if tools := a.RealtimeSessionTools(convo.ToolsSchema(), opts.ServiceTools); !tools.Empty() {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ConvertToolCallsToText replaces each tool call and each tool result with a
// text message describing it. Seeding a Gemini Live conversation with tool
// calls, on a reconnect or a handoff from another worker, fails with a 1007
// error.
//
// A message calling several functions becomes one assistant message summarizing
// them all; each result becomes a user message naming the function it came
// from, or a fallback name when its call is not in the conversation.
func ConvertToolCallsToText(messages []frames.Message) []frames.Message {
	names := map[string]string{}
	out := make([]frames.Message, 0, len(messages))
	for _, m := range messages {
		switch {
		case m.IsLLMSpecific():
			out = append(out, m)
		case len(m.ToolCalls) > 0:
			summaries := make([]string, 0, len(m.ToolCalls))
			for _, c := range m.ToolCalls {
				names[c.ID] = c.Name
				summaries = append(summaries, fmt.Sprintf("[Called function %s with args %s]", c.Name, c.Args))
			}
			out = append(out, frames.Message{Role: frames.RoleAssistant, Text: strings.Join(summaries, " ")})
		case len(m.ToolResults) > 0:
			for _, r := range m.ToolResults {
				name, ok := names[r.ID]
				if !ok {
					name = unnamedToolResult
				}
				out = append(out, frames.Message{
					Role: frames.RoleUser, Text: fmt.Sprintf("[Function %s returned %s]", name, r.Content),
				})
			}
		default:
			out = append(out, m)
		}
	}
	return out
}
