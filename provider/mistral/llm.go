package mistral

import (
	"log/slog"
	"maps"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// NewLLM builds a Mistral AI LLM service. Mistral speaks the OpenAI
// chat-completions API, with three departures from it: it has no developer
// role, it constrains the shape of a conversation (see shapeMessages), and it
// reports the calls to make from the whole message history rather than from
// what it just streamed (see dropAnsweredCalls). It also names the sampling
// seed differently.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	if cfg.Seed != nil {
		// Mistral reads the seed under a name of its own. Send it under that
		// name only: the modeled field would go out as "seed" as well.
		extra := maps.Clone(cfg.Extra)
		if extra == nil {
			extra = map[string]any{}
		}
		extra["random_seed"] = *cfg.Seed
		cfg.Extra = extra
		cfg.Seed = nil
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "MistralLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
		ShapeMessages:   shapeMessages,
		Base:            []llm.Option{llm.WithFunctionCallFilter(dropAnsweredCalls)},
	}, cfg)
}

// dropAnsweredCalls keeps only the calls the conversation has no result for.
//
// Most providers report a call once, from the chunks streamed as the model
// makes it, and the completion that answers the call's result does not report
// it again. Mistral reports the calls it finds across the whole message
// history, so that second completion asks for the same call a second time.
// Running it again would repeat whatever the tool did: send the message twice,
// start the playback twice, dispatch twice.
//
// A call whose result is already in the conversation has therefore been made
// already, and is dropped. That covers a call still running as well as one that
// has reported, because the result message is written when the call starts.
func dropAnsweredCalls(convo *frames.LLMContext, calls []frames.ToolCall) []frames.ToolCall {
	answered := make(map[string]struct{})
	for _, m := range convo.Messages() {
		for _, r := range m.ToolResults {
			answered[r.ID] = struct{}{}
		}
	}
	kept := make([]frames.ToolCall, 0, len(calls))
	for _, c := range calls {
		// An id is what pairs a call to its result, so a call without one cannot
		// be matched to anything and is left to run.
		if _, done := answered[c.ID]; done && c.ID != "" {
			slog.Debug("dropping a function call the conversation has already answered",
				"function", c.Name, "tool_call_id", c.ID)
			continue
		}
		kept = append(kept, c)
	}
	return kept
}

// shapeMessages rewrites the conversation to satisfy the three constraints
// Mistral puts on a message history that the OpenAI schema does not:
//
//  1. A tool result must be followed by an assistant message. One that is not
//     is rejected, so a minimal assistant message is inserted after it.
//  2. Only the leading run of system messages is accepted. A system message
//     after any other message is sent as a user message instead.
//  3. A conversation ending on an assistant message is a partial reply Mistral
//     is being asked to continue, which it only does when the message is marked
//     as a prefix.
func shapeMessages(msgs []chat.Message) []chat.Message {
	if len(msgs) == 0 {
		return msgs
	}

	out := make([]chat.Message, 0, len(msgs)+1)
	for i, m := range msgs {
		out = append(out, m)
		if m.Role != chat.RoleTool {
			continue
		}
		if i == len(msgs)-1 || msgs[i+1].Role != chat.RoleAssistant {
			out = append(out, chat.Message{Role: chat.RoleAssistant, Content: " "})
		}
	}

	for i := range out {
		if out[i].Role == chat.RoleSystem {
			continue
		}
		// Past the leading run, every remaining system message is demoted.
		for j := i; j < len(out); j++ {
			if out[j].Role == chat.RoleSystem {
				out[j].Role = chat.RoleUser
			}
		}
		break
	}

	if last := &out[len(out)-1]; last.Role == chat.RoleAssistant {
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
