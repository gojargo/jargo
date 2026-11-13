package mistral

import (
	"log/slog"

	mistraladapter "github.com/gojargo/jargo/adapter/mistral"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// NewLLM builds a Mistral AI LLM service. Mistral speaks the OpenAI
// chat-completions API, with three departures from it: it has no developer
// role, it constrains the shape of a conversation (which its adapter settles),
// and it reports the calls to make from the whole message history rather than
// from what it just streamed (see dropAnsweredCalls). It also names the
// sampling seed differently, and takes fewer of OpenAI's request fields (see
// mistralRequest).
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "MistralLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
		// The endpoint has no max_completion_tokens field.
		NoMaxCompletionTokens: true,
		Adapter:               &mistraladapter.Adapter{},
		Base:                  []llm.Option{llm.WithFunctionCallFilter(dropAnsweredCalls)},
		Request:               mistralRequest,
	}, cfg)
}

// mistralRequest shapes a request the way Mistral takes it. The seed is read
// under a name of its own, so it is sent as random_seed, and as that alone: the
// request is built from the settings as they stand, so a seed an update sets
// is renamed too. Mistral has no use for the usage request, the service tier or
// the completion bound OpenAI's reasoning models read, so they are left off.
// The caller's Extra fields are merged over all of it.
func mistralRequest(body map[string]any) {
	if seed, ok := body["seed"]; ok {
		body["random_seed"] = seed
		delete(body, "seed")
	}
	delete(body, "stream_options")
	delete(body, "service_tier")
	delete(body, "max_completion_tokens")
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
