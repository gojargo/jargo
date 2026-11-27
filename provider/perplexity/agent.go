package perplexity

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/responses"
)

const (
	// agentBaseURL is the Agent API base, which speaks the Responses protocol.
	agentBaseURL = "https://api.perplexity.ai/v1"
	// defaultAgentModel is the model the Agent API service uses by default.
	defaultAgentModel = "openai/gpt-5.6-luna"
	// fastPreset is Perplexity's lowest-latency preset, which keeps reasoning
	// low for models from every provider it serves.
	fastPreset = "fast"
)

// errModelOrPresetRequired reports a configuration that leaves the Agent API
// neither a model nor a preset to choose one.
var errModelOrPresetRequired = errors.New("perplexity: the Agent API needs a Model, a Preset, or both")

// AgentConfig configures the Agent API service. The sampling controls are
// pointers so a deliberate zero is distinguishable from "unset"; a nil value is
// omitted from the request, leaving the API default.
type AgentConfig struct {
	// APIKey is the Perplexity API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted Agent API.
	BaseURL string
	// Model is the model id, such as "openai/gpt-5.5" or
	// "anthropic/claude-haiku-4-5"; empty uses openai/gpt-5.6-luna. With a
	// Preset, it replaces the preset's own model.
	Model string
	// PresetModel leaves the model to the Preset: the request names none, and
	// Model is ignored. It needs a Preset.
	PresetModel bool
	// Preset is a Perplexity preset that sets the model, its instructions, its
	// search budget and how much it reasons, such as "fast", "low", "medium",
	// "high" or "xhigh". Empty sends none, and a request with no Reasoning
	// either then gets the "fast" preset.
	Preset string
	// NoWebSearch stops the service offering the web_search tool on every
	// request, alongside any tools from the context. A preset searches the web
	// whatever this is set to.
	NoWebSearch bool
	// MaxSteps is the maximum number of search and reasoning steps per
	// response; 0 leaves Perplexity's default.
	MaxSteps int `validate:"gte=0"`
	// Instructions is the system prompt, sent alongside the input rather than as
	// a message. A system message on the context takes precedence.
	Instructions string
	// MaxOutputTokens caps the response length; 0 omits it. Anthropic models
	// need it set.
	MaxOutputTokens int
	// Temperature is the sampling temperature; nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter; nil omits it.
	TopP *float64
	// Reasoning configures how much the model reasons. Set, it replaces the
	// "fast" preset otherwise sent when there is no Preset.
	Reasoning *responses.ReasoningConfig
	// Extra sets arbitrary additional request fields not modeled above, applied
	// to every request. They override the modeled fields on conflict.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c AgentConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.PresetModel && c.Preset == "" {
		return errModelOrPresetRequired
	}
	return nil
}

// NewAgentLLM builds a service for Perplexity's Agent API.
//
// The Agent API answers questions with web search, using models from
// Perplexity and other providers. It speaks the OpenAI Responses protocol over
// HTTP, so function calling and multi-turn context work as they do with the
// Responses HTTP service.
//
// By default the service uses openai/gpt-5.6-luna with web search on. When
// neither Preset nor Reasoning is set, it sends Perplexity's "fast" preset,
// its lowest-latency configuration, with the model in place of the preset's
// own, and logs that it did for each model. Answers can carry citation markers
// such as "[1]".
func NewAgentLLM(cfg AgentConfig) *responses.HTTPLLMService {
	model := cfg.Model
	switch {
	case cfg.PresetModel:
		model = ""
	case model == "":
		model = defaultAgentModel
	}
	base := cfg.BaseURL
	if base == "" {
		base = agentBaseURL
	}
	return responses.NewHTTPVariantLLM(responses.Config{
		APIKey:          cfg.APIKey,
		BaseURL:         base,
		Model:           model,
		Instructions:    cfg.Instructions,
		MaxOutputTokens: cfg.MaxOutputTokens,
		Temperature:     cfg.Temperature,
		TopP:            cfg.TopP,
		Reasoning:       cfg.Reasoning,
		Extra:           cfg.Extra,
	}, responses.HTTPVariant{
		Name:            "PerplexityAgentHTTPLLM",
		ThinkingDefault: cfg.applyThinkingDefault,
		Request:         cfg.editRequest,
		Events:          withoutServerToolCalls,
		Usage:           cacheCreationTokens,
	})
}

// applyThinkingDefault sends the "fast" preset when no preset or reasoning is
// configured.
//
// A request that already sets reasoning or a preset, from the config or from
// Extra, is left as it is, and so is every one when the reporter's
// UseProviderThinkingDefault is set. The first request for each model logs the
// preset sent, or that the model runs at Perplexity's default.
func (c AgentConfig) applyThinkingDefault(body map[string]any, model string, report responses.ThinkingReporter) {
	if _, ok := body["reasoning"]; ok {
		return
	}
	if _, ok := body["preset"]; ok || c.Preset != "" {
		return
	}
	if report.UseProviderThinkingDefault() {
		report.ReportThinkingDefault(model, "", true, "Perplexity", "Preset or Reasoning in perplexity.AgentConfig")
		return
	}
	body["preset"] = fastPreset
	report.ReportThinkingDefault(model, fmt.Sprintf("preset=%q", fastPreset), true,
		"Perplexity", "Preset or Reasoning in perplexity.AgentConfig")
}

// editRequest adds Perplexity's own fields to a request body: the web_search
// tool, the preset and the step limit. A request without a model leaves it to
// the preset.
func (c AgentConfig) editRequest(body map[string]any) {
	if model, _ := body["model"].(string); model == "" {
		// The preset chooses the model.
		delete(body, "model")
	}
	if !c.NoWebSearch {
		tools, _ := body["tools"].([]any)
		if !hasWebSearch(tools) {
			body["tools"] = append(tools, map[string]any{"type": "web_search"})
		}
	}
	if c.Preset != "" {
		body["preset"] = c.Preset
	}
	if c.MaxSteps > 0 {
		body["max_steps"] = c.MaxSteps
	}
}

// hasWebSearch reports whether the tools already include web_search.
func hasWebSearch(tools []any) bool {
	for _, tool := range tools {
		if t, ok := tool.(map[string]any); ok && t["type"] == "web_search" {
			return true
		}
	}
	return false
}

// streamEvent is the part of a Responses stream event the filter reads.
type streamEvent struct {
	Type        string `json:"type"`
	OutputIndex int    `json:"output_index"`
	Item        *struct {
		Type string `json:"type"`
	} `json:"item"`
}

// withoutServerToolCalls returns a filter for one response's events that drops
// the function calls the server runs itself.
//
// With a preset, Perplexity runs its web search on the server: it announces the
// search as a function call and completes that output as another kind of item,
// such as search_results. Each function call's announcement is held until its
// output completes, and dropped unless it completes as a function call, so the
// search never reaches the service as a call to run. The service takes a
// call's arguments from its completed item, so argument events pass through.
func withoutServerToolCalls() func(data []byte) [][]byte {
	announced := map[int][]byte{} // output index -> the event announcing it
	return func(data []byte) [][]byte {
		var ev streamEvent
		if json.Unmarshal(data, &ev) != nil {
			return [][]byte{data}
		}
		switch ev.Type {
		case "response.output_item.added":
			if ev.Item != nil && ev.Item.Type == "function_call" {
				announced[ev.OutputIndex] = data
				return nil
			}
		case "response.output_item.done":
			added, ok := announced[ev.OutputIndex]
			delete(announced, ev.OutputIndex)
			if ok && ev.Item != nil && ev.Item.Type == "function_call" {
				return [][]byte{added, data}
			}
		}
		return [][]byte{data}
	}
}

// cacheCreationTokens reads the tokens Perplexity reports as written to the
// prompt cache, which it names cache_creation_input_tokens.
func cacheCreationTokens(raw json.RawMessage, tokens *frames.LLMTokenUsage) {
	var u struct {
		InputTokensDetails struct {
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		} `json:"input_tokens_details"`
	}
	if json.Unmarshal(raw, &u) != nil {
		return
	}
	if n := u.InputTokensDetails.CacheCreationInputTokens; n > 0 {
		tokens.CacheCreationTokens = &n
	}
}
