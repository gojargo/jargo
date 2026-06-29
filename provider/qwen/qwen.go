// Package qwen provides Alibaba's Qwen LLM over the DashScope OpenAI-compatible
// endpoint.
package qwen

import "github.com/gojargo/jargo/provider/openai"

const (
	// baseURL is the international DashScope OpenAI-compatible endpoint; callers
	// inside mainland China should override it with the dashscope.aliyuncs.com
	// host via openai.LLMConfig.BaseURL.
	baseURL      = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	defaultModel = "qwen-plus"
)

// NewLLM builds a Qwen (DashScope) LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("QwenLLM", baseURL, defaultModel, cfg)
}
