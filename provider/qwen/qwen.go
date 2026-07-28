// Package qwen provides Alibaba's Qwen LLM over the DashScope OpenAI-compatible
// endpoint.
package qwen

const (
	// baseURL is the international DashScope OpenAI-compatible endpoint; callers
	// inside mainland China should override it with the dashscope.aliyuncs.com
	// host via chat.LLMConfig.BaseURL.
	baseURL      = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	defaultModel = "qwen-plus"
)
