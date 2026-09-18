// Package deepseek provides DeepSeek's OpenAI-compatible LLM service.
package deepseek

const (
	baseURL = "https://api.deepseek.com/v1"
	// defaultModel is DeepSeek's current name for V4.1 Flash. The V4 Flash
	// identifiers it replaces name retired models and are only temporarily
	// routed here.
	defaultModel = "deepseek-flash"
)
