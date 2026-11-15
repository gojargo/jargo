// Package crusoe provides Crusoe Cloud's OpenAI-compatible LLM service, on its
// Managed Inference API.
package crusoe

const (
	// baseURL is written without a trailing slash, since the chat-completions
	// path is appended to it as is.
	baseURL      = "https://api.inference.crusoecloud.com/v1"
	defaultModel = "openai/gpt-oss-120b"
)
