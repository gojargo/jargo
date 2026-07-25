// Package perplexity provides Perplexity's OpenAI-compatible LLM service. Its
// base URL has no "/v1" path segment, unlike most compatible providers.
package perplexity

const (
	baseURL      = "https://api.perplexity.ai"
	defaultModel = "sonar"
)
