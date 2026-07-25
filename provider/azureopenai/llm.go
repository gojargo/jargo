package azureopenai

import (
	"net/http"
	"strings"

	"github.com/gojargo/jargo/provider/openai"
)

// shaper addresses and authorizes requests the Azure OpenAI way.
type shaper struct{ apiVersion string }

// Endpoint appends the chat-completions path and the required api-version query.
func (s shaper) Endpoint(baseURL string) string {
	return baseURL + "/chat/completions?api-version=" + s.apiVersion
}

// Authorize sets Azure's api-key header.
func (shaper) Authorize(req *http.Request, apiKey string) {
	req.Header.Set("api-key", apiKey)
}

// NewLLM builds an Azure OpenAI LLM service.
func NewLLM(cfg Config) *openai.LLMService {
	apiVersion := cfg.APIVersion
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	base := strings.TrimSuffix(cfg.Endpoint, "/") + "/openai/deployments/" + cfg.Deployment

	llmCfg := cfg.LLMConfig
	llmCfg.BaseURL = "" // the URL is built from Endpoint/Deployment, not BaseURL
	if llmCfg.Model == "" {
		llmCfg.Model = cfg.Deployment
	}
	return openai.NewShapedLLM("AzureOpenAILLM", base, cfg.Deployment, shaper{apiVersion: apiVersion}, llmCfg)
}
