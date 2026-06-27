// Package azureopenai provides an LLM service for Azure OpenAI. Azure exposes
// the same chat-completions API as OpenAI but addresses it per model deployment
// (<endpoint>/openai/deployments/<deployment>/chat/completions?api-version=...)
// and authorizes with an api-key header rather than a bearer token, so this
// provider reuses the OpenAI-compatible LLM with an Azure request shaper.
package azureopenai

import (
	"net/http"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai"
)

// defaultAPIVersion is a recent stable Azure OpenAI REST API version.
const defaultAPIVersion = "2024-10-21"

// Config configures the Azure OpenAI LLM provider.
type Config struct {
	// Endpoint is the Azure OpenAI resource endpoint, e.g.
	// https://my-resource.openai.azure.com. Required.
	Endpoint string `validate:"required,url"`
	// Deployment is the model deployment name. Required.
	Deployment string `validate:"required"`
	// APIVersion is the Azure OpenAI REST API version; empty uses a recent stable.
	APIVersion string
	// LLMConfig carries the shared OpenAI LLM options (APIKey, sampling, MaxTokens
	// and so on). Its BaseURL is ignored: the URL is built from Endpoint and
	// Deployment.
	openai.LLMConfig
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

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
