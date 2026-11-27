package openai

import (
	"net/http"
	"strings"

	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
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

// LLMService is the Azure OpenAI LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds an Azure OpenAI LLM service.
func NewLLM(cfg Config) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg Config, opts ...llm.Option) *chat.LLMService {
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
	return chat.NewCompatLLM(chat.Compat{
		Name:         "AzureOpenAILLM",
		BaseURL:      base,
		DefaultModel: cfg.Deployment,
		Shaper:       shaper{apiVersion: apiVersion},
	}, llmCfg, opts...)
}
