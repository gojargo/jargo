// Package openai provides an LLM and a transcription service for Azure
// OpenAI. Azure exposes the same APIs as OpenAI but addresses them per model
// deployment and authorizes with an api-key header rather than a bearer token,
// so both services reuse the OpenAI-compatible clients with an Azure request
// shaper.
package openai

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
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
	chat.LLMConfig
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
