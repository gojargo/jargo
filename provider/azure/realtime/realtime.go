// Package realtime is a speech-to-speech service built on Azure OpenAI's
// Realtime API. It wraps the OpenAI Realtime service and changes only how the
// session is addressed and authorized: Azure serves the model per resource and
// deployment rather than by name, and authorizes with an api-key header rather
// than a bearer token. Everything downstream of that behaves identically.
package realtime

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	openairealtime "github.com/gojargo/jargo/provider/openai/realtime"
)

// defaultAPIVersion is a recent Azure OpenAI Realtime API version.
const defaultAPIVersion = "2025-04-01-preview"

// realtimePath is the Realtime endpoint under an Azure OpenAI resource.
const realtimePath = "/openai/realtime"

// Config configures the Azure OpenAI Realtime service.
type Config struct {
	// APIKey is the Azure OpenAI resource key, sent as the api-key header.
	// Required.
	APIKey string `validate:"required"`
	// Endpoint is the Azure OpenAI resource endpoint, e.g.
	// https://my-resource.openai.azure.com. It is dialed over WebSocket, so an
	// https scheme is rewritten to wss. Required unless URL is set.
	Endpoint string `validate:"required_without=URL"`
	// Deployment is the realtime model deployment name. Required unless URL is
	// set.
	Deployment string `validate:"required_without=URL"`
	// APIVersion is the Azure OpenAI REST API version; empty uses a recent one.
	APIVersion string
	// URL overrides the whole endpoint, for a deployment whose address is not
	// derivable from the fields above. It must already carry the api-version and
	// deployment query parameters.
	URL string
	// Voice is the model voice; empty uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// TranscriptionModel transcribes the user's audio; empty uses a default. Set
	// it to "-" to disable input transcription.
	TranscriptionModel string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// New builds an Azure OpenAI Realtime service.
func New(cfg Config) *openairealtime.Service {
	conn := &connector{
		apiKey: cfg.APIKey,
		url:    endpoint(cfg),
	}
	return openairealtime.NewWithConnector("AzureRealtime", conn, openairealtime.Config{
		Voice:              cfg.Voice,
		Instructions:       cfg.Instructions,
		TranscriptionModel: cfg.TranscriptionModel,
	})
}

// endpoint builds the session URL. Azure selects the model with a deployment
// query parameter rather than a model name, so nothing about the model travels
// in the session configuration.
func endpoint(cfg Config) string {
	if cfg.URL != "" {
		return cfg.URL
	}
	apiVersion := cfg.APIVersion
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	q := url.Values{}
	q.Set("api-version", apiVersion)
	q.Set("deployment", cfg.Deployment)
	return wsScheme(strings.TrimSuffix(cfg.Endpoint, "/")) + realtimePath + "?" + q.Encode()
}

// wsScheme rewrites an https endpoint to the WebSocket scheme, so the resource
// endpoint can be copied from the Azure portal as-is.
func wsScheme(endpoint string) string {
	switch {
	case strings.HasPrefix(endpoint, "https://"):
		return "wss://" + strings.TrimPrefix(endpoint, "https://")
	case strings.HasPrefix(endpoint, "http://"):
		return "ws://" + strings.TrimPrefix(endpoint, "http://")
	default:
		return endpoint
	}
}

// connector dials the Azure endpoint with an api-key header.
type connector struct {
	apiKey string
	url    string
}

// Endpoint returns the session URL and the header Azure authorizes with.
func (c *connector) Endpoint(context.Context) (string, http.Header, error) {
	header := http.Header{}
	header.Set("api-key", c.apiKey)
	return c.url, header, nil
}
