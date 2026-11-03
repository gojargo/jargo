// Package realtime is a speech-to-speech service built on Azure OpenAI's
// Realtime API. It wraps the OpenAI Realtime service and changes only how the
// session is addressed and authorized: Azure serves the model as a deployment
// of a resource, and authorizes with an api-key header or a Microsoft Entra ID
// token. Everything downstream of that behaves identically.
//
// It reaches Azure over its v1 API surface, whose realtime protocol is the one
// the OpenAI service speaks. An endpoint carrying a dated api-version serves the
// superseded preview protocol, which rejects the session configuration and
// names its events differently, so it is not usable here.
package realtime

import (
	"context"
	"log/slog"
	"net/http"
	"net/url"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	openairealtime "github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// TokenProvider supplies a Microsoft Entra ID bearer token, for the
// https://ai.azure.com/.default scope. It is called each time the session is
// opened, so it can refresh the token.
type TokenProvider func(ctx context.Context) (string, error)

// Config configures the Azure OpenAI Realtime service.
type Config struct {
	// BaseURL is the Realtime WebSocket endpoint on Azure's v1 API surface,
	// e.g. wss://my-resource.openai.azure.com/openai/v1/realtime. The
	// deployment is appended as ?model=<Model>. A URL that already carries a
	// query string is used as it stands, leaving the deployment for it to name.
	// Required.
	BaseURL string `validate:"required"`
	// APIKey is the Azure OpenAI resource key, sent as the api-key header.
	// Required unless TokenProvider is set.
	APIKey string `validate:"required_without=TokenProvider"`
	// TokenProvider supplies a Microsoft Entra ID bearer token, used instead of
	// APIKey when set.
	TokenProvider TokenProvider
	// Model names the deployment.
	Model string
	// Instructions is the system instruction for the session.
	Instructions string
	// SessionProperties configure the session; see the OpenAI Realtime
	// service's Config.
	SessionProperties events.SessionProperties
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// New builds an Azure OpenAI Realtime service.
func New(cfg Config) *openairealtime.Service {
	if strings.Contains(cfg.BaseURL, "api-version=") {
		// The preview surface rejects the session configuration and names its
		// events differently, so it fails partway through a connection that opens
		// cleanly. It is said up front, where the URL that caused it is in view.
		slog.Warn("BaseURL selects Azure's preview realtime API, whose protocol this service doesn't " +
			"implement; use an /openai/v1/realtime URL instead")
	}
	conn := &connector{baseURL: cfg.BaseURL, apiKey: cfg.APIKey, tokenProvider: cfg.TokenProvider}
	svc := openairealtime.NewWithConnector("AzureRealtime", conn, openairealtime.Config{
		Model:             cfg.Model,
		Instructions:      cfg.Instructions,
		SessionProperties: cfg.SessionProperties,
	})
	conn.model = svc.Model()
	return svc
}

// connector dials the Azure endpoint with an api-key header or an Entra ID
// bearer token.
type connector struct {
	baseURL       string
	model         string
	apiKey        string
	tokenProvider TokenProvider
}

// Endpoint returns the session URL, naming the deployment unless the URL
// already does, and the header Azure authorizes with.
func (c *connector) Endpoint(ctx context.Context) (string, http.Header, error) {
	endpoint := c.baseURL
	if !strings.Contains(endpoint, "?") {
		endpoint += "?model=" + url.QueryEscape(c.model)
	}
	header := http.Header{}
	if c.tokenProvider != nil {
		token, err := c.tokenProvider(ctx)
		if err != nil {
			return "", nil, err
		}
		header.Set("Authorization", "Bearer "+token)
		return endpoint, header, nil
	}
	header.Set("api-key", c.apiKey)
	return endpoint, header, nil
}
