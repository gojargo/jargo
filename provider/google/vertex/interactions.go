package vertex

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gojargo/jargo/provider/google/interactions"
)

// InteractionsLLMConfig configures the Gemini Interactions LLM service on Vertex
// AI. The generation controls mirror the Interactions service's; only the
// addressing and authorization differ.
type InteractionsLLMConfig struct {
	// ProjectID is the Google Cloud project serving the model. Required.
	ProjectID string `validate:"required"`
	// Location is the location of the Vertex AI endpoint: "global" (the
	// default), "us" or "eu". Vertex serves the Interactions API from these
	// locations only, for Gemini 3 models and later.
	Location string
	// Credentials authorize the requests. Required.
	Credentials Credentials

	// The settings below are the Interactions service's; see
	// interactions.Config for each. An empty Model uses gemini-3.8-flash.

	Model             string
	MaxTokens         int
	Temperature       *float64
	TopP              *float64
	TopK              *int
	SafetySettings    []interactions.SafetySetting `validate:"omitempty,dive"`
	Thinking          *interactions.ThinkingConfig `validate:"omitempty"`
	Seed              *int
	Tools             []map[string]any
	ToolConfig        map[string]any
	StreamIdleTimeout *time.Duration
	RetryOnTimeout    bool
	RetryTimeout      time.Duration `validate:"omitempty,min=0"`
	Extra             map[string]any
}

// Validate reports whether the configuration is usable.
func (c InteractionsLLMConfig) Validate() error { return validateConfig(c, c.Credentials) }

// NewInteractionsLLM builds the Gemini Interactions LLM service on Vertex AI. It
// is the same service as interactions.NewLLM, authorized with Google Cloud
// credentials and sent to a Vertex AI project.
func NewInteractionsLLM(cfg InteractionsLLMConfig) *interactions.Service {
	return newInteractionsLLM(cfg, "")
}

// newInteractionsLLM builds the service with its requests sent to host, or to
// the location's endpoint when host is empty.
func newInteractionsLLM(cfg InteractionsLLMConfig, host string) *interactions.Service {
	if cfg.Model == "" {
		cfg.Model = defaultLLMModel
	}
	shaper := &interactionsShaper{
		auth:      &authorizer{creds: cfg.Credentials},
		projectID: cfg.ProjectID,
		location:  location(cfg.Location),
		host:      host,
	}
	return interactions.NewShapedLLM("GoogleInteractionsVertexLLM", shaper, interactions.Config{
		Model:             cfg.Model,
		MaxTokens:         cfg.MaxTokens,
		Temperature:       cfg.Temperature,
		TopP:              cfg.TopP,
		TopK:              cfg.TopK,
		SafetySettings:    cfg.SafetySettings,
		Thinking:          cfg.Thinking,
		Seed:              cfg.Seed,
		Tools:             cfg.Tools,
		ToolConfig:        cfg.ToolConfig,
		StreamIdleTimeout: cfg.StreamIdleTimeout,
		RetryOnTimeout:    cfg.RetryOnTimeout,
		RetryTimeout:      cfg.RetryTimeout,
		Extra:             cfg.Extra,
	})
}

// interactionsShaper addresses and authorizes the Interactions API the Vertex
// way.
type interactionsShaper struct {
	auth      *authorizer
	projectID string
	location  string
	// host overrides the derived endpoint; tests set it.
	host string
}

// Endpoint addresses the API by project and location. The global location is
// served from the unprefixed host.
func (s *interactionsShaper) Endpoint() string {
	host := s.host
	switch {
	case host != "":
	case s.location == defaultLocation:
		host = "https://aiplatform.googleapis.com"
	default:
		host = fmt.Sprintf("https://%s-aiplatform.googleapis.com", s.location)
	}
	return fmt.Sprintf("%s/v1beta1/projects/%s/locations/%s/interactions", host, s.projectID, s.location)
}

// Authorize sets the OAuth bearer token Vertex requires.
func (s *interactionsShaper) Authorize(ctx context.Context, req *http.Request) error {
	return s.auth.authorize(ctx, req)
}
