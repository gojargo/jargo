package vertex

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gojargo/jargo/provider/google/gemini"
)

// LLMConfig configures the Vertex AI Gemini LLM service. The generation
// controls mirror the Gemini service's; only the addressing and authorization
// differ.
type LLMConfig struct {
	// ProjectID is the Google Cloud project serving the model. Required.
	ProjectID string `validate:"required"`
	// Location is the GCP region of the Vertex endpoint (e.g. "us-east4");
	// empty uses a default.
	Location string
	// Credentials authorize the requests. Required.
	Credentials Credentials
	// Model is the model id; empty uses a low-latency flash default.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// TopK is the top-k sampling parameter; nil omits it.
	TopK *int
	// SafetySettings are the content-safety filters, one per harm category.
	// Empty sends none, leaving every category at the API's default.
	SafetySettings []gemini.SafetySetting `validate:"omitempty,dive"`
	// Thinking controls the model's internal reasoning; nil applies the
	// low-latency default for the model. See gemini.Config.
	Thinking *gemini.ThinkingConfig `validate:"omitempty"`
	// Seed makes sampling repeatable; nil omits it.
	Seed *int
	// StreamIdleTimeout bounds the gap between two chunks of a streamed
	// response; nil uses 20 seconds and a zero value waits indefinitely.
	StreamIdleTimeout *time.Duration
	// RetryOnTimeout re-issues a request once when its first chunk does not
	// arrive within RetryTimeout.
	RetryOnTimeout bool
	// RetryTimeout is how long the first chunk is waited for before the request
	// is re-issued; 0 uses 5 seconds.
	RetryTimeout time.Duration `validate:"omitempty,min=0"`
	// Extra sets arbitrary additional generationConfig fields not modeled above,
	// applied to every request.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c LLMConfig) Validate() error { return validateConfig(c, c.Credentials) }

// NewLLM builds a Vertex AI Gemini LLM service.
func NewLLM(cfg LLMConfig) *gemini.Service {
	if cfg.Model == "" {
		cfg.Model = defaultLLMModel
	}
	shaper := &llmShaper{
		auth:      &authorizer{creds: cfg.Credentials},
		projectID: cfg.ProjectID,
		location:  location(cfg.Location),
	}
	return gemini.NewShapedLLM("GoogleVertexLLM", shaper, gemini.Config{
		Model:             cfg.Model,
		MaxTokens:         cfg.MaxTokens,
		Temperature:       cfg.Temperature,
		TopP:              cfg.TopP,
		TopK:              cfg.TopK,
		SafetySettings:    cfg.SafetySettings,
		Thinking:          cfg.Thinking,
		Seed:              cfg.Seed,
		StreamIdleTimeout: cfg.StreamIdleTimeout,
		RetryOnTimeout:    cfg.RetryOnTimeout,
		RetryTimeout:      cfg.RetryTimeout,
		Extra:             cfg.Extra,
	})
}

// llmShaper addresses and authorizes generateContent the Vertex way.
type llmShaper struct {
	auth      *authorizer
	projectID string
	location  string
	// host overrides the derived regional endpoint; tests set it.
	host string
}

// Endpoint addresses the model by project and location on the regional
// endpoint, rather than by name on the shared Gemini API.
func (s *llmShaper) Endpoint(model string, stream bool) string {
	host := s.host
	if host == "" {
		host = fmt.Sprintf("https://%s-aiplatform.googleapis.com", s.location)
	}
	if !stream {
		return fmt.Sprintf("%s/v1/%s:generateContent", host, modelPath(s.projectID, s.location, model))
	}
	return fmt.Sprintf(
		"%s/v1/%s:streamGenerateContent?alt=sse",
		host, modelPath(s.projectID, s.location, model),
	)
}

// Authorize sets the OAuth bearer token Vertex requires.
func (s *llmShaper) Authorize(ctx context.Context, req *http.Request) error {
	return s.auth.authorize(ctx, req)
}
