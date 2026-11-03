package vertex

import (
	"context"
	"fmt"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/google/live"
)

// liveBidiPath is the Vertex AI bidirectional-streaming endpoint. Vertex serves
// the Live API under its own service name rather than the Gemini API's.
const liveBidiPath = "/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"

// S2SConfig configures the Vertex AI Gemini Live speech-to-speech service.
type S2SConfig struct {
	// ProjectID is the Google Cloud project serving the model. Required.
	ProjectID string `validate:"required"`
	// Location is the GCP region of the Vertex endpoint (e.g. "us-east4");
	// empty uses a default.
	Location string
	// Credentials authorize the session. Required.
	Credentials Credentials
	// Model is the Live model id; empty uses a current default.
	Model string
	// Voice is the prebuilt voice name (e.g. "Puck", "Kore"); empty uses
	// Charon.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// Tools are tools configured on the service itself, as an alternative to
	// advertising them through the conversation; see live.Config.Tools.
	Tools frames.ToolsSchema

	// The settings below are Gemini Live's; see live.Config for each.

	MaxTokens                *int     `validate:"omitempty,min=1"`
	Temperature              *float64 `validate:"omitempty,min=0,max=2"`
	TopP                     *float64 `validate:"omitempty,min=0,max=1"`
	TopK                     *int     `validate:"omitempty,min=0"`
	FrequencyPenalty         *float64 `validate:"omitempty,min=0,max=2"`
	PresencePenalty          *float64 `validate:"omitempty,min=0,max=2"`
	Seed                     *int
	Modalities               live.Modality `validate:"omitempty,oneof=AUDIO TEXT"`
	Language                 language.Language
	MediaResolution          live.MediaResolution
	VAD                      *live.VADParams
	TurnCoverage             live.TurnCoverage
	ContextWindowCompression *live.ContextWindowCompression
	Thinking                 *gemini.ThinkingConfig `validate:"omitempty"`
	EnableAffectiveDialog    bool
	Proactivity              *live.Proactivity
}

// Validate reports whether the configuration is usable.
func (c S2SConfig) Validate() error { return validateConfig(c, c.Credentials) }

// NewS2S builds a Vertex AI Gemini Live speech-to-speech service. It behaves
// exactly like the Gemini Live service; only the endpoint, the authorization
// and the model's resource name differ.
func NewS2S(cfg S2SConfig) *live.Service {
	if cfg.Model == "" {
		cfg.Model = defaultLiveModel
	}
	conn := &liveConnector{
		auth:      &authorizer{creds: cfg.Credentials},
		projectID: cfg.ProjectID,
		location:  location(cfg.Location),
	}
	return live.NewWithConnector("GoogleVertexLive", conn, live.Config{
		Model:        cfg.Model,
		Voice:        cfg.Voice,
		Instructions: cfg.Instructions,
		Tools:        cfg.Tools,

		MaxTokens:                cfg.MaxTokens,
		Temperature:              cfg.Temperature,
		TopP:                     cfg.TopP,
		TopK:                     cfg.TopK,
		FrequencyPenalty:         cfg.FrequencyPenalty,
		PresencePenalty:          cfg.PresencePenalty,
		Seed:                     cfg.Seed,
		Modalities:               cfg.Modalities,
		Language:                 cfg.Language,
		MediaResolution:          cfg.MediaResolution,
		VAD:                      cfg.VAD,
		TurnCoverage:             cfg.TurnCoverage,
		ContextWindowCompression: cfg.ContextWindowCompression,
		Thinking:                 cfg.Thinking,
		EnableAffectiveDialog:    cfg.EnableAffectiveDialog,
		Proactivity:              cfg.Proactivity,
	})
}

// The Vertex endpoint decides non-blocking support itself, whatever the model.
var _ live.NonBlockingToolsGate = (*liveConnector)(nil)

// liveConnector dials the Vertex AI Live endpoint with an OAuth token.
type liveConnector struct {
	auth      *authorizer
	projectID string
	location  string
	// host overrides the derived regional endpoint; tests set it.
	host string
}

// Endpoint returns the regional Live endpoint and a bearer token to dial it
// with. Vertex authorizes on the handshake rather than with a query parameter.
func (c *liveConnector) Endpoint(ctx context.Context) (string, http.Header, error) {
	header, err := c.auth.header(ctx)
	if err != nil {
		return "", nil, err
	}
	host := c.host
	if host == "" {
		host = fmt.Sprintf("wss://%s-aiplatform.googleapis.com", c.location)
	}
	return host + liveBidiPath, header, nil
}

// SupportsNonBlockingTools reports false whatever the model: the Vertex AI Live
// endpoint does not yet support non-blocking tool calls. Sending the NON_BLOCKING
// behavior on a function declaration, or scheduling on a function response,
// appears to break tool calling against Vertex. A tool registered as not
// canceled on interruption therefore runs as a call the model waits for.
func (*liveConnector) SupportsNonBlockingTools(string) bool { return false }

// ModelPath names the model by project and location, as Vertex requires.
func (c *liveConnector) ModelPath(model string) string {
	return modelPath(c.projectID, c.location, model)
}
