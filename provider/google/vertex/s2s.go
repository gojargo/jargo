package vertex

import (
	"context"
	"fmt"
	"net/http"

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
	// Voice is the prebuilt voice name (e.g. "Puck", "Kore"); empty uses a
	// default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
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
	})
}

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

// ModelPath names the model by project and location, as Vertex requires.
func (c *liveConnector) ModelPath(model string) string {
	return modelPath(c.projectID, c.location, model)
}
