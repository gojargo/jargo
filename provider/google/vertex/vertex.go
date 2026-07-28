// Package vertex provides Google's Gemini models as served by Vertex AI: a
// streaming LLM (NewLLM) and the Live speech-to-speech service (NewS2S). Both
// wrap the corresponding Gemini service, changing only how requests are
// addressed and authorized.
//
// Vertex addresses a model by project and location rather than by name alone,
// and authorizes with a short-lived OAuth access token rather than an API key.
// Supply the credentials explicitly: either the service-account key JSON, or a
// token source the application built itself. jargo reads no environment
// variables, so Application Default Credentials are opt-in through a token
// source:
//
//	creds, err := google.FindDefaultCredentials(ctx, vertex.Scope) // x/oauth2/google
//	svc := vertex.NewLLM(vertex.LLMConfig{
//		ProjectID:   "my-project",
//		Credentials: vertex.Credentials{TokenSource: creds.TokenSource},
//	})
package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/gojargo/jargo/internal/validate"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jwt"
)

// Scope is the OAuth scope Vertex AI requires. An application building its own
// token source should request it.
const Scope = "https://www.googleapis.com/auth/cloud-platform"

const (
	// defaultLocation is the region used when none is configured.
	defaultLocation = "us-east4"
	// defaultLLMModel is the cascaded model used when none is configured.
	defaultLLMModel = "gemini-2.5-flash"
	// defaultLiveModel is the Live model used when none is configured.
	defaultLiveModel = "gemini-live-2.5-flash-native-audio"
)

// errNoCredentials is returned when neither a token source nor service-account
// key JSON was supplied.
//
//nolint:gochecknoglobals // sentinel error
var errNoCredentials = errors.New("vertex: no credentials configured")

// errBadCredentials is returned when the supplied key is not a usable
// service-account key.
//
//nolint:gochecknoglobals // sentinel error
var errBadCredentials = errors.New("vertex: invalid service account key")

// Credentials says how to authorize against Vertex AI. Exactly one field is
// needed; TokenSource wins when both are set.
type Credentials struct {
	// TokenSource mints and refreshes the access token. Use it for anything
	// beyond a service-account key: Application Default Credentials, workload
	// identity, or an impersonated service account.
	TokenSource oauth2.TokenSource
	// JSON is a service-account key, the contents of the key file Google Cloud
	// issues. It is exchanged for a token source scoped to Vertex AI.
	JSON []byte
}

// serviceAccountKey is the subset of a Google service-account key file needed to
// exchange it for an access token.
type serviceAccountKey struct {
	Type        string `json:"type"`
	ClientEmail string `json:"client_email"` //nolint:tagliatelle // Google key-file field
	PrivateKey  string `json:"private_key"`  //nolint:tagliatelle // Google key-file field
	TokenURI    string `json:"token_uri"`    //nolint:tagliatelle // Google key-file field
}

// defaultTokenURI is where a service-account assertion is exchanged for an
// access token when the key file does not say.
//
//nolint:gosec // G101: a public endpoint URL, not a credential
const defaultTokenURI = "https://oauth2.googleapis.com/token"

// tokenSource resolves the configured credentials into a token source. The key
// is exchanged through a signed assertion, so nothing here reads the
// environment: an application wanting Application Default Credentials builds
// that token source itself and passes it in.
func (c Credentials) tokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	if c.TokenSource != nil {
		return c.TokenSource, nil
	}
	if len(c.JSON) == 0 {
		return nil, errNoCredentials
	}
	var key serviceAccountKey
	if err := json.Unmarshal(c.JSON, &key); err != nil {
		return nil, fmt.Errorf("vertex: service account key: %w", err)
	}
	if key.Type != "" && key.Type != "service_account" {
		return nil, fmt.Errorf("%w: key is of type %q, want service_account", errBadCredentials, key.Type)
	}
	if key.ClientEmail == "" || key.PrivateKey == "" {
		return nil, fmt.Errorf("%w: key is missing client_email or private_key", errBadCredentials)
	}
	tokenURI := key.TokenURI
	if tokenURI == "" {
		tokenURI = defaultTokenURI
	}
	cfg := &jwt.Config{
		Email:      key.ClientEmail,
		PrivateKey: []byte(key.PrivateKey),
		TokenURL:   tokenURI,
		Scopes:     []string{Scope},
	}
	return cfg.TokenSource(ctx), nil
}

// Validate reports whether the credentials are usable.
func (c Credentials) Validate() error {
	if c.TokenSource == nil && len(c.JSON) == 0 {
		return errNoCredentials
	}
	return nil
}

// authorizer turns credentials into a bearer token on demand, resolving the
// token source once and letting it handle refreshes from then on.
type authorizer struct {
	creds Credentials

	// mu guards source, which is resolved on first use: a service is built
	// without a context, so the exchange cannot happen in the constructor.
	mu     sync.Mutex
	source oauth2.TokenSource
}

// token returns a current access token.
func (a *authorizer) token(ctx context.Context) (string, error) {
	a.mu.Lock()
	if a.source == nil {
		source, err := a.creds.tokenSource(ctx)
		if err != nil {
			a.mu.Unlock()
			return "", err
		}
		a.source = source
	}
	source := a.source
	a.mu.Unlock()

	// The token source does its own locking and may block on a refresh, so it is
	// called outside the mutex.
	tok, err := source.Token()
	if err != nil {
		return "", fmt.Errorf("vertex: access token: %w", err)
	}
	return tok.AccessToken, nil
}

// authorize sets the bearer token on req.
func (a *authorizer) authorize(ctx context.Context, req *http.Request) error {
	tok, err := a.token(ctx)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	return nil
}

// header builds the request headers a WebSocket handshake needs.
func (a *authorizer) header(ctx context.Context) (http.Header, error) {
	tok, err := a.token(ctx)
	if err != nil {
		return nil, err
	}
	h := http.Header{}
	h.Set("Authorization", "Bearer "+tok)
	return h, nil
}

// location is the configured region, or the default.
func location(configured string) string {
	if configured == "" {
		return defaultLocation
	}
	return configured
}

// modelPath is the resource name Vertex identifies a publisher model by.
func modelPath(projectID, loc, model string) string {
	return fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", projectID, loc, model)
}

// validateConfig runs the struct tags then the credentials check, which spans
// two fields and so cannot be expressed as a tag.
func validateConfig(cfg any, creds Credentials) error {
	if err := validate.Struct(cfg); err != nil {
		return err
	}
	return creds.Validate()
}
