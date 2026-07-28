package vertex

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/oauth2"
)

// staticSource is a token source that hands out a fixed token, standing in for
// whatever an application builds.
type staticSource struct{ token string }

func (s staticSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: s.token}, nil
}

// failingSource stands in for a token source that cannot mint a token.
type failingSource struct{}

//nolint:gochecknoglobals // sentinel error used only by the failing test source
var errNoToken = errors.New("token source unavailable")

func (failingSource) Token() (*oauth2.Token, error) { return nil, errNoToken }

// TestCredentialsValidate checks a configuration with no way to authorize is
// rejected before it reaches the network.
func TestCredentialsValidate(t *testing.T) {
	if err := (Credentials{}).Validate(); !errors.Is(err, errNoCredentials) {
		t.Errorf("Validate() = %v, want errNoCredentials", err)
	}
	if err := (Credentials{TokenSource: staticSource{"t"}}).Validate(); err != nil {
		t.Errorf("Validate() with a token source = %v, want nil", err)
	}
	if err := (Credentials{JSON: []byte("{}")}).Validate(); err != nil {
		t.Errorf("Validate() with key JSON = %v, want nil", err)
	}
}

// TestCredentialsTokenSource checks how each credential form resolves, and that
// a malformed service-account key is rejected with a useful message.
func TestCredentialsTokenSource(t *testing.T) {
	ctx := context.Background()

	t.Run("token source wins over key JSON", func(t *testing.T) {
		creds := Credentials{TokenSource: staticSource{"explicit"}, JSON: []byte(`{"type":"service_account"}`)}
		src, err := creds.tokenSource(ctx)
		if err != nil {
			t.Fatalf("tokenSource: %v", err)
		}
		tok, err := src.Token()
		if err != nil {
			t.Fatalf("Token: %v", err)
		}
		if tok.AccessToken != "explicit" {
			t.Errorf("token = %q, want the configured source's", tok.AccessToken)
		}
	})

	t.Run("no credentials", func(t *testing.T) {
		if _, err := (Credentials{}).tokenSource(ctx); !errors.Is(err, errNoCredentials) {
			t.Errorf("tokenSource() = %v, want errNoCredentials", err)
		}
	})

	t.Run("malformed JSON", func(t *testing.T) {
		if _, err := (Credentials{JSON: []byte("not json")}).tokenSource(ctx); err == nil {
			t.Error("tokenSource() accepted a key that is not JSON")
		}
	})

	t.Run("wrong credential type", func(t *testing.T) {
		key := []byte(`{"type":"authorized_user","client_email":"a@b","private_key":"k"}`)
		_, err := (Credentials{JSON: key}).tokenSource(ctx)
		if !errors.Is(err, errBadCredentials) {
			t.Errorf("tokenSource() = %v, want errBadCredentials", err)
		}
	})

	t.Run("missing fields", func(t *testing.T) {
		key := []byte(`{"type":"service_account","client_email":"a@b"}`)
		_, err := (Credentials{JSON: key}).tokenSource(ctx)
		if !errors.Is(err, errBadCredentials) {
			t.Errorf("tokenSource() = %v, want errBadCredentials", err)
		}
	})
}

// TestAuthorizerFailureSurfaces checks a token that cannot be minted fails the
// request rather than sending it unauthorized.
func TestAuthorizerFailureSurfaces(t *testing.T) {
	a := &authorizer{creds: Credentials{TokenSource: failingSource{}}}
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, "https://example", nil)
	if err := a.authorize(context.Background(), req); err == nil {
		t.Fatal("authorize() = nil, want the token failure surfaced")
	}
	if req.Header.Get("Authorization") != "" {
		t.Error("an Authorization header was set despite the token failing")
	}
}

// TestLLMConfigValidate pins which LLMConfig fields the provider requires.
func TestLLMConfigValidate(t *testing.T) {
	creds := Credentials{TokenSource: staticSource{"t"}}
	cases := []struct {
		name  string
		cfg   LLMConfig
		valid bool
	}{
		{"project and credentials", LLMConfig{ProjectID: "p", Credentials: creds}, true},
		{"missing project", LLMConfig{Credentials: creds}, false},
		{"missing credentials", LLMConfig{ProjectID: "p"}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.cfg.Validate()
			if c.valid && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if !c.valid && err == nil {
				t.Error("Validate() = nil, want an error")
			}
		})
	}
}

// TestS2SConfigValidate pins which S2SConfig fields the provider requires.
func TestS2SConfigValidate(t *testing.T) {
	creds := Credentials{TokenSource: staticSource{"t"}}
	if err := (S2SConfig{ProjectID: "p", Credentials: creds}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
	if err := (S2SConfig{Credentials: creds}).Validate(); err == nil {
		t.Error("Validate() without a project = nil, want an error")
	}
	if err := (S2SConfig{ProjectID: "p"}).Validate(); err == nil {
		t.Error("Validate() without credentials = nil, want an error")
	}
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	creds := Credentials{TokenSource: staticSource{"t"}}
	llm := NewLLM(LLMConfig{ProjectID: "p", Credentials: creds})
	if got := llm.Name(); !strings.HasPrefix(got, "GoogleVertexLLM#") {
		t.Errorf("LLM name = %q, want the GoogleVertexLLM label", got)
	}
	s2s := NewS2S(S2SConfig{ProjectID: "p", Credentials: creds})
	if got := s2s.Name(); !strings.HasPrefix(got, "GoogleVertexLive#") {
		t.Errorf("S2S name = %q, want the GoogleVertexLive label", got)
	}
}

// TestModelPath checks the resource name Vertex identifies a model by.
func TestModelPath(t *testing.T) {
	got := modelPath("my-project", "us-east4", "gemini-2.5-flash")
	want := "projects/my-project/locations/us-east4/publishers/google/models/gemini-2.5-flash"
	if got != want {
		t.Errorf("modelPath = %q, want %q", got, want)
	}
}

// TestLocationDefault checks the region falls back when unset.
func TestLocationDefault(t *testing.T) {
	if got := location(""); got != defaultLocation {
		t.Errorf("location(\"\") = %q, want %q", got, defaultLocation)
	}
	if got := location("europe-west4"); got != "europe-west4" {
		t.Errorf("location = %q, want the configured region", got)
	}
}

// TestLLMShaperEndpoint checks the LLM addresses the regional endpoint by
// project and location, which is what distinguishes Vertex from the Gemini API.
func TestLLMShaperEndpoint(t *testing.T) {
	s := &llmShaper{projectID: "my-project", location: "europe-west4"}
	got := s.Endpoint("gemini-2.5-flash")
	want := "https://europe-west4-aiplatform.googleapis.com/v1/" +
		"projects/my-project/locations/europe-west4/publishers/google/models/" +
		"gemini-2.5-flash:streamGenerateContent?alt=sse"
	if got != want {
		t.Errorf("Endpoint =\n  %q\nwant\n  %q", got, want)
	}
}

// TestLLMShaperAuthorize checks the OAuth bearer token reaches the request.
func TestLLMShaperAuthorize(t *testing.T) {
	s := &llmShaper{auth: &authorizer{creds: Credentials{TokenSource: staticSource{"tok-123"}}}}
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://example", nil)
	if err := s.Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer tok-123" {
		t.Errorf("Authorization = %q, want the bearer token", got)
	}
}

// TestLiveConnector checks the Live session dials the regional endpoint with a
// bearer token and names the model by its full resource path.
func TestLiveConnector(t *testing.T) {
	c := &liveConnector{
		auth:      &authorizer{creds: Credentials{TokenSource: staticSource{"tok-456"}}},
		projectID: "my-project",
		location:  "us-east4",
	}

	endpoint, header, err := c.Endpoint(context.Background())
	if err != nil {
		t.Fatalf("Endpoint: %v", err)
	}
	want := "wss://us-east4-aiplatform.googleapis.com" + liveBidiPath
	if endpoint != want {
		t.Errorf("endpoint = %q, want %q", endpoint, want)
	}
	if got := header.Get("Authorization"); got != "Bearer tok-456" {
		t.Errorf("Authorization = %q, want the bearer token", got)
	}
	if strings.Contains(endpoint, "key=") {
		t.Error("endpoint carries an api key, but Vertex authorizes on the handshake")
	}

	got := c.ModelPath("gemini-live-2.5-flash-native-audio")
	wantPath := "projects/my-project/locations/us-east4/publishers/google/models/" +
		"gemini-live-2.5-flash-native-audio"
	if got != wantPath {
		t.Errorf("ModelPath = %q, want %q", got, wantPath)
	}
}

// TestLiveConnectorTokenFailure checks a session is not dialed unauthorized when
// the token cannot be minted.
func TestLiveConnectorTokenFailure(t *testing.T) {
	c := &liveConnector{auth: &authorizer{creds: Credentials{TokenSource: failingSource{}}}}
	if _, _, err := c.Endpoint(context.Background()); err == nil {
		t.Fatal("Endpoint() = nil error, want the token failure surfaced")
	}
}

// TestLLMRequestReachesVertex drives the LLM against a fake regional endpoint
// and checks the request is addressed and authorized the Vertex way.
func TestLLMRequestReachesVertex(t *testing.T) {
	var gotPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n"))
	}))
	defer srv.Close()

	shaper := &llmShaper{
		auth:      &authorizer{creds: Credentials{TokenSource: staticSource{"tok-789"}}},
		projectID: "my-project",
		location:  "us-east4",
		host:      srv.URL,
	}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, shaper.Endpoint("gemini-2.5-flash"), nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if err := shaper.Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	wantPath := "/v1/projects/my-project/locations/us-east4/publishers/google/models/" +
		"gemini-2.5-flash:streamGenerateContent?alt=sse"
	if gotPath != wantPath {
		t.Errorf("request path = %q, want %q", gotPath, wantPath)
	}
	if gotAuth != "Bearer tok-789" {
		t.Errorf("Authorization = %q, want the bearer token", gotAuth)
	}
}
