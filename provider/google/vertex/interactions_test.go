package vertex

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestInteractionsEndpoint checks the service addresses the Vertex project, at
// the global location by default and on the unprefixed host there.
func TestInteractionsEndpoint(t *testing.T) {
	for _, tc := range []struct{ location, want string }{
		{"", "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/interactions"},
		{"eu", "https://eu-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/eu/interactions"},
	} {
		s := &interactionsShaper{projectID: "test-project", location: location(tc.location)}
		if got := s.Endpoint(); got != tc.want {
			t.Errorf("location %q: endpoint = %s, want %s", tc.location, got, tc.want)
		}
	}
}

// TestInteractionsDefaultModel checks the default model, a model from the
// config replacing it, and the default token cap carried over from the
// Interactions service.
func TestInteractionsDefaultModel(t *testing.T) {
	creds := Credentials{TokenSource: staticSource{"t"}}
	svc := NewInteractionsLLM(InteractionsLLMConfig{ProjectID: "test-project", Credentials: creds})
	if got := svc.Model(); got != "gemini-3.8-flash" {
		t.Errorf("model = %q, want gemini-3.8-flash", got)
	}
	if got := svc.Name(); !strings.HasPrefix(got, "GoogleInteractionsVertexLLM#") {
		t.Errorf("name = %q, want the GoogleInteractionsVertexLLM label", got)
	}
	svc = NewInteractionsLLM(InteractionsLLMConfig{
		ProjectID: "test-project", Credentials: creds, Model: "gemini-3.6-flash",
	})
	if got := svc.Model(); got != "gemini-3.6-flash" {
		t.Errorf("model = %q, want the configured one", got)
	}
}

// TestInteractionsConfigValidate checks a project and credentials are required.
func TestInteractionsConfigValidate(t *testing.T) {
	if err := (InteractionsLLMConfig{ProjectID: "p"}).Validate(); err == nil {
		t.Error("a config without credentials validated")
	}
	if err := (InteractionsLLMConfig{Credentials: Credentials{TokenSource: staticSource{"t"}}}).Validate(); err == nil {
		t.Error("a config without a project validated")
	}
}

// TestInteractionsRequestReachesVertex checks a request carries the bearer token
// the credentials mint, and the Vertex model's thinking default.
func TestInteractionsRequestReachesVertex(t *testing.T) {
	var auth, path string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"event_type":"interaction.completed","interaction":{"status":"completed"}}` + "\n\n"))
	}))
	defer srv.Close()

	svc := newInteractionsLLM(InteractionsLLMConfig{
		ProjectID: "test-project", Location: "us", Credentials: Credentials{TokenSource: staticSource{"tok"}},
	}, srv.URL)
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	_ = svc.Generate(t.Context(), convo, func(string) error { return nil })

	if auth != "Bearer tok" {
		t.Errorf("Authorization = %q, want the minted token", auth)
	}
	if path != "/v1beta1/projects/test-project/locations/us/interactions" {
		t.Errorf("path = %s", path)
	}
	g, _ := body["generation_config"].(map[string]any)
	if g["thinking_level"] != "low" {
		t.Errorf("generation_config = %v, want gemini-3.8-flash's low level", g)
	}
}
