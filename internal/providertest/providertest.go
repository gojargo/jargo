// Package providertest holds shared assertions for provider packages whose
// behavior is fully determined by the base service they delegate to.
//
// A dozen providers are nothing but a name, a base URL and a default model
// handed to chat.NewCompatLLM. Testing each one separately would be a dozen
// copies of the same httptest server; testing them all from one package would
// leave every one of them reporting zero coverage, because Go credits a package
// only for the tests that live in it. So the assertions live here and each
// provider keeps a small test that calls them.
package providertest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// CompatLLMBuilder builds an OpenAI-compatible LLM service from a config, i.e.
// the NewLLM function every compatible provider exposes.
type CompatLLMBuilder func(chat.LLMConfig) *chat.LLMService

// CompatLLM checks that build produces a usable OpenAI-compatible LLM service:
// it is named wantName, defaults to wantModel, lets a caller override the model,
// addresses the chat-completions endpoint under the configured base URL, sends a
// Bearer key, and streams the response back.
//
// It deliberately does not assert the provider's own default base URL — that
// constant names a host the tests must never contact. What it does assert is
// that a configured base URL is honored, which is the half a provider can get
// wrong without anyone noticing.
func CompatLLM(t *testing.T, wantName, wantModel string, build CompatLLMBuilder) {
	t.Helper()

	t.Run("defaults", func(t *testing.T) {
		svc := build(chat.LLMConfig{APIKey: "test-key"})
		if svc == nil {
			t.Fatal("NewLLM returned nil")
		}
		// processor.New appends a "#<id>" instance counter, so only the label
		// the provider chose is stable across runs.
		if got := svc.Name(); !strings.HasPrefix(got, wantName+"#") {
			t.Errorf("Name() = %q, want the %q label", got, wantName)
		}
	})

	t.Run("streams a completion", func(t *testing.T) {
		req := captureRequest(t, build, chat.LLMConfig{APIKey: "test-key"}, "Hello there")

		if req.path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", req.path)
		}
		if got := req.header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want the Bearer key", got)
		}
		if got := req.header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		if req.body.Model != wantModel {
			t.Errorf("model = %q, want the provider default %q", req.body.Model, wantModel)
		}
		if !req.body.Stream {
			t.Error("stream = false, want true")
		}
		if req.reply != "Hello there" {
			t.Errorf("streamed reply = %q, want %q", req.reply, "Hello there")
		}
	})

	t.Run("model override", func(t *testing.T) {
		req := captureRequest(t, build, chat.LLMConfig{APIKey: "k", Model: "override-model"}, "hi")
		if req.body.Model != "override-model" {
			t.Errorf("model = %q, want the override", req.body.Model)
		}
	})
}

// capturedRequest is what the fake endpoint saw, plus what the service streamed
// back to its caller.
type capturedRequest struct {
	path   string
	header http.Header
	body   struct {
		Model    string `json:"model"`
		Stream   bool   `json:"stream"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	reply string
}

// captureRequest points the service at a fake chat-completions endpoint, runs one
// generation, and reports what crossed the wire in both directions.
func captureRequest(t *testing.T, build CompatLLMBuilder, cfg chat.LLMConfig, reply string) capturedRequest {
	t.Helper()

	var got capturedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.header = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&got.body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}

		chunk, err := json.Marshal(map[string]any{
			"choices": []map[string]any{{"delta": map[string]string{"content": reply}}},
		})
		if err != nil {
			t.Errorf("encoding response chunk: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + string(chunk) + "\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	svc := build(cfg)
	if svc == nil {
		t.Fatal("NewLLM returned nil")
	}

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var out strings.Builder
	if err := svc.Generate(t.Context(), convo, func(text string) error {
		out.WriteString(text)
		return nil
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got.reply = out.String()
	return got
}

// Validator is the shape every provider config shares: a Validate that reports
// whether the config is usable before anything tries to connect with it.
type Validator interface {
	Validate() error
}

// ConfigCase is one configuration and whether Validate should accept it.
type ConfigCase struct {
	// Name describes what the case exercises, e.g. "missing API key".
	Name string
	// Cfg is the configuration under test.
	Cfg Validator
	// Valid is whether Validate should return nil.
	Valid bool
}

// Configs runs each case through Validate. It is the check that a provider's
// `validate` struct tags actually say what the field comments claim: a required
// credential must be rejected when absent, and an otherwise-empty config must be
// accepted when the provider documents defaults for everything else.
func Configs(t *testing.T, cases []ConfigCase) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			err := c.Cfg.Validate()
			if c.Valid && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if !c.Valid && err == nil {
				t.Error("Validate() = nil, want an error")
			}
		})
	}
}

// namer is satisfied by every service, which embeds a processor.
type namer interface {
	Name() string
}

// Service checks a constructor returned a usable service carrying wantLabel.
// Constructors take no error return, so a misconfigured provider surfaces only
// as a nil service or a mislabeled processor in the logs and traces.
func Service(t *testing.T, wantLabel string, svc namer) {
	t.Helper()
	// A nil interface means the constructor returned nothing; a non-nil
	// interface holding a nil pointer would panic on Name, which is also a
	// failure worth catching here rather than in a pipeline.
	if svc == nil {
		t.Fatalf("constructor returned nil, want a %s service", wantLabel)
	}
	// processor.New appends a "#<id>" instance counter.
	if got := svc.Name(); !strings.HasPrefix(got, wantLabel+"#") {
		t.Errorf("Name() = %q, want the %q label", got, wantLabel)
	}
}
