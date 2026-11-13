package ollama_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// TestNewLLM checks the Ollama shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "OllamaLLM", "llama2", ollama.NewLLM)
}

// TestLocalOllamaNeedsNoAPIKey checks a service built with no key still sends a
// well-formed one, the placeholder a local server ignores, rather than an empty
// Bearer token.
func TestLocalOllamaNeedsNoAPIKey(t *testing.T) {
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	svc := ollama.NewLLM(chat.LLMConfig{BaseURL: srv.URL})
	if err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if auth != "Bearer ollama" {
		t.Errorf("Authorization = %q, want the placeholder key", auth)
	}
}
