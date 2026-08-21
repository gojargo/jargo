package chat_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// ttfbService points an LLM service at a stub streaming the given SSE body.
func ttfbService(t *testing.T, body string) *chat.LLMService {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return chat.NewLLM(chat.LLMConfig{APIKey: "k", BaseURL: srv.URL})
}

// Ported from upstream. The chunk carrying the token counts has no choices of
// its own, so it is not the model answering. TTFB has to run past it to the
// first chunk that holds output.
func TestTTFBEndsAtTheFirstOutputChunk(t *testing.T) {
	const usageOnly = "data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":0,\"total_tokens\":10}}\n\n"
	const text = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]\n\n"

	t.Run("a usage-only chunk does not end it", func(t *testing.T) {
		if providertest.TTFBReported(t, ttfbService(t, usageOnly+"data: [DONE]\n\n")) {
			t.Error("TTFB was reported for a stream that carried no model output")
		}
	})

	t.Run("the first output chunk ends it", func(t *testing.T) {
		if !providertest.TTFBReported(t, ttfbService(t, usageOnly+text)) {
			t.Error("TTFB was not reported for a stream that carried model output")
		}
	})
}
