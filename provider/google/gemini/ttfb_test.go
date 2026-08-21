package gemini

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
)

// ttfbService points an LLM service at a stub streaming the given SSE body.
func ttfbService(t *testing.T, body string) *Service {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(Config{APIKey: "k"})
	svc.shaper = &testShaper{base: srv.URL}
	return svc
}

// Ported from upstream. Gemini can open a stream with a chunk carrying only
// usage metadata and no candidates. That chunk is not the model answering, so
// TTFB has to run past it to the first chunk that holds output; stopping on it
// would report a latency the model never achieved.
func TestTTFBEndsAtTheFirstOutputChunk(t *testing.T) {
	const usageOnly = "data: {\"usageMetadata\":{\"promptTokenCount\":10}}\n\n"
	const text = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n"

	t.Run("a usage-only chunk does not end it", func(t *testing.T) {
		if providertest.TTFBReported(t, ttfbService(t, usageOnly)) {
			t.Error("TTFB was reported for a stream that carried no model output")
		}
	})

	t.Run("the first output chunk ends it", func(t *testing.T) {
		if !providertest.TTFBReported(t, ttfbService(t, usageOnly+text)) {
			t.Error("TTFB was not reported for a stream that carried model output")
		}
	})
}
