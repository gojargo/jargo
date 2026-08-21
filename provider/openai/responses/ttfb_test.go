package responses

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
)

// ttfbService points a Responses service at a stub streaming the given SSE body.
func ttfbService(t *testing.T, body string) *HTTPService {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL})
}

// Ported from upstream. response.created only acknowledges the request, so TTFB
// runs past it to the first event carrying output: a chunk of text, or the item
// announcing a function call for a turn that produces no text at all.
func TestTTFBEndsAtTheFirstOutputEvent(t *testing.T) {
	const created = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n"
	const completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n"
	const text = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"
	const call = "data: {\"type\":\"response.output_item.added\",\"output_index\":0," +
		"\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\"}}\n\n"

	t.Run("the created event does not end it", func(t *testing.T) {
		if providertest.TTFBReported(t, ttfbService(t, created+completed)) {
			t.Error("TTFB was reported for a stream that carried no model output")
		}
	})

	t.Run("the first text delta ends it", func(t *testing.T) {
		if !providertest.TTFBReported(t, ttfbService(t, created+text+completed)) {
			t.Error("TTFB was not reported for a stream that carried model output")
		}
	})

	t.Run("a turn that only calls a tool still reports it", func(t *testing.T) {
		if !providertest.TTFBReported(t, ttfbService(t, created+call+completed)) {
			t.Error("TTFB was not reported for a turn that only called a tool")
		}
	})
}
