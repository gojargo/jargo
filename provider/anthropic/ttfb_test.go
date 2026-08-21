package anthropic

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
)

// ttfbService points an Anthropic service at a stub streaming the given SSE
// body. Anthropic names each event in its own `event:` line, which the SDK
// reads to pick the type.
func ttfbService(t *testing.T, body string) *Service {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return NewLLM(Config{APIKey: "k", BaseURL: srv.URL})
}

const (
	// messageStart opens the stream and carries no model output.
	messageStart = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\"," +
		"\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"stop_reason\":null," +
		"\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n"
	ping = "event: ping\ndata: {\"type\":\"ping\"}\n\n"

	textBlockStart = "event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0," +
		"\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
	textDelta = "event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0," +
		"\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"

	thinkingBlockStart = "event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0," +
		"\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n"

	messageStop = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
)

// Ported from upstream. The events that open the stream (message_start, ping)
// carry no model output, so TTFB runs past them to the first content block.
func TestTTFBEndsAtTheFirstContentBlock(t *testing.T) {
	t.Run("message_start does not end it", func(t *testing.T) {
		if providertest.TTFBReported(t, ttfbService(t, messageStart+ping+messageStop)) {
			t.Error("TTFB was reported for a stream that carried no model output")
		}
	})

	t.Run("the first content block ends it", func(t *testing.T) {
		body := messageStart + ping + textBlockStart + textDelta + messageStop
		if !providertest.TTFBReported(t, ttfbService(t, body)) {
			t.Error("TTFB was not reported for a stream that carried model output")
		}
	})

	// Reasoning is output, so it ends TTFB just as answer text would.
	t.Run("a thinking block before any answer text ends it", func(t *testing.T) {
		body := messageStart + thinkingBlockStart + messageStop
		if !providertest.TTFBReported(t, ttfbService(t, body)) {
			t.Error("TTFB was not reported for a stream that opened with reasoning")
		}
	})
}
