package chat_test

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// TestTTFATStopsAtTheFirstToolCallDelta checks a tool-only turn ends TTFAT at
// the first delta carrying the call, not after its arguments have streamed.
func TestTTFATStopsAtTheFirstToolCallDelta(t *testing.T) {
	const gap = 300 * time.Millisecond
	const first = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\"," +
		"\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"
	const rest = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0," +
		"\"function\":{\"arguments\":\"{}\"}}]}}]}\n\ndata: [DONE]\n\n"
	srv := httptest.NewServer(providertest.StagedStream("text/event-stream", first, rest, gap))
	t.Cleanup(srv.Close)

	got, ok := providertest.TTFATReported(t, chat.NewLLM(chat.LLMConfig{APIKey: "k", BaseURL: srv.URL}))
	if !ok {
		t.Fatal("no TTFAT was reported for a turn answered with a tool call")
	}
	if got.ThinkingTime >= gap/2 {
		t.Errorf("thinking time = %s, want about none: the call was the first output", got.ThinkingTime)
	}
}
