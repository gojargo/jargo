package responses

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/providertest"
)

// TestTTFATStopsAtTheFunctionCallItem checks a tool-only turn ends TTFAT at the
// item announcing the call, not after its arguments have streamed.
func TestTTFATStopsAtTheFunctionCallItem(t *testing.T) {
	const gap = 300 * time.Millisecond
	const first = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n" +
		"data: {\"type\":\"response.output_item.added\",\"output_index\":0," +
		"\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"get_weather\"}}\n\n"
	const rest = "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0," +
		"\"arguments\":\"{}\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n"
	srv := httptest.NewServer(providertest.StagedStream("text/event-stream", first, rest, gap))
	t.Cleanup(srv.Close)

	got, ok := providertest.TTFATReported(t, NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL}))
	if !ok {
		t.Fatal("no TTFAT was reported for a turn answered with a tool call")
	}
	if got.ThinkingTime >= gap/2 {
		t.Errorf("thinking time = %s, want about none: the call was the first output", got.ThinkingTime)
	}
}
