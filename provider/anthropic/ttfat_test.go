package anthropic

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/providertest"
)

const ttfatGap = 300 * time.Millisecond

const (
	toolUseStart = "event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0," +
		"\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"get_weather\",\"input\":{}}}\n\n"
	toolUseArgs = "event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0," +
		"\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n"
	toolUseStop = "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	messageEnd  = "event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
)

// Ported from upstream (test_anthropic_stops_at_the_tool_call_not_end_of_stream).
// A tool-only turn ends TTFAT where the tool_use block starts; the arguments
// streaming after it are not thinking.
func TestTTFATStopsAtTheToolCallNotEndOfStream(t *testing.T) {
	srv := httptest.NewServer(providertest.StagedStream("text/event-stream",
		messageStart+toolUseStart, toolUseArgs+toolUseStop+messageEnd, ttfatGap))
	t.Cleanup(srv.Close)

	got, ok := providertest.TTFATReported(t, NewLLM(Config{APIKey: "k", BaseURL: srv.URL}))
	if !ok {
		t.Fatal("no TTFAT was reported for a turn answered with a tool call")
	}
	if got.ThinkingTime >= ttfatGap/2 {
		t.Errorf("thinking time = %s, want about none: the call started with the first output", got.ThinkingTime)
	}
}
