package gemini

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/providertest"
)

const ttfatGap = 300 * time.Millisecond

const (
	callChunk = "data: {\"candidates\":[{\"content\":{\"role\":\"model\"," +
		"\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{}}}]}}]}\n\n"
	emptyChunk  = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"\"}]}}]}\n\n"
	answerChunk = "data: {\"candidates\":[{\"content\":{\"role\":\"model\"," +
		"\"parts\":[{\"text\":\"Let me look that up.\"}]}}]}\n\n"
)

// ttfatService points a Gemini service at a stub streaming first, then rest
// after the gap.
func ttfatService(t *testing.T, first, rest string) *Service {
	t.Helper()
	srv := httptest.NewServer(providertest.StagedStream("text/event-stream", first, rest, ttfatGap))
	t.Cleanup(srv.Close)
	svc := NewLLM(Config{APIKey: "k"})
	svc.shaper = &testShaper{base: srv.URL}
	return svc
}

// Ported from upstream (test_google_stops_at_the_tool_call_not_end_of_stream).
// A tool-only turn ends TTFAT at the chunk carrying the call, not at the end of
// the trailing chunks.
func TestTTFATStopsAtTheToolCallNotEndOfStream(t *testing.T) {
	got, ok := providertest.TTFATReported(t, ttfatService(t, callChunk, emptyChunk+emptyChunk))
	if !ok {
		t.Fatal("no TTFAT was reported for a turn answered with a tool call")
	}
	if got.ThinkingTime >= ttfatGap/2 {
		t.Errorf("thinking time = %s, want about none: the call came with the first output", got.ThinkingTime)
	}
}

// Ported from upstream (test_answer_text_before_a_tool_call_wins). Text and
// tool calls share one measurement, and whichever comes first ends it.
func TestTTFATAnswerTextBeforeAToolCallWins(t *testing.T) {
	got, ok := providertest.TTFATReported(t, ttfatService(t, answerChunk, callChunk))
	if !ok {
		t.Fatal("no TTFAT was reported")
	}
	if got.ThinkingTime >= ttfatGap/2 {
		t.Errorf("thinking time = %s, want about none: the text came first", got.ThinkingTime)
	}
}
