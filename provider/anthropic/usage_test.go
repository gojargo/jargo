package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// usageReports runs one generation through a service streaming body, with usage
// metrics on, and returns the usage it reported.
func usageReports(t *testing.T, body string) []frames.LLMTokenUsage {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL})

	noInitial := false
	var mu sync.Mutex
	var got []frames.LLMTokenUsage
	done := make(chan struct{})
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableUsageMetrics: true, SendInitialEmptyMetrics: &noInitial},
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f := f.(type) {
		case *frames.MetricsFrame:
			for _, d := range f.Data {
				if u, ok := d.(frames.LLMUsageMetricsData); ok {
					mu.Lock()
					got = append(got, u.Value)
					mu.Unlock()
				}
			}
		case *frames.LLMFullResponseEndFrame:
			close(done)
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	t.Cleanup(func() {
		worker.StopWhenDone()
		<-runDone
	})
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response never ended")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.LLMTokenUsage(nil), got...)
}

func sseEvent(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }

// usageStream is a reply of "Hi" whose message_start reports start, and whose
// message_delta reports delta.
func usageStream(start, delta string) string {
	return sseEvent("message_start", `{"type":"message_start","message":{"id":"m1","type":"message",`+
		`"role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":`+start+`}}`) +
		sseEvent("content_block_start",
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		sseEvent("content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`) +
		sseEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":`+delta+`}`) +
		sseEvent("message_stop", `{"type":"message_stop"}`)
}

// Ported from upstream. message_delta's counts are cumulative, so they replace
// message_start's, and the thinking tokens it reports are the reasoning tokens.
func TestReportsThinkingTokensFromTheFinalUsage(t *testing.T) {
	got := usageReports(t, usageStream(
		`{"input_tokens":40,"output_tokens":2,"cache_creation_input_tokens":0,"cache_read_input_tokens":100}`,
		`{"input_tokens":40,"output_tokens":130,"cache_read_input_tokens":100,`+
			`"output_tokens_details":{"thinking_tokens":120}}`,
	))
	if len(got) != 1 {
		t.Fatalf("got %d usage reports, want one", len(got))
	}
	u := got[0]
	cacheRead, _ := frames.TokenCount(u.CacheReadTokens)
	if u.PromptTokens != 40 || u.CompletionTokens != 130 || cacheRead != 100 || u.TotalTokens != 270 {
		t.Errorf("usage = %+v (cache read %d), want prompt 40, completion 130, cache read 100, total 270", u, cacheRead)
	}
	if reasoning, ok := frames.TokenCount(u.ReasoningTokens); !ok || reasoning != 120 {
		t.Errorf("reasoning tokens = %d (reported %v), want 120", reasoning, ok)
	}
}

// Ported from upstream. A count message_delta leaves out keeps message_start's,
// and a response that reports no thinking tokens has no reasoning count.
func TestKeepsInputCountsAMessageDeltaLeavesOut(t *testing.T) {
	got := usageReports(t, usageStream(
		`{"input_tokens":40,"output_tokens":2,"cache_read_input_tokens":100}`,
		`{"output_tokens":15}`,
	))
	if len(got) != 1 {
		t.Fatalf("got %d usage reports, want one", len(got))
	}
	u := got[0]
	cacheRead, _ := frames.TokenCount(u.CacheReadTokens)
	if u.PromptTokens != 40 || u.CompletionTokens != 15 || cacheRead != 100 {
		t.Errorf("usage = %+v (cache read %d), want prompt 40, completion 15, cache read 100", u, cacheRead)
	}
	if _, ok := frames.TokenCount(u.ReasoningTokens); ok {
		t.Error("reasoning tokens reported, want none")
	}
}

// Ported from upstream (test_llm_token_usage_totals.py, the Bedrock case, which
// runs through this service). A response that counted no tokens at all is not
// reported.
func TestAnEmptyUsageReportIsSkipped(t *testing.T) {
	if got := usageReports(t, usageStream(`{"input_tokens":0,"output_tokens":0}`, `{"output_tokens":0}`)); len(got) != 0 {
		t.Errorf("got %+v, want no report for a usage that counted nothing", got)
	}
}
