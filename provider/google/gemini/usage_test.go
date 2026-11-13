package gemini

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// responseFrames runs one turn through a service streaming reply, with usage
// metrics collected, and returns every frame that reached the end of the
// pipeline up to the end of the response, in order.
func responseFrames(t *testing.T, reply string) []frames.Frame {
	t.Helper()
	srv := newGenServer(t, reply)
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})

	var mu sync.Mutex
	var got []frames.Frame
	done := make(chan struct{})
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableUsageMetrics: true},
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, f)
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			close(done)
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	t.Cleanup(func() {
		worker.StopWhenDone()
		<-runDone
	})

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response never ended")
	}
	mu.Lock()
	defer mu.Unlock()
	return got
}

// reportedUsage runs one turn streaming reply and returns every token-usage
// report it made.
func reportedUsage(t *testing.T, reply string) []frames.LLMTokenUsage {
	t.Helper()
	var reported []frames.LLMTokenUsage
	for _, f := range responseFrames(t, reply) {
		m, ok := f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range m.Data {
			if u, ok := d.(frames.LLMUsageMetricsData); ok {
				reported = append(reported, u.Value)
			}
		}
	}
	return reported
}

// TestThinkingTokensAreCountedAsCompletionTokens checks the thinking Gemini
// counts apart from the candidates is reported as output, which it is.
func TestThinkingTokensAreCountedAsCompletionTokens(t *testing.T) {
	reported := reportedUsage(t, sse(
		`{"candidates":[{"content":{"parts":[{"text":"Hi!"}]}}],`+
			`"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":10,`+
			`"thoughtsTokenCount":120,"totalTokenCount":170}}`,
	))

	if len(reported) != 1 {
		t.Fatalf("reports = %+v, want exactly one", reported)
	}
	got := reported[0]
	if got.CompletionTokens != 130 {
		t.Errorf("completion tokens = %d, want the candidates and the thinking", got.CompletionTokens)
	}
	if n, ok := frames.TokenCount(got.ReasoningTokens); !ok || n != 120 {
		t.Errorf("reasoning tokens = %d (reported: %v), want 120", n, ok)
	}
	if got.TotalTokens != got.PromptTokens+got.CompletionTokens {
		t.Errorf("usage = %+v, want the total to be the prompt and the completion", got)
	}
}

// TestToolResultsAreCountedAsPromptTokens checks the code execution, URL context
// and search results fed back to the model are reported as input, which they
// are.
func TestToolResultsAreCountedAsPromptTokens(t *testing.T) {
	reported := reportedUsage(t, sse(
		`{"candidates":[{"content":{"parts":[{"text":"5117"}]}}],`+
			`"usageMetadata":{"promptTokenCount":23,"candidatesTokenCount":57,`+
			`"toolUsePromptTokenCount":79,"totalTokenCount":159}}`,
	))

	if len(reported) != 1 {
		t.Fatalf("reports = %+v, want exactly one", reported)
	}
	got := reported[0]
	if got.PromptTokens != 102 || got.CompletionTokens != 57 || got.TotalTokens != 159 {
		t.Errorf("usage = %+v, want 102 in / 57 out / 159 total", got)
	}
}
