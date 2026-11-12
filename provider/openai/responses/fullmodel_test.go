package responses

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// TestTheModelTheResponseNamesLabelsTheNextSpans checks the model a completed
// response names, a dated build for an alias, is what the spans after it are
// labeled with, as the chat service does.
func TestTheModelTheResponseNamesLabelsTheNextSpans(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n" +
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4.1-2025-04-14\"}}\n\n"))
	}))
	t.Cleanup(srv.Close)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL, Model: "gpt-4.1"})

	ended := make(chan struct{}, 2)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		EnableTracing: true, ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			ended <- struct{}{}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for range 2 {
		convo := frames.NewLLMContext("be brief")
		convo.AddUserMessage("hello")
		task.QueueFrame(frames.NewLLMContextFrame(convo))
		select {
		case <-ended:
		case <-time.After(5 * time.Second):
			t.Fatal("the response never ended")
		}
	}
	task.StopWhenDone()
	<-done

	var models []string
	for _, s := range rec.Ended() {
		if s.Name() != "llm" {
			continue
		}
		for _, kv := range s.Attributes() {
			if kv.Key == "gen_ai.request.model" {
				models = append(models, kv.Value.AsString())
			}
		}
	}
	if len(models) != 2 || models[1] != "gpt-4.1-2025-04-14" {
		t.Errorf("span models = %v, want the second labeled gpt-4.1-2025-04-14", models)
	}
}
