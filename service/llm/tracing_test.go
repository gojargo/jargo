package llm_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestGenerationEmitsSpan(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	gen := &fakeGen{deltas: []string{"Hi"}}
	svc := llm.New("FakeLLM", gen)
	svc.SetModel("test-model")

	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("llm did not complete")
	}
	task.StopWhenDone()
	<-runDone

	var span sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == "llm" {
			span = s
		}
	}
	if span == nil {
		t.Fatalf("no llm span recorded; got %d spans", len(rec.Ended()))
	}
	attrs := map[string]string{}
	for _, kv := range span.Attributes() {
		attrs[string(kv.Key)] = kv.Value.String()
	}
	if attrs["llm.model"] != "test-model" {
		t.Fatalf("llm.model = %q, want test-model (attrs: %v)", attrs["llm.model"], attrs)
	}
	if attrs["llm.service"] == "" {
		t.Fatalf("missing llm.service attribute (attrs: %v)", attrs)
	}
	// gen_ai.* + input/output so a backend (Langfuse) renders the generation.
	if attrs["gen_ai.request.model"] != "test-model" {
		t.Fatalf("gen_ai.request.model = %q, want test-model", attrs["gen_ai.request.model"])
	}
	if attrs["gen_ai.operation.name"] != "chat" {
		t.Fatalf("gen_ai.operation.name = %q, want chat", attrs["gen_ai.operation.name"])
	}
	if attrs["output"] != "Hi" {
		t.Fatalf("output = %q, want %q", attrs["output"], "Hi")
	}
	if !strings.Contains(attrs["input"], "be brief") || !strings.Contains(attrs["input"], "hi") {
		t.Fatalf("input missing system/user content: %q", attrs["input"])
	}
}
