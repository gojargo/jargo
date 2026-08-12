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
		EnableTracing:           true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
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
	for key, want := range map[string]string{
		"gen_ai.provider.name":       "fake",
		"gen_ai.request.model":       "test-model",
		"gen_ai.operation.name":      "chat",
		"gen_ai.output.type":         "text",
		"stream":                     "true",
		"output":                     "Hi",
		"gen_ai.system_instructions": "be brief",
	} {
		if attrs[key] != want {
			t.Errorf("%s = %q, want %q (attrs: %v)", key, attrs[key], want, attrs)
		}
	}
	if !strings.Contains(attrs["input"], "be brief") || !strings.Contains(attrs["input"], "hi") {
		t.Fatalf("input missing system/user content: %q", attrs["input"])
	}
}

// TestUntracedPipelineEmitsNoSpan checks that a pipeline which was not asked to
// trace raises no service spans, even with a TracerProvider installed for the
// application's own tracing. Without the check they would be exported as orphans,
// with no conversation or turn to hang from.
func TestUntracedPipelineEmitsNoSpan(t *testing.T) {
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
		ReachedDownstreamFilter: pipeline.AnyFrame,
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

	for _, s := range rec.Ended() {
		t.Errorf("untraced pipeline recorded a %q span", s.Name())
	}
}
