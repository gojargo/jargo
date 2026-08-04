package pipeline_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// spanning is a processor that opens a span the way a service does: parented to
// the tracing context it was set up with, rather than to whatever the frame's
// context points at.
type spanning struct {
	*processor.Base
	traced chan struct{}
}

func newSpanning() *spanning {
	s := &spanning{traced: make(chan struct{}, 1)}
	s.Base = processor.New("Spanning", s)
	return s
}

func (s *spanning) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		_, span := tracing.Tracer().Start(s.Tracing().Parent(ctx), "service")
		span.End()
		select {
		case s.traced <- struct{}{}:
		default:
		}
	}
	return s.PushFrame(ctx, f, dir)
}

// TestTaskTracesOneConversation checks that a traced task is one trace: the
// conversation span is the root, it carries the caller's own attributes, and a
// service span raised inside the pipeline lands in the same trace.
func TestTaskTracesOneConversation(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	svc := newSpanning()
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableTracing:  true,
		ConversationID: "sess-1",
		AdditionalSpanAttributes: []attribute.KeyValue{
			attribute.String("langfuse.session.id", "day-1"),
			attribute.String("langfuse.user.id", "user-1"),
		},
	})
	if task.Tracing() == nil {
		t.Fatal("a traced task should carry a tracing context")
	}

	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()
	if err := task.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	select {
	case <-svc.traced:
	case <-time.After(time.Second):
		t.Fatal("the processor never opened its span")
	}

	var conversation, service sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		switch s.Name() {
		case "conversation":
			conversation = s
		case "service":
			service = s
		}
	}
	if conversation == nil || service == nil {
		t.Fatalf("want a conversation and a service span, got %d spans", len(rec.Ended()))
	}
	if conversation.Parent().IsValid() {
		t.Fatal("the conversation span should be the root of the trace")
	}
	if service.SpanContext().TraceID() != conversation.SpanContext().TraceID() {
		t.Fatal("the service span should be in the conversation's trace")
	}
	want := map[string]string{
		"conversation.id":     "sess-1",
		"langfuse.session.id": "day-1",
		"langfuse.user.id":    "user-1",
	}
	for _, kv := range conversation.Attributes() {
		if w, ok := want[string(kv.Key)]; ok {
			if kv.Value.AsString() != w {
				t.Fatalf("conversation %s = %q, want %q", kv.Key, kv.Value.AsString(), w)
			}
			delete(want, string(kv.Key))
		}
	}
	if len(want) > 0 {
		t.Fatalf("conversation span is missing %v", want)
	}
}

// TestTaskWithoutTracing checks that an untraced task raises no spans and hands
// its processors no tracing context, which their spans handle by staying where
// the frame's context put them.
func TestTaskWithoutTracing(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	svc := newSpanning()
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{})
	if task.Tracing() != nil {
		t.Fatal("an untraced task should carry no tracing context")
	}

	task.QueueFrame(frames.NewTextFrame("hello"))
	task.StopWhenDone()
	if err := task.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	for _, s := range rec.Ended() {
		if s.Name() == "conversation" || s.Name() == "turn" {
			t.Fatalf("untraced task raised a %q span", s.Name())
		}
	}
}
