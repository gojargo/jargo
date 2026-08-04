package tracing_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// TestParentPrefersTurn checks that a service span hangs from the turn being
// spoken, falls back to the conversation between turns, and leaves the caller's
// context alone when nothing is being traced.
func TestParentPrefersTurn(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	tracer := otel.Tracer("test")
	_, conversation := tracer.Start(context.Background(), "conversation")
	_, turn := tracer.Start(
		trace.ContextWithSpanContext(context.Background(), conversation.SpanContext()), "turn")

	tc := tracing.NewTracingContext()
	if got := tc.Parent(context.Background()); trace.SpanContextFromContext(got).IsValid() {
		t.Fatal("Parent on an empty context should carry no span")
	}

	tc.SetConversationContext(conversation.SpanContext(), "sess-1")
	if got := trace.SpanContextFromContext(tc.Parent(context.Background())); !got.Equal(conversation.SpanContext()) {
		t.Fatalf("between turns, parent = %v, want the conversation span", got)
	}
	if id := tc.ConversationID(); id != "sess-1" {
		t.Fatalf("ConversationID() = %q, want sess-1", id)
	}

	tc.SetTurnContext(turn.SpanContext())
	if got := trace.SpanContextFromContext(tc.Parent(context.Background())); !got.Equal(turn.SpanContext()) {
		t.Fatalf("during a turn, parent = %v, want the turn span", got)
	}

	// A closed turn hands the conversation back.
	tc.SetTurnContext(trace.SpanContext{})
	if got := trace.SpanContextFromContext(tc.Parent(context.Background())); !got.Equal(conversation.SpanContext()) {
		t.Fatalf("after the turn, parent = %v, want the conversation span", got)
	}

	turn.End()
	conversation.End()
}

// TestParentKeepsContextValues checks that re-parenting a span does not cost the
// caller the rest of its context: a service goes on using it for the work the
// span covers.
func TestParentKeepsContextValues(t *testing.T) {
	tc := tracing.NewTracingContext()
	tc.SetConversationContext(trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1},
		SpanID:  trace.SpanID{2},
	}), "sess-1")

	ctx, cancel := context.WithCancel(context.Background())
	parent := tc.Parent(ctx)
	cancel()
	if parent.Err() == nil {
		t.Fatal("the parented context should be canceled with the one it came from")
	}
}

// TestNilTracingContext checks that a pipeline running without tracing can call
// through without guarding every use.
func TestNilTracingContext(t *testing.T) {
	var tc *tracing.TracingContext
	ctx := context.Background()
	tc.SetConversationContext(trace.SpanContext{}, "sess-1")
	tc.SetTurnContext(trace.SpanContext{})
	if tc.Parent(ctx) != ctx {
		t.Fatal("a nil tracing context should leave the context alone")
	}
	if tc.ConversationID() != "" {
		t.Fatal("a nil tracing context should report no conversation")
	}
}

func TestGenerateConversationID(t *testing.T) {
	if a, b := tracing.GenerateConversationID(), tracing.GenerateConversationID(); a == b || a == "" {
		t.Fatalf("generated ids = %q, %q, want two distinct non-empty ids", a, b)
	}
}
