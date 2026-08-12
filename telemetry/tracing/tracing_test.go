package tracing_test

import (
	"context"
	"sync"
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

// TestTracingContextsAreIsolated checks that two pipelines running at once do
// not see each other's spans. Each task creates its own context, and that is
// what keeps concurrent conversations in separate traces.
func TestTracingContextsAreIsolated(t *testing.T) {
	a, b := tracing.NewTracingContext(), tracing.NewTracingContext()
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2},
	})

	a.SetConversationContext(sc, "conv-a")
	a.SetTurnContext(sc)

	if b.ConversationContext().IsValid() || b.TurnContext().IsValid() {
		t.Fatal("one pipeline's spans should not be visible to another")
	}
	if b.ConversationID() != "" {
		t.Fatalf("ConversationID() = %q, want empty", b.ConversationID())
	}
}

// TestTracingContextClearsTurnOnly checks that closing a turn leaves the
// conversation open, so the turn after it still has something to hang from.
func TestTracingContextClearsTurnOnly(t *testing.T) {
	tc := tracing.NewTracingContext()
	conversation := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2},
	})
	turn := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{3},
	})
	tc.SetConversationContext(conversation, "conv-1")
	tc.SetTurnContext(turn)

	tc.SetTurnContext(trace.SpanContext{})
	if tc.TurnContext().IsValid() {
		t.Fatal("the turn should be cleared")
	}
	if !tc.ConversationContext().Equal(conversation) || tc.ConversationID() != "conv-1" {
		t.Fatal("clearing the turn should leave the conversation open")
	}
}

// TestTracingContextConcurrentAccess exercises the lock the tracing context
// carries. Upstream runs single-threaded and needs none; here the observer
// writes the turn from the frame path while services read it from goroutines of
// their own, so the race detector has something to check.
func TestTracingContextConcurrentAccess(t *testing.T) {
	tc := tracing.NewTracingContext()
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2},
	})

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				tc.SetTurnContext(sc)
				tc.SetTurnContext(trace.SpanContext{})
			}
		})
		wg.Go(func() {
			for range 100 {
				tc.Parent(context.Background())
				tc.ConversationID()
			}
		})
	}
	wg.Wait()
}

func TestGenerateConversationID(t *testing.T) {
	if a, b := tracing.GenerateConversationID(), tracing.GenerateConversationID(); a == b || a == "" {
		t.Fatalf("generated ids = %q, %q, want two distinct non-empty ids", a, b)
	}
}
