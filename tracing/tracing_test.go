package tracing_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestStartConversationRecordsID(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	_, span := tracing.StartConversation(context.Background(), "sess-1")
	span.End()

	spans := rec.Ended()
	if len(spans) != 1 || spans[0].Name() != "conversation" {
		t.Fatalf("spans = %+v, want one 'conversation' span", spans)
	}
	var id string
	for _, kv := range spans[0].Attributes() {
		if string(kv.Key) == "conversation.id" {
			id = kv.Value.AsString()
		}
	}
	if id != "sess-1" {
		t.Fatalf("conversation.id = %q, want sess-1", id)
	}
}
