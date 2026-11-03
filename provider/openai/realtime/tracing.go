package realtime

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/gojargo/jargo/provider/openai/realtime/events"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
)

// The Realtime API is a session rather than a request answered by a reply: the
// model holds the conversation itself and generates continuously. The spans here
// follow that shape, covering the operations the session performs: its setup,
// each response asked for, each response completed, and the user's speech as
// the session transcribed it.

// traceSetup records the session being configured, once the server reports it
// created. What it records is the conversation's own toolset, when there is a
// conversation advertising one; the session's properties are not recorded.
func (s *Service) traceSetup(ctx context.Context) {
	_, span := s.StartSpan(ctx, "llm_setup")
	defer span.End()
	attrs := tracing.OpenAIRealtimeAttributes{Model: s.model(), Operation: "llm_setup"}
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo != nil {
		if schema := convo.ToolsSchema(); !schema.Empty() {
			tools := s.adapter.ToProviderToolsFormat(schema)
			attrs.Tools = toolNamesOf(tools)
			if data, err := json.Marshal(tools); err == nil {
				attrs.ToolsSerialized = string(data)
			}
		}
	}
	tracing.SetOpenAIRealtimeAttributes(span, attrs)
}

// traceRequest records a response being asked for, with the conversation as it
// stands, and returns a context carrying the span and the function that ends
// it.
func (s *Service) traceRequest(ctx context.Context) (context.Context, func()) {
	spanCtx, span := s.StartSpan(ctx, "llm_request")
	attrs := tracing.OpenAIRealtimeAttributes{Model: s.model(), Operation: "llm_request"}
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo != nil {
		if msgs := s.adapter.MessagesForLogging(convo); len(msgs) > 0 {
			if data, err := json.Marshal(msgs); err == nil {
				attrs.ContextMessages = string(data)
			}
		}
	}
	tracing.SetOpenAIRealtimeAttributes(span, attrs)
	return spanCtx, func() { span.End() }
}

// toolNamesOf reads the name out of each tool the session advertises, one entry
// per tool so the count stays right even for a tool that carries no name.
func toolNamesOf(tools []map[string]any) []string {
	names := make([]string, len(tools))
	for i, t := range tools {
		if name, ok := t["name"].(string); ok {
			names[i] = name
		}
	}
	return names
}

// traceResponse records one completed model turn: how it ended, what it said,
// and what it asked to have called. It returns a context carrying the span so
// the turn's usage lands on the operation that incurred it, and the function
// that ends it.
func (s *Service) traceResponse(ctx context.Context, r *events.Response) (context.Context, func()) {
	spanCtx, span := s.StartSpan(ctx, "llm_response")
	attrs := tracing.OpenAIRealtimeAttributes{
		Model:     s.model(),
		Operation: "llm_response",
	}
	if r != nil {
		attrs.FunctionCalls = functionCallNames(r.Output)
		attrs.Extra = responseAttrs(r)
	}
	tracing.SetOpenAIRealtimeAttributes(span, attrs)
	// The time to first byte of the response being reported, read as it is
	// reported.
	if ttfb, ok := s.CurrentTTFB(); ok {
		span.SetAttributes(attribute.Float64("metrics.ttfb", ttfb.Seconds()))
	}
	return spanCtx, func() { span.End() }
}

// responseAttrs reports how the turn ended and what it produced. A field the
// response does not carry is left off rather than recorded as empty.
func responseAttrs(r *events.Response) map[string]any {
	attrs := map[string]any{}
	if r.Status != "" {
		attrs["response.status"] = r.Status
	}
	if r.ID != "" {
		attrs["response.id"] = r.ID
	}
	if len(r.Output) > 0 {
		attrs["response.output_items"] = len(r.Output)
	}
	if spoken := assistantTranscript(r.Output); spoken != "" {
		attrs["output"] = spoken
	}
	if len(attrs) == 0 {
		return nil
	}
	return attrs
}

// assistantTranscript is everything the model said this turn, as the model
// itself transcribed it. A native-audio turn has no text output otherwise.
func assistantTranscript(output []events.ConversationItem) string {
	var parts []string
	for _, item := range output {
		if item.Role != roleAssistant {
			continue
		}
		for _, c := range item.Content {
			if c.Transcript != "" {
				parts = append(parts, c.Transcript)
			}
		}
	}
	return strings.Join(parts, " ")
}

// functionCallNames is the name of each function the model asked to have called,
// in the order it asked, one entry per call.
func functionCallNames(output []events.ConversationItem) []string {
	var names []string
	for _, item := range output {
		if item.Type == "function_call" {
			names = append(names, item.Name)
		}
	}
	return names
}
