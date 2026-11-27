package realtime

import (
	"context"
	"encoding/json"

	"github.com/gojargo/jargo/telemetry/tracing"
)

// The Realtime API is a session rather than a request answered by a reply: the
// model holds the conversation itself and generates continuously. The one span
// raised here covers the session's setup. The model's replies and the function
// calls it makes are traced from the frames the service pushes (see
// llm.WithRealtimeTracing).

// traceSetup records the session being configured, once the server reports it
// created. What it records is the conversation's own toolset, when there is a
// conversation advertising one; the session's properties are not recorded.
func (s *LLMService) traceSetup(ctx context.Context) {
	_, span := s.StartSpan(ctx, "llm_setup")
	defer span.End()
	attrs := tracing.OpenAIRealtimeAttributes{Service: s.TypeName(), Model: s.model(), Operation: "llm_setup"}
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
