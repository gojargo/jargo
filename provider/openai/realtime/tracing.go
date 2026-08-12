package realtime

import (
	"context"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/gojargo/jargo/telemetry/tracing"
)

// The Realtime API is a session rather than a request answered by a reply: the
// model holds the conversation itself and generates continuously. The spans here
// follow that shape, covering the operations the session performs.
//
// The model keeps the conversation server-side, so this service never sends it a
// message list and there is nothing for a request span to carry. What it does
// send is configuration, at the start and again whenever the toolset changes,
// and that is what the setup span records.

// instructionLimit is how much of the session instructions a span carries. They
// can run to many thousands of characters, and a span is a report on the session
// rather than a copy of what was sent to it.
const instructionLimit = 500

// traceSetup records a session configuration message: the properties the session
// was given and the tools it advertises. It covers the configuration rather than
// the connection, and closes once the message has been built.
func (s *Service) traceSetup(ctx context.Context, session map[string]any) {
	_, span := s.StartSpan(ctx, "llm_setup")
	defer span.End()

	properties := make(map[string]any, len(session))
	for k, v := range session {
		if k == "instructions" {
			if text, ok := v.(string); ok {
				properties[k] = truncate(text, instructionLimit)
			}
			continue
		}
		properties[k] = v
	}

	attrs := tracing.OpenAIRealtimeAttributes{
		Model:             s.cfg.Model,
		Operation:         "llm_setup",
		SessionProperties: properties,
	}
	if tools, ok := session["tools"].([]map[string]any); ok && len(tools) > 0 {
		attrs.Tools = toolNamesOf(tools)
		if data, err := json.Marshal(tools); err == nil {
			attrs.ToolsSerialized = string(data)
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

// traceResponse records one completed model turn: how it ended, what it said,
// and what it asked to have called. It returns a context carrying the span so
// the turn's usage lands on the operation that incurred it, and the function
// that ends it.
func (s *Service) traceResponse(ctx context.Context, r *responseObject) (context.Context, func()) {
	spanCtx, span := s.StartSpan(ctx, "llm_response")
	attrs := tracing.OpenAIRealtimeAttributes{
		Model:     s.cfg.Model,
		Operation: "llm_response",
	}
	if r != nil {
		attrs.FunctionCalls = functionCallNames(r.Output)
		attrs.Extra = responseAttrs(r)
	}
	tracing.SetOpenAIRealtimeAttributes(span, attrs)
	return spanCtx, func() { span.End() }
}

// responseAttrs reports how the turn ended and what it produced. A field the
// response does not carry is left off rather than recorded as empty.
func responseAttrs(r *responseObject) map[string]any {
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
func assistantTranscript(output []responseOutputItem) string {
	var parts []string
	for _, item := range output {
		if item.Role != "assistant" {
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
func functionCallNames(output []responseOutputItem) []string {
	var names []string
	for _, item := range output {
		if item.Type == "function_call" {
			names = append(names, item.Name)
		}
	}
	return names
}

// truncate shortens s to at most n bytes, marking that it was cut. The cut is
// made on a rune boundary so the result stays valid UTF-8.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "..."
}
