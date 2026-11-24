package live

import (
	"context"
	"encoding/json"
	"unicode/utf8"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/telemetry/tracing"
)

// The Live API is a session rather than a request answered by a reply: the model
// generates continuously and reports what it did as it goes. The one span raised
// here covers the session's setup. The model's replies and the function calls
// it makes are traced from the frames the service pushes (see
// llm.WithRealtimeTracing).

// instructionLimit is how much of the system instruction a span carries. An
// instruction can run to many thousands of characters, and a span is a report on
// the session rather than a copy of what was sent to it.
const instructionLimit = 500

// baseAttrs are the attributes every span of the session carries.
func (s *Service) baseAttrs(operation string) tracing.GeminiLiveAttributes {
	return tracing.GeminiLiveAttributes{
		Service:    s.TypeName(),
		Model:      s.model(),
		Operation:  operation,
		Language:   string(s.live.Language.Or("")),
		Modalities: string(s.live.Modalities.Or(ModalityAudio)),
	}
}

// traceSetup records the session's configuration as a span of its own once the
// server has acknowledged it, so a trace shows what the model was told before it
// began answering. The instruction recorded is the effective one: the composed
// instruction, else the conversation's own prompt.
func (s *Service) traceSetup(ctx context.Context) {
	_, span := s.StartSpan(ctx, "llm_setup")
	defer span.End()
	instruction := s.SystemInstruction()
	if instruction == "" {
		s.mu.Lock()
		convo := s.convo
		s.mu.Unlock()
		if convo != nil {
			instruction = convo.System()
		}
	}
	attrs := s.baseAttrs("llm_setup")
	attrs.Settings = map[string]any{
		"voice":                s.live.Voice.Or(""),
		"response_modalities":  string(s.live.Modalities.Or(ModalityAudio)),
		"input_transcription":  true,
		"output_transcription": true,
	}
	attrs.Tools, attrs.ToolsSerialized = setupTools(s.cfg.Tools.Standard)
	attrs.Extra = instructionAttrs(instruction)
	tracing.SetGeminiLiveAttributes(span, attrs)
}

// setupTools describes the tools the service was configured with for the setup
// span: their names, one entry per tool, and their definitions serialized for
// detailed inspection.
func setupTools(tools []frames.Tool) ([]string, string) {
	if len(tools) == 0 {
		return nil, ""
	}
	names := make([]string, len(tools))
	definitions := make([]map[string]any, len(tools))
	for i, tool := range tools {
		names[i] = tool.Name
		// The parameters are a JSON schema object; a tool without one takes no
		// arguments.
		var schema struct {
			Properties map[string]any `json:"properties"`
			Required   []string       `json:"required"`
		}
		_ = json.Unmarshal(tool.Parameters, &schema)
		if schema.Properties == nil {
			schema.Properties = map[string]any{}
		}
		if schema.Required == nil {
			schema.Required = []string{}
		}
		definitions[i] = map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
			"properties":  schema.Properties,
			"required":    schema.Required,
		}
	}
	serialized, err := json.Marshal(definitions)
	if err != nil {
		return names, ""
	}
	return names, string(serialized)
}

// instructionAttrs carries the system instruction the session was set up with,
// truncated to what a span should hold.
func instructionAttrs(instruction string) map[string]any {
	if instruction == "" {
		return nil
	}
	return map[string]any{"gen_ai.system_instructions": truncate(instruction, instructionLimit)}
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
