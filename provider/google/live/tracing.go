package live

import (
	"context"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/gojargo/jargo/telemetry/tracing"
)

// The Live API is a session rather than a request answered by a reply: the model
// generates continuously and reports what it did as it goes. The spans here
// follow that shape, covering the operations the session performs: its setup,
// each completed turn, and each tool call and tool result. The user's speech as
// the model transcribed it is traced from the transcripts the service pushes.

// instructionLimit is how much of the system instruction a span carries. An
// instruction can run to many thousands of characters, and a span is a report on
// the session rather than a copy of what was sent to it.
const instructionLimit = 500

// The limits on a tool call's arguments and a tool result a span carries.
const (
	toolArgumentsLimit = 1000
	toolResultLimit    = 2000
)

// baseAttrs are the attributes every span of the session carries.
func (s *Service) baseAttrs(operation string) tracing.GeminiLiveAttributes {
	return tracing.GeminiLiveAttributes{
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
	attrs.Extra = instructionAttrs(instruction)
	tracing.SetGeminiLiveAttributes(span, attrs)
}

// instructionAttrs carries the system instruction the session was set up with,
// truncated to what a span should hold.
func instructionAttrs(instruction string) map[string]any {
	if instruction == "" {
		return nil
	}
	return map[string]any{"gen_ai.system_instructions": truncate(instruction, instructionLimit)}
}

// traceResponse records one completed turn: what the model produced over it and
// what it cost. text is the model's text in the text modality, and spoken the
// transcript of what it said in the audio one. The usage the message carries
// is recorded on the span. It returns the function that ends the span.
func (s *Service) traceResponse(ctx context.Context, msg *serverMessage, text, spoken string) func() {
	spanCtx, span := s.StartSpan(ctx, "llm_response")
	attrs := s.baseAttrs("llm_response")
	extra := map[string]any{}
	switch {
	case text != "":
		extra["output"] = text
		extra["output_modality"] = string(ModalityText)
	case spoken != "":
		extra["output"] = spoken
		extra["output_modality"] = string(ModalityAudio)
	}
	if msg != nil && msg.ServerContent != nil && msg.ServerContent.TurnComplete {
		extra["turn_complete"] = true
	}
	attrs.Extra = extra
	tracing.SetGeminiLiveAttributes(span, attrs)
	if msg != nil && msg.UsageMetadata != nil {
		tracing.SetTokenUsage(spanCtx, msg.UsageMetadata.tokenUsage())
	}
	return func() { span.End() }
}

// traceToolCall records the functions the model asked to call in one message:
// the first call in full, with its arguments, and the names of all of them.
func (s *Service) traceToolCall(ctx context.Context, calls []functionCall) {
	if len(calls) == 0 {
		return
	}
	_, span := s.StartSpan(ctx, "llm_tool_call")
	defer span.End()
	names := make([]string, 0, len(calls))
	for _, c := range calls {
		names = append(names, c.Name)
	}
	first := calls[0]
	arguments := "{}"
	if len(first.Args) > 0 {
		arguments = string(first.Args)
	}
	if len(arguments) > toolArgumentsLimit {
		arguments = arguments[:toolArgumentsLimit] + "..."
	}
	attrs := s.baseAttrs("llm_tool_call")
	attrs.Extra = map[string]any{
		"tool.function_name":      first.Name,
		"tool.call_id":            first.ID,
		"tool.calls_count":        len(calls),
		"tool.all_function_names": strings.Join(names, ","),
		"tool.arguments":          arguments,
	}
	tracing.SetGeminiLiveAttributes(span, attrs)
}

// traceToolResult records a tool result sent to the model: the call it answers,
// and the result itself when it is an object, with whether it reports an error
// or a success.
func (s *Service) traceToolResult(ctx context.Context, toolCallID, name string, result any) {
	_, span := s.StartSpan(ctx, "llm_tool_result")
	defer span.End()
	attrs := s.baseAttrs("llm_tool_result")
	attrs.Extra = toolResultAttrs(toolCallID, name, result)
	tracing.SetGeminiLiveAttributes(span, attrs)
}

// toolResultAttrs are the attributes of a tool result's span.
func toolResultAttrs(toolCallID, name string, result any) map[string]any {
	extra := map[string]any{}
	if toolCallID != "" {
		extra["tool.call_id"] = toolCallID
	}
	if name != "" {
		extra["tool.function_name"] = name
	}
	payload, ok := result.(map[string]any)
	if !ok {
		return extra
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return extra
	}
	encoded := string(raw)
	if len(encoded) > toolResultLimit {
		encoded = encoded[:toolResultLimit] + "..."
	}
	extra["tool.result"] = encoded
	_, isError := payload["error"]
	_, isSuccess := payload["success"]
	switch {
	case isError:
		extra["tool.result_status"] = "error"
	case isSuccess:
		extra["tool.result_status"] = "success"
	default:
		extra["tool.result_status"] = "completed"
	}
	return extra
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
