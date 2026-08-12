package tracing

import (
	"math"
	"reflect"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// providerNames maps a service whose name does not reduce to its provider by
// the usual rule to the provider name the GenAI conventions give it. A service
// absent from the table takes its name with the kind suffix removed.
//
//nolint:gochecknoglobals // a lookup table, not state
var providerNames = map[string]string{
	// AWS
	"BedrockLLM": "aws.bedrock",
	// Azure
	"AzureOpenAILLM": "azure.ai.openai",
	// Google
	"GoogleLLM":       "gcp.gemini",
	"GoogleVertexLLM": "gcp.vertex_ai",
	// Others
	"XAILLM":     "x_ai",
	"MistralLLM": "mistral_ai",
}

// ProviderName is the gen_ai.provider.name value for a service, derived from
// the service's name by dropping the suffix naming its kind ("OpenAILLM"
// becomes "openai") and lowercasing what is left. Services whose provider name
// the conventions spell differently are looked up instead.
//
// The suffix is removed only from the end of the name, not wherever it appears,
// so a provider whose own name ends in the kind it implements keeps it:
// "XTTSTTS" is the XTTS provider's speech synthesis, and reduces to "xtts"
// rather than to "x".
func ProviderName(service, kind string) string {
	if p, ok := providerNames[service]; ok {
		return p
	}
	return strings.ToLower(strings.TrimSuffix(service, kind))
}

// scalarAttr renders v as a span attribute, reporting false for a value that is
// not a scalar. Settings and the caller-supplied extras are plain data that may
// hold a value a span cannot carry (a nil, a struct, a slice); those are left
// off the span rather than rendered as text.
func scalarAttr(key string, v any) (attribute.KeyValue, bool) {
	if v == nil {
		return attribute.KeyValue{}, false
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.String:
		return attribute.String(key, rv.String()), true
	case reflect.Bool:
		return attribute.Bool(key, rv.Bool()), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return attribute.Int64(key, rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if u := rv.Uint(); u <= math.MaxInt64 {
			return attribute.Int64(key, int64(u)), true
		}
		return attribute.KeyValue{}, false
	case reflect.Float32, reflect.Float64:
		return attribute.Float64(key, rv.Float()), true
	default:
		return attribute.KeyValue{}, false
	}
}

// prefixed renders a map of plain values as span attributes under a common
// prefix, dropping the ones a span cannot carry.
func prefixed(prefix string, values map[string]any) []attribute.KeyValue {
	out := make([]attribute.KeyValue, 0, len(values))
	for k, v := range values {
		if kv, ok := scalarAttr(prefix+k, v); ok {
			out = append(out, kv)
		}
	}
	return out
}

// setStaged writes the attribute groups to the span in order, so a later group
// overrides a key an earlier one set. The extras a caller supplies are written
// last and so have the final say, which is what lets a service correct an
// attribute the common set got wrong for it.
func setStaged(span trace.Span, groups ...[]attribute.KeyValue) {
	for _, g := range groups {
		if len(g) > 0 {
			span.SetAttributes(g...)
		}
	}
}

// TTSAttributes describes one speech synthesis for its span.
type TTSAttributes struct {
	// Service is the synthesizing service's name, which the provider name is
	// derived from.
	Service string
	// Model is the model identifier the synthesis ran against.
	Model string
	// VoiceID is the voice spoken in.
	VoiceID string
	// Text is what was synthesized. Empty leaves the attribute off.
	Text string
	// OperationName names the operation; empty means "tts".
	OperationName string
	// CharacterCount is how many characters were synthesized. Nil leaves the
	// attribute off; zero is recorded.
	CharacterCount *int
	// TTFB is the time to first byte in seconds. Nil leaves the attribute off.
	TTFB *float64
	// Settings are the service settings the caller gave, recorded under
	// settings.*.
	Settings map[string]any
	// Extra are additional attributes, written last so they override.
	Extra map[string]any
}

// SetTTSAttributes records a synthesis on its span.
func SetTTSAttributes(span trace.Span, a TTSAttributes) {
	operation := a.OperationName
	if operation == "" {
		operation = "tts"
	}
	std := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", ProviderName(a.Service, "TTS")),
		attribute.String("gen_ai.request.model", a.Model),
		attribute.String("gen_ai.operation.name", operation),
		attribute.String("gen_ai.output.type", "speech"),
		attribute.String("voice_id", a.VoiceID),
	}
	if a.Text != "" {
		std = append(std, attribute.String("text", a.Text))
	}
	if a.CharacterCount != nil {
		std = append(std, attribute.Int("metrics.character_count", *a.CharacterCount))
	}
	if a.TTFB != nil {
		std = append(std, attribute.Float64("metrics.ttfb", *a.TTFB))
	}
	setStaged(span, std, prefixed("settings.", a.Settings), prefixed("", a.Extra))
}

// STTAttributes describes one transcription segment for its span.
type STTAttributes struct {
	// Service is the transcribing service's name, which the provider name is
	// derived from.
	Service string
	// Model is the model identifier the transcription ran against.
	Model string
	// OperationName names the operation; empty means "stt".
	OperationName string
	// Transcript is the text transcribed. Empty leaves the attribute off.
	Transcript string
	// Final reports whether the transcript is the finalized one. Nil leaves the
	// attribute off.
	Final *bool
	// Language is the language transcribed. Empty leaves the attribute off.
	Language string
	// UserID names whose speech was transcribed. Empty leaves the attribute off.
	UserID string
	// VADEnabled reports whether voice activity detection is in use. It is
	// always recorded.
	VADEnabled bool
	// TTFB is the time to first byte in seconds. Nil leaves the attribute off.
	TTFB *float64
	// Settings are the service settings the caller gave, recorded under
	// settings.*.
	Settings map[string]any
	// Extra are additional attributes, written last so they override.
	Extra map[string]any
}

// SetSTTAttributes records a transcription segment on its span.
func SetSTTAttributes(span trace.Span, a STTAttributes) {
	operation := a.OperationName
	if operation == "" {
		operation = "stt"
	}
	std := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", ProviderName(a.Service, "STT")),
		attribute.String("gen_ai.request.model", a.Model),
		attribute.String("gen_ai.operation.name", operation),
		attribute.Bool("vad_enabled", a.VADEnabled),
	}
	if a.Transcript != "" {
		std = append(std, attribute.String("transcript", a.Transcript))
	}
	if a.Final != nil {
		std = append(std, attribute.Bool("is_final", *a.Final))
	}
	if a.Language != "" {
		std = append(std, attribute.String("language", a.Language))
	}
	if a.UserID != "" {
		std = append(std, attribute.String("user_id", a.UserID))
	}
	if a.TTFB != nil {
		std = append(std, attribute.Float64("metrics.ttfb", *a.TTFB))
	}
	setStaged(span, std, prefixed("settings.", a.Settings), prefixed("", a.Extra))
}

// requestParams are the generation parameters the GenAI conventions name, which
// are recorded under gen_ai.request.* rather than as service parameters.
//
//nolint:gochecknoglobals // a lookup table, not state
var requestParams = map[string]bool{
	"temperature":           true,
	"max_tokens":            true,
	"max_completion_tokens": true,
	"top_p":                 true,
	"top_k":                 true,
	"frequency_penalty":     true,
	"presence_penalty":      true,
	"seed":                  true,
}

// LLMAttributes describes one generation for its span.
type LLMAttributes struct {
	// Service is the generating service's name, which the provider name is
	// derived from.
	Service string
	// Model is the model identifier that answered.
	Model string
	// Stream reports whether the generation streamed. It is always recorded.
	Stream bool
	// Messages is the serialized conversation sent to the model. Empty leaves
	// the attribute off.
	Messages string
	// Output is the model's reply text. Empty leaves the attribute off.
	Output string
	// Tools is the serialized tool configuration. Empty leaves the attribute
	// off.
	Tools string
	// ToolCount is how many tools were offered. Nil leaves the attribute off.
	ToolCount *int
	// ToolChoice is how the model was told to choose among them. Empty leaves
	// the attribute off.
	ToolChoice string
	// SystemInstructions is the system prompt in force. Empty leaves the
	// attribute off.
	SystemInstructions string
	// TTFB is the time to first byte in seconds. Nil leaves the attribute off.
	TTFB *float64
	// Parameters are the generation parameters. The ones the conventions name
	// are recorded under gen_ai.request.*, the rest under param.*.
	Parameters map[string]any
	// ExtraParameters are provider-specific parameters, recorded under extra.*.
	ExtraParameters map[string]any
	// Extra are additional attributes, written last so they override.
	Extra map[string]any
}

// SetLLMAttributes records a generation on its span.
func SetLLMAttributes(span trace.Span, a LLMAttributes) {
	std := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", ProviderName(a.Service, "LLM")),
		attribute.String("gen_ai.request.model", a.Model),
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.output.type", "text"),
		attribute.Bool("stream", a.Stream),
	}
	if a.Messages != "" {
		std = append(std, attribute.String("input", a.Messages))
	}
	if a.Output != "" {
		std = append(std, attribute.String("output", a.Output))
	}
	if a.Tools != "" {
		std = append(std, attribute.String("tools", a.Tools))
	}
	if a.ToolCount != nil {
		std = append(std, attribute.Int("tool_count", *a.ToolCount))
	}
	if a.ToolChoice != "" {
		std = append(std, attribute.String("tool_choice", a.ToolChoice))
	}
	if a.SystemInstructions != "" {
		std = append(std, attribute.String("gen_ai.system_instructions", a.SystemInstructions))
	}
	if a.TTFB != nil {
		std = append(std, attribute.Float64("metrics.ttfb", *a.TTFB))
	}

	params := make([]attribute.KeyValue, 0, len(a.Parameters))
	for k, v := range a.Parameters {
		prefix := "param."
		if requestParams[k] {
			prefix = "gen_ai.request."
		}
		if kv, ok := scalarAttr(prefix+k, v); ok {
			params = append(params, kv)
		}
	}
	setStaged(span, std, params, prefixed("extra.", a.ExtraParameters), prefixed("", a.Extra))
}

// nested renders a map of plain values under a prefix, expanding one level of
// nesting so a grouped setting (turn detection, voice activity detection) lands
// as one attribute per field rather than as an unrenderable map.
func nested(prefix string, values map[string]any) []attribute.KeyValue {
	out := make([]attribute.KeyValue, 0, len(values))
	for k, v := range values {
		if kv, ok := scalarAttr(prefix+k, v); ok {
			out = append(out, kv)
			continue
		}
		if sub, ok := v.(map[string]any); ok {
			out = append(out, prefixed(prefix+k+".", sub)...)
		}
	}
	return out
}

// toolNames renders the names of the tools offered, for filtering a trace by the
// tool a turn had available. Tools without a name contribute nothing.
func toolNames(tools []string) (string, bool) {
	named := make([]string, 0, len(tools))
	for _, t := range tools {
		if t != "" {
			named = append(named, t)
		}
	}
	if len(named) == 0 {
		return "", false
	}
	return strings.Join(named, ","), true
}

// GeminiLiveAttributes describes one operation of a Gemini Live session.
type GeminiLiveAttributes struct {
	// Model is the model identifier the session runs against.
	Model string
	// Operation names what the span covers (setup, a response, a tool call).
	Operation string
	// VoiceID is the voice the model speaks in. Empty leaves the attribute off.
	VoiceID string
	// Language is the language of the session. Empty leaves the attribute off.
	Language string
	// Modalities are the response modalities configured. Empty leaves the
	// attribute off.
	Modalities string
	// Tools are the names of the tools offered, one entry per tool. A tool with
	// no name contributes to the count but not to the names.
	Tools []string
	// ToolsSerialized is the full tool definitions. Empty leaves them off.
	ToolsSerialized string
	// Transcript is speech transcribed this operation. Empty leaves it off.
	Transcript string
	// TranscriptIsInput reports whether the transcript is the user's speech
	// rather than the model's. Nil leaves the attribute off.
	TranscriptIsInput *bool
	// TextOutput is text the model produced. Empty leaves the attribute off.
	TextOutput string
	// AudioDataSize is the size of the audio payload in bytes. Nil leaves the
	// attribute off.
	AudioDataSize *int
	// Settings are the session settings, recorded under settings.*. A grouped
	// setting is expanded one level.
	Settings map[string]any
	// Extra are additional attributes, written last so they override.
	Extra map[string]any
}

// SetGeminiLiveAttributes records one Gemini Live operation on its span.
func SetGeminiLiveAttributes(span trace.Span, a GeminiLiveAttributes) {
	std := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", "gcp.gemini"),
		attribute.String("gen_ai.request.model", a.Model),
		attribute.String("gen_ai.operation.name", a.Operation),
		attribute.String("service.operation", a.Operation),
	}
	if a.VoiceID != "" {
		std = append(std, attribute.String("voice_id", a.VoiceID))
	}
	if a.Language != "" {
		std = append(std, attribute.String("language", a.Language))
	}
	if a.Modalities != "" {
		std = append(std, attribute.String("modalities", a.Modalities))
	}
	if a.Transcript != "" {
		std = append(std, attribute.String("transcript", a.Transcript))
		if a.TranscriptIsInput != nil {
			std = append(std, attribute.Bool("transcript.is_input", *a.TranscriptIsInput))
		}
	}
	if a.TextOutput != "" {
		std = append(std, attribute.String("text_output", a.TextOutput))
	}
	if a.AudioDataSize != nil {
		std = append(std, attribute.Int("audio.data_size_bytes", *a.AudioDataSize))
	}
	if len(a.Tools) > 0 {
		std = append(std,
			attribute.Int("tools.count", len(a.Tools)),
			attribute.Bool("tools.available", true))
		if names, ok := toolNames(a.Tools); ok {
			std = append(std, attribute.String("tools.names", names))
		}
	}
	if a.ToolsSerialized != "" {
		std = append(std, attribute.String("tools.definitions", a.ToolsSerialized))
	}
	setStaged(span, std, nested("settings.", a.Settings), prefixed("", a.Extra))
}

// OpenAIRealtimeAttributes describes one operation of an OpenAI Realtime session.
type OpenAIRealtimeAttributes struct {
	// Model is the model identifier the session runs against.
	Model string
	// Operation names what the span covers (setup, a request, a response).
	Operation string
	// SessionProperties are the session configuration, recorded under session.*.
	// A grouped property is expanded one level.
	SessionProperties map[string]any
	// Transcript is speech transcribed this operation. Empty leaves it off.
	Transcript string
	// TranscriptIsInput reports whether the transcript is the user's speech
	// rather than the model's. Nil leaves the attribute off.
	TranscriptIsInput *bool
	// ContextMessages is the serialized conversation sent to the model, recorded
	// as the span's input. Empty leaves the attribute off.
	ContextMessages string
	// Tools are the names of the tools offered, one entry per tool.
	Tools []string
	// ToolsSerialized is the full tool definitions. Empty leaves them off.
	ToolsSerialized string
	// FunctionCalls are the names of the calls the model requested, in the order
	// it requested them.
	FunctionCalls []string
	// AudioDataSize is the size of the audio payload in bytes. Nil leaves the
	// attribute off.
	AudioDataSize *int
	// Extra are additional attributes, written last so they override.
	Extra map[string]any
}

// SetOpenAIRealtimeAttributes records one Realtime operation on its span.
func SetOpenAIRealtimeAttributes(span trace.Span, a OpenAIRealtimeAttributes) {
	std := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", "openai"),
		attribute.String("gen_ai.request.model", a.Model),
		attribute.String("gen_ai.operation.name", a.Operation),
		attribute.String("service.operation", a.Operation),
	}
	if a.Transcript != "" {
		std = append(std, attribute.String("transcript", a.Transcript))
		if a.TranscriptIsInput != nil {
			std = append(std, attribute.Bool("transcript.is_input", *a.TranscriptIsInput))
		}
	}
	if a.ContextMessages != "" {
		std = append(std, attribute.String("input", a.ContextMessages))
	}
	if a.AudioDataSize != nil {
		std = append(std, attribute.Int("audio.data_size_bytes", *a.AudioDataSize))
	}
	if len(a.Tools) > 0 {
		std = append(std,
			attribute.Int("tools.count", len(a.Tools)),
			attribute.Bool("tools.available", true))
		if names, ok := toolNames(a.Tools); ok {
			std = append(std, attribute.String("tools.names", names))
		}
	}
	if a.ToolsSerialized != "" {
		std = append(std, attribute.String("tools.definitions", a.ToolsSerialized))
	}
	if len(a.FunctionCalls) > 0 {
		std = append(std, attribute.Int("function_calls.count", len(a.FunctionCalls)))
		if first := a.FunctionCalls[0]; first != "" {
			std = append(std, attribute.String("function_calls.first_name", first))
		}
		if names, ok := toolNames(a.FunctionCalls); ok {
			std = append(std, attribute.String("function_calls.all_names", names))
		}
	}
	setStaged(span, std, nested("session.", a.SessionProperties), prefixed("", a.Extra))
}
