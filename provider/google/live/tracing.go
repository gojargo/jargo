package live

import (
	"context"
	"unicode/utf8"

	"github.com/gojargo/jargo/telemetry/tracing"
)

// The Live API is a session rather than a request answered by a reply: the model
// generates continuously and reports what it did as it goes. The spans here
// follow that shape, covering the operations the session performs rather than
// wrapping calls the service makes.
//
// The protocol this service speaks carries no tool calling, so there is no tool
// call and no tool result to record. When tool calling is added, its operations
// belong here beside these.

// instructionLimit is how much of the system instruction a span carries. An
// instruction can run to many thousands of characters, and a span is a report on
// the session rather than a copy of what was sent to it.
const instructionLimit = 500

// traceSetup records the session's configuration as a span of its own, so a
// trace shows what the model was told before it began answering. The span covers
// the configuration rather than the connection, and closes once the setup has
// been sent.
func (s *Service) traceSetup(ctx context.Context) {
	_, span := s.StartSpan(ctx, "llm_setup")
	defer span.End()
	tracing.SetGeminiLiveAttributes(span, tracing.GeminiLiveAttributes{
		Model:      s.cfg.Model,
		Operation:  "llm_setup",
		VoiceID:    s.cfg.Voice,
		Modalities: modalityAudio,
		Settings: map[string]any{
			"voice":                s.cfg.Voice,
			"response_modalities":  modalityAudio,
			"input_transcription":  true,
			"output_transcription": true,
		},
		Extra: instructionAttrs(s.cfg.Instructions),
	})
}

// instructionAttrs carries the system instruction the session was set up with,
// truncated to what a span should hold.
func instructionAttrs(instruction string) map[string]any {
	if instruction == "" {
		return nil
	}
	return map[string]any{"gen_ai.system_instructions": truncate(instruction, instructionLimit)}
}

// traceResponse records one completed model turn: what it produced and what it
// cost. It returns a context carrying the span so the usage reported for the
// turn lands on the operation that incurred it, and the function that ends it.
//
// The turn's accounting is what a cost dashboard prices, and it arrives on the
// same message that reports the turn complete, so the two are recorded together.
func (s *Service) traceResponse(ctx context.Context, msg serverMessage) (context.Context, func()) {
	spanCtx, span := s.StartSpan(ctx, "llm_response")
	attrs := tracing.GeminiLiveAttributes{
		Model:      s.cfg.Model,
		Operation:  "llm_response",
		VoiceID:    s.cfg.Voice,
		Modalities: modalityAudio,
	}
	if sc := msg.ServerContent; sc != nil {
		if sc.OutputTranscription != nil {
			// What the model said, as the model itself transcribed it. A
			// native-audio turn has no text output otherwise.
			attrs.TextOutput = sc.OutputTranscription.Text
		}
		if sc.GenerationComplete {
			attrs.Extra = map[string]any{"turn_complete": true}
		}
	}
	tracing.SetGeminiLiveAttributes(span, attrs)
	return spanCtx, func() { span.End() }
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
