package llm

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Realtime (speech-to-speech) tracing.
//
// A realtime model takes audio in and sends audio out; there is no separate
// speech-to-text stage to measure. So instead of stt spans and a span per
// message handler, a realtime service gets:
//
//   - one llm_response span per reply, from the end of the user's speech to
//     the end of the reply (or the interruption), nested inside its turn span.
//     The user's transcript and the reply text are attributes on it, along
//     with token usage, the time to first byte and the modalities.
//   - one llm_tool_call span per function call, beside the reply.
//
// A reply opens on an LLMFullResponseStartFrame, or on the first
// FunctionCallInProgressFrame when no start came first. It ends on an
// LLMFullResponseEndFrame or an InterruptionFrame and is exported once its late
// data (usage, the user's transcript) is in: at once for a reply that answers
// no speech, otherwise when the next reply opens or after a short grace period.
//
// The spans are driven by the frames the service itself pushes (read after each
// push), by the usage it reports through StartLLMUsageMetrics and by its
// cleanup, so a service needs no tracing code of its own. Only downstream
// frames drive the spans, plus the TranscriptionFrame realtime services push
// upstream.

const (
	// realtimeToolAttrMaxChars is how much of a tool call's arguments or result
	// a span carries.
	realtimeToolAttrMaxChars = 2000
	// realtimeReplyExportGrace is how long a reply that answers speech waits
	// for its late data before it is exported.
	realtimeReplyExportGrace = 2 * time.Second
)

// WithRealtimeTracing traces a realtime (speech-to-speech) service: one
// llm_response span per reply and one llm_tool_call span per function call,
// both followed from the frames the service pushes. A realtime service passes
// it when it builds its base and needs no other tracing code for its replies.
//
// It does nothing on a pipeline that is not traced, which is checked as each
// frame is pushed.
func WithRealtimeTracing() Option {
	return func(b *Base) {
		if b.realtime == nil {
			b.realtime = newRealtimeTracer(b)
		}
	}
}

// textPiece is one piece of reply text, and whether it carries the spaces that
// separate it from its neighbors.
type textPiece struct {
	text           string
	includesSpaces bool
}

// sumTokenUsage adds a usage report to a running total, field by field.
//
// Services that report usage in parts (one report per partial usage event, or
// one per model call on a held thinking turn) are summed into one figure for
// the reply. A field that neither report sets stays nil, so "not reported" is
// kept apart from zero.
func sumTokenUsage(total *frames.LLMTokenUsage, usage frames.LLMTokenUsage) *frames.LLMTokenUsage {
	if total == nil {
		sum := usage
		return &sum
	}
	add := func(a, b *int64) *int64 {
		if a == nil {
			return b
		}
		if b == nil {
			return a
		}
		return new(*a + *b)
	}
	return &frames.LLMTokenUsage{
		PromptTokens:         total.PromptTokens + usage.PromptTokens,
		CompletionTokens:     total.CompletionTokens + usage.CompletionTokens,
		TotalTokens:          total.TotalTokens + usage.TotalTokens,
		CacheReadTokens:      add(total.CacheReadTokens, usage.CacheReadTokens),
		CacheCreationTokens:  add(total.CacheCreationTokens, usage.CacheCreationTokens),
		ReasoningTokens:      add(total.ReasoningTokens, usage.ReasoningTokens),
		InputAudioTokens:     add(total.InputAudioTokens, usage.InputAudioTokens),
		OutputAudioTokens:    add(total.OutputAudioTokens, usage.OutputAudioTokens),
		CacheReadAudioTokens: add(total.CacheReadAudioTokens, usage.CacheReadAudioTokens),
		InputTextTokens:      add(total.InputTextTokens, usage.InputTextTokens),
		OutputTextTokens:     add(total.OutputTextTokens, usage.OutputTextTokens),
	}
}

// joinTextPieces joins text pieces, adding a space between pieces that don't
// carry their own.
func joinTextPieces(pieces []textPiece) string {
	var out strings.Builder
	for _, p := range pieces {
		if p.text == "" {
			continue
		}
		if !p.includesSpaces && out.Len() > 0 {
			out.WriteString(" ")
		}
		out.WriteString(p.text)
	}
	return strings.TrimSpace(out.String())
}

// truncateToolAttr cuts text to the size a tool attribute may have, marking
// that it was cut.
func truncateToolAttr(text string) string {
	runes := []rune(text)
	if len(runes) > realtimeToolAttrMaxChars {
		return string(runes[:realtimeToolAttrMaxChars]) + "..."
	}
	return text
}

// toolArgumentsAttr renders a call's arguments as JSON for a span attribute,
// truncated to a sane size. A call made with no arguments is an empty object.
func toolArgumentsAttr(args json.RawMessage) string {
	if len(args) == 0 {
		return "{}"
	}
	return truncateToolAttr(string(args))
}

// toolResultAttr renders a call's result as JSON for a span attribute,
// truncated to a sane size. A result that is already a JSON object or array is
// recorded as it is; any other is recorded as a JSON string.
func toolResultAttr(result string) string {
	trimmed := strings.TrimSpace(result)
	structured := strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
	if structured && json.Valid([]byte(trimmed)) {
		return truncateToolAttr(result)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return truncateToolAttr(result)
	}
	return truncateToolAttr(string(encoded))
}

// realtimeReply is one model reply of a realtime service, from the end of the
// user's speech to the end of the reply.
//
// The span is opened when the reply starts and exported later, once the data
// that arrives after the reply ends (usage, the user's transcript) is in, or
// the grace period runs out.
type realtimeReply struct {
	span  trace.Span
	start time.Time
	// end is when the reply ended, zero while it is open.
	end           time.Time
	answersSpeech bool
	seq           int
	interrupted   bool
	inputPieces   []string
	llmText       []textPiece
	ttsText       []textPiece
	hadAudio      bool
	// ttfb is the time to first byte in seconds, nil when none was reported.
	ttfb       *float64
	usage      *frames.LLMTokenUsage
	graceTimer *time.Timer
	exported   bool
}

// complete reports whether the reply can be exported before the grace period
// runs out.
//
// Usage is reported once, so a reply that answers no speech is done when it
// has usage. A reply that answers speech is never complete early: the user's
// transcript comes in pieces (one frame per sentence on some services, plus a
// trailing fragment on a timeout), and nothing marks the last one. It waits
// for the grace period or the next reply.
func (r *realtimeReply) complete() bool {
	return r.usage != nil && !r.answersSpeech
}

// realtimeTracer is the realtime tracing state of one service.
//
// Frames reach a realtime service's PushFrame from several of its goroutines,
// and the grace timer runs on one of its own, so everything here changes with
// mu held.
type realtimeTracer struct {
	svc *Base
	// grace is how long a reply that answers speech waits for its late data.
	grace time.Duration

	mu      sync.Mutex
	reply   *realtimeReply
	pending *realtimeReply
	seq     int
	// speechEnd is the speech end since the previous reply opened, zero when
	// there is none, and speechEndTurn the turn that was current when it ended.
	speechEnd     time.Time
	speechEndTurn trace.SpanContext
	// speechOnset is the earliest speech onset since the last speech end, zero
	// when there is none.
	speechOnset time.Time
	// userStartedSinceReply reports whether the user has spoken since the last
	// reply that answered them opened. Speech not yet answered keeps its
	// transcripts out of the replies around it.
	userStartedSinceReply bool
	// Data that arrived while no reply was open, kept for the next one.
	heldTTFB       *float64
	heldText       []textPiece
	heldTTSText    []textPiece
	heldTranscript []string
	toolSpans      map[string]trace.Span
}

// newRealtimeTracer builds the realtime tracing state of svc.
func newRealtimeTracer(svc *Base) *realtimeTracer {
	return &realtimeTracer{
		svc:                   svc,
		grace:                 realtimeReplyExportGrace,
		userStartedSinceReply: true,
		toolSpans:             map[string]trace.Span{},
	}
}

// turnContext is the context a span is parented from: the turn being spoken,
// else the conversation.
func (t *realtimeTracer) turnContext() context.Context {
	return t.svc.Tracing().Parent(context.Background())
}

// discardHeld drops the reply data held for the next reply.
func (t *realtimeTracer) discardHeld() {
	t.heldTTFB = nil
	t.heldText = nil
	t.heldTTSText = nil
}

// exportReply sets the reply's attributes and ends its span at the recorded
// end time.
func (t *realtimeTracer) exportReply(reply *realtimeReply) {
	if reply.exported {
		return
	}
	reply.exported = true
	if reply.graceTimer != nil {
		reply.graceTimer.Stop()
		reply.graceTimer = nil
	}
	if t.pending == reply {
		t.pending = nil
	}
	if t.reply == reply {
		t.reply = nil
	}
	span := reply.span
	outputType, outputModality := "text", "TEXT"
	if reply.hadAudio {
		outputType, outputModality = "speech", "AUDIO"
	}
	span.SetAttributes(attribute.String("gen_ai.output.type", outputType))
	pieces := reply.llmText
	if len(pieces) == 0 {
		pieces = reply.ttsText
	}
	if output := joinTextPieces(pieces); output != "" {
		span.SetAttributes(attribute.String("output", output))
	}
	span.SetAttributes(attribute.String("output_modality", outputModality))
	if len(reply.inputPieces) > 0 {
		input := make([]string, len(reply.inputPieces))
		for i, p := range reply.inputPieces {
			input[i] = strings.TrimSpace(p)
		}
		span.SetAttributes(
			attribute.String("input", strings.Join(input, " ")),
			attribute.String("input_modality", "AUDIO"),
		)
	}
	if reply.ttfb != nil {
		span.SetAttributes(
			attribute.Float64("metrics.ttfb", *reply.ttfb),
			attribute.Float64("gen_ai.response.time_to_first_chunk", *reply.ttfb),
		)
	}
	if reply.usage != nil {
		tracing.SetTokenUsage(trace.ContextWithSpan(context.Background(), span), *reply.usage)
	}
	span.SetAttributes(attribute.Bool("llm.interrupted", reply.interrupted))
	end := reply.end
	if end.IsZero() {
		end = time.Now()
	}
	if end.Before(reply.start) {
		end = reply.start
	}
	span.End(trace.WithTimestamp(end))
}

// exportIfComplete exports the pending reply when nothing more is expected for
// it.
func (t *realtimeTracer) exportIfComplete() {
	if pending := t.pending; pending != nil && pending.complete() {
		t.exportReply(pending)
	}
}

// scheduleGraceExport exports the pending reply after the grace period, unless
// it was already exported.
//
// Exporting the reply stops the timer. The timer still checks that its reply
// is the one waiting, in case it fires before the stop lands.
func (t *realtimeTracer) scheduleGraceExport(reply *realtimeReply) {
	reply.graceTimer = time.AfterFunc(t.grace, func() {
		t.mu.Lock()
		defer t.mu.Unlock()
		if t.pending == reply && !reply.exported {
			t.exportReply(reply)
		}
	})
}

// endReply records the end of the open reply and moves it to pending.
func (t *realtimeTracer) endReply(end time.Time, interrupted bool) {
	reply := t.reply
	if reply == nil {
		return
	}
	if end.Before(reply.start) {
		end = reply.start
	}
	reply.end = end
	reply.interrupted = interrupted
	t.reply = nil
	if t.pending != nil && t.pending != reply {
		t.exportReply(t.pending)
	}
	t.pending = reply
	if reply.complete() {
		t.exportReply(reply)
	} else {
		t.scheduleGraceExport(reply)
	}
}

// openReply opens the span of a reply that is starting.
func (t *realtimeTracer) openReply() {
	if t.pending != nil {
		t.exportReply(t.pending)
	}
	answersSpeech := !t.speechEnd.IsZero()
	start := t.speechEnd
	if !answersSpeech {
		start = time.Now()
	}
	parent := t.turnContext()
	if answersSpeech && t.speechEndTurn.IsValid() {
		parent = trace.ContextWithSpanContext(context.Background(), t.speechEndTurn)
	}
	_, span := tracing.Tracer().Start(parent, "llm_response", trace.WithTimestamp(start))
	t.seq++
	reply := &realtimeReply{span: span, start: start, answersSpeech: answersSpeech, seq: t.seq}
	span.SetAttributes(
		attribute.String("gen_ai.provider.name", tracing.ProviderName(t.svc.TypeName(), "LLM")),
		attribute.String("gen_ai.request.model", t.svc.tracedModelName()),
		attribute.String("gen_ai.operation.name", "chat"),
	)
	// Data that arrived before the start frame belongs to this reply.
	reply.ttfb = t.heldTTFB
	reply.llmText = append([]textPiece(nil), t.heldText...)
	reply.ttsText = append([]textPiece(nil), t.heldTTSText...)
	if answersSpeech && len(t.heldTranscript) > 0 {
		reply.inputPieces = t.heldTranscript
		t.heldTranscript = nil
	}
	t.discardHeld()
	t.speechEnd = time.Time{}
	t.speechEndTurn = trace.SpanContext{}
	// speechOnset is kept: a user already speaking when the reply starts ends
	// it at their onset, clamped to the reply's own start.
	if answersSpeech {
		// This reply answers the user's speech; what they say from now on is
		// theirs. A reply that answers no speech (a tool follow-up) leaves an
		// utterance already under way to the reply that will answer it.
		t.userStartedSinceReply = false
	}
	t.reply = reply
}

// markPendingInterrupted marks a reply the service ended itself on
// interruption.
//
// Some services push the LLMFullResponseEndFrame from their interruption
// handler, so the reply has already ended when the InterruptionFrame passes.
// The user's speech onset falling inside the reply tells that end apart from
// one that ended the reply normally before the user spoke.
func (t *realtimeTracer) markPendingInterrupted() {
	pending, onset := t.pending, t.speechOnset
	if pending == nil || pending.exported || pending.end.IsZero() || onset.IsZero() {
		return
	}
	if onset.Before(pending.end) {
		pending.interrupted = true
		pending.end = onset
		if onset.Before(pending.start) {
			pending.end = pending.start
		}
	}
}

// interruptedEnd is when an interrupted reply ends: the user's speech onset,
// if known.
func (t *realtimeTracer) interruptedEnd() time.Time {
	if !t.speechOnset.IsZero() {
		return t.speechOnset
	}
	return time.Now()
}

// noteSpeechOnset records the user starting to speak.
func (t *realtimeTracer) noteSpeechOnset(onset time.Time) {
	if t.speechOnset.IsZero() || onset.Before(t.speechOnset) {
		t.speechOnset = onset
	}
	t.userStartedSinceReply = true
	t.discardHeld()
}

// noteSpeechEnd records the user's speech ending, and the turn it ended in.
func (t *realtimeTracer) noteSpeechEnd(end time.Time) {
	t.speechEnd = end
	t.speechEndTurn = t.svc.Tracing().TurnContext()
	t.speechOnset = time.Time{}
	t.discardHeld()
}

// handleTranscript routes a user transcript to the reply it answers, or holds
// it for the next.
func (t *realtimeTracer) handleTranscript(text string) {
	if text == "" {
		return
	}
	target := t.reply
	if target == nil {
		target = t.pending
	}
	if target != nil && !t.userStartedSinceReply {
		target.inputPieces = append(target.inputPieces, text)
	} else {
		t.heldTranscript = append(t.heldTranscript, text)
	}
}

// closeAll closes everything still open: pipeline end or service cleanup.
func (t *realtimeTracer) closeAll() {
	if reply := t.reply; reply != nil {
		reply.end = time.Now()
		if reply.end.Before(reply.start) {
			reply.end = reply.start
		}
		reply.interrupted = true
		t.exportReply(reply)
	}
	if t.pending != nil {
		t.exportReply(t.pending)
	}
	for _, span := range t.toolSpans {
		span.SetAttributes(attribute.Bool("tool.cancelled", true))
		span.End()
	}
	t.toolSpans = map[string]trace.Span{}
}

// handleToolFrame follows one function call: its span opens when the call is
// in progress and ends on its final result or its cancellation.
func (t *realtimeTracer) handleToolFrame(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.FunctionCallInProgressFrame:
		if _, open := t.toolSpans[fr.ToolCallID]; open {
			return
		}
		_, span := tracing.Tracer().Start(t.turnContext(), "llm_tool_call")
		span.SetAttributes(
			attribute.String("gen_ai.operation.name", "execute_tool"),
			attribute.String("gen_ai.tool.name", fr.ToolName),
			attribute.String("gen_ai.tool.call.id", fr.ToolCallID),
			attribute.String("gen_ai.tool.call.arguments", toolArgumentsAttr(fr.Args)),
		)
		t.toolSpans[fr.ToolCallID] = span
	case *frames.FunctionCallResultFrame:
		if !fr.Properties.Final() {
			return
		}
		span, open := t.toolSpans[fr.ToolCallID]
		if !open {
			return
		}
		delete(t.toolSpans, fr.ToolCallID)
		span.SetAttributes(attribute.String("gen_ai.tool.call.result", toolResultAttr(fr.Result)))
		if fr.Error != "" {
			span.SetAttributes(attribute.String("error.type", "tool_error"))
			span.SetStatus(codes.Error, fr.Error)
		}
		span.End()
	case *frames.FunctionCallCancelFrame:
		span, open := t.toolSpans[fr.ToolCallID]
		if !open {
			return
		}
		delete(t.toolSpans, fr.ToolCallID)
		span.SetAttributes(attribute.Bool("tool.cancelled", true))
		span.End()
	}
}

// vadTime is the wall-clock time a VAD frame was stamped with, which is now
// for a frame that carries none.
func vadTime(timestamp time.Time) time.Time {
	if timestamp.IsZero() {
		return time.Now()
	}
	return timestamp
}

// secondsBefore is the moment secs seconds before at.
func secondsBefore(at time.Time, secs float64) time.Time {
	return at.Add(-time.Duration(secs * float64(time.Second)))
}

// handleFrame follows one frame the service pushed.
func (t *realtimeTracer) handleFrame(f frames.Frame, dir processor.Direction) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if dir == processor.Upstream {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			t.handleTranscript(tf.Text)
		}
		return
	}

	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		if t.reply == nil {
			t.openReply()
		}
	case *frames.LLMFullResponseEndFrame:
		t.endReply(time.Now(), false)
	case *frames.InterruptionFrame:
		if t.reply != nil {
			t.endReply(t.interruptedEnd(), true)
		} else {
			t.markPendingInterrupted()
		}
		t.discardHeld()
	case *frames.FunctionCallInProgressFrame:
		if t.reply == nil {
			// A function call is model output, so a reply is under way even
			// when no start frame announced it: some services push none for a
			// response that only calls tools.
			t.openReply()
		}
		t.handleToolFrame(fr)
	case *frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame:
		t.handleToolFrame(fr)
	case *frames.EndFrame, *frames.CancelFrame:
		t.closeAll()
	default:
		t.handleSpeechFrame(f)
		t.handleOutputFrame(f)
	}
}

// handleSpeechFrame follows the frames that say when the user speaks: where a
// reply starts, and where an interrupted one ends.
func (t *realtimeTracer) handleSpeechFrame(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.VADUserStoppedSpeakingFrame:
		t.noteSpeechEnd(secondsBefore(vadTime(fr.Timestamp), fr.StopSecs))
	case *frames.VADUserStartedSpeakingFrame:
		t.noteSpeechOnset(secondsBefore(vadTime(fr.Timestamp), fr.StartSecs))
	case *frames.UserStoppedSpeakingFrame:
		// A turn strategy can end the user's turn after a service using its own
		// VAD has started replying; that stop is not the next reply's start.
		// Local VAD, when present, is the better measure.
		if t.reply == nil && t.speechEnd.IsZero() {
			t.noteSpeechEnd(time.Now())
			return
		}
		if t.reply != nil && !t.speechOnset.IsZero() && t.speechOnset.Before(t.reply.start) {
			// The onset was of the speech this stop ends, not a barge-in on the
			// reply, so the next onset must replace it.
			t.speechOnset = time.Time{}
		}
		t.discardHeld()
	case *frames.UserStartedSpeakingFrame:
		t.noteSpeechOnset(time.Now())
	case *frames.ProposedUserStartedSpeakingFrame:
		t.noteSpeechOnset(time.Now())
	}
}

// handleOutputFrame follows what the model produces for a reply: its text, its
// audio and the time to first byte measured for it. What arrives with no reply
// open is held for the next one.
func (t *realtimeTracer) handleOutputFrame(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		piece := textPiece{text: fr.Text, includesSpaces: fr.IncludesInterFrameSpaces}
		if t.reply != nil {
			t.reply.llmText = append(t.reply.llmText, piece)
		} else {
			t.heldText = append(t.heldText, piece)
		}
	case *frames.TTSTextFrame:
		piece := textPiece{text: fr.Text, includesSpaces: fr.IncludesInterFrameSpaces}
		if t.reply != nil {
			t.reply.ttsText = append(t.reply.ttsText, piece)
		} else {
			t.heldTTSText = append(t.heldTTSText, piece)
		}
	case *frames.TTSAudioRawFrame:
		if t.reply != nil {
			t.reply.hadAudio = true
		}
	case *frames.MetricsFrame:
		t.handleMetrics(fr)
	}
}

// handleMetrics takes the time to first byte this service reported: for the
// open reply when it has none yet, else held for the next one.
func (t *realtimeTracer) handleMetrics(f *frames.MetricsFrame) {
	for _, d := range f.Data {
		data, ok := d.(frames.TTFBMetricsData)
		if !ok || data.Processor != t.svc.Name() {
			continue
		}
		ttfb := data.Value.Seconds()
		switch {
		case t.reply != nil:
			if t.reply.ttfb == nil {
				t.reply.ttfb = &ttfb
			}
		case t.heldTTFB == nil:
			t.heldTTFB = &ttfb
		}
		return
	}
}

// handleUsage adds a usage report to the reply it belongs to: the open one,
// else the one waiting to be exported, which it may complete.
func (t *realtimeTracer) handleUsage(u frames.LLMTokenUsage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	target := t.reply
	if target == nil {
		target = t.pending
	}
	if target == nil {
		return
	}
	target.usage = sumTokenUsage(target.usage, u)
	if target == t.pending {
		t.exportIfComplete()
	}
}

// cleanup closes everything still open when the service is cleaned up.
func (t *realtimeTracer) cleanup() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closeAll()
}

// traceRealtimeFrame follows a frame this service has pushed, on a traced
// realtime service.
func (b *Base) traceRealtimeFrame(f frames.Frame, dir processor.Direction) {
	if b.realtime != nil && b.TracingEnabled() {
		b.realtime.handleFrame(f, dir)
	}
}

// StartLLMUsageMetrics reports the token usage of a reply. It is for a service
// generating continuously, which calls it with every usage report its provider
// sends, without checking whether usage metrics are collected: the usage is
// pushed as a MetricsFrame only when they are, and on a traced realtime
// service it is recorded on the reply's llm_response span either way.
func (b *Base) StartLLMUsageMetrics(ctx context.Context, u frames.LLMTokenUsage) error {
	var err error
	if b.UsageMetricsEnabled() {
		err = b.PushTokenUsage(ctx, u)
	}
	if err == nil && b.realtime != nil && b.TracingEnabled() {
		b.realtime.handleUsage(u)
	}
	return err
}
