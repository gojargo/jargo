package llm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// The realtime tracer is driven through a service reduced to its base: frames
// are pushed as a realtime service would push them, and the spans that ended
// are checked.

// testGrace is the grace period the tests run with. Replies that answer speech
// export after it, so it is kept short.
const testGrace = 50 * time.Millisecond

// realtimeStubGen is the generator of a service that generates continuously,
// which is never asked to generate.
type realtimeStubGen struct{}

func (realtimeStubGen) Generate(context.Context, *frames.LLMContext, Emit) error {
	return nil
}

// realtimeStub is a traced realtime service and what its spans are read from.
type realtimeStub struct {
	*Base
	t   *testing.T
	rec *tracetest.SpanRecorder
	tc  *tracing.TracingContext
}

// newRealtimeStub builds a realtime service set up on a pipeline traced or not.
func newRealtimeStub(t *testing.T, traced bool) *realtimeStub {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	svc := New("StubRealtime", realtimeStubGen{}, WithContinuousGeneration(), WithRealtimeTracing())
	svc.SetModel("rt-model")
	svc.realtime.grace = testGrace
	tc := tracing.NewTracingContext()
	ctx := context.Background()
	if err := svc.Setup(ctx, processor.Setup{Clock: clock.NewSystem(), TracingEnabled: traced, Tracing: tc}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = svc.Cleanup(ctx) })
	return &realtimeStub{Base: svc, t: t, rec: rec, tc: tc}
}

// push pushes a frame downstream, as the service pushes most of what it sends.
func (s *realtimeStub) push(f frames.Frame) {
	s.t.Helper()
	if err := s.PushFrame(context.Background(), f, processor.Downstream); err != nil {
		s.t.Fatalf("push: %v", err)
	}
}

// pushUp pushes a frame upstream.
func (s *realtimeStub) pushUp(f frames.Frame) {
	s.t.Helper()
	if err := s.PushFrame(context.Background(), f, processor.Upstream); err != nil {
		s.t.Fatalf("push upstream: %v", err)
	}
}

// usage reports token usage, as the service does with what its provider sends.
func (s *realtimeStub) usage(u frames.LLMTokenUsage) {
	s.t.Helper()
	if err := s.StartLLMUsageMetrics(context.Background(), u); err != nil {
		s.t.Fatalf("usage: %v", err)
	}
}

// settle lets the grace period run out so pending replies export.
func settle() { time.Sleep(3 * testGrace) }

// spans are the ended spans with the given name, in the order they ended.
func (s *realtimeStub) spans(name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, sp := range s.rec.Ended() {
		if sp.Name() == name {
			out = append(out, sp)
		}
	}
	return out
}

// replies are the llm_response spans that ended.
func (s *realtimeStub) replies() []sdktrace.ReadOnlySpan { return s.spans("llm_response") }

// oneReply is the single llm_response span that ended.
func (s *realtimeStub) oneReply() sdktrace.ReadOnlySpan {
	s.t.Helper()
	got := s.replies()
	if len(got) != 1 {
		s.t.Fatalf("llm_response spans = %d, want one", len(got))
	}
	return got[0]
}

// noReplies fails when an llm_response span has ended.
func (s *realtimeStub) noReplies(why string) {
	s.t.Helper()
	if got := s.replies(); len(got) != 0 {
		s.t.Fatalf("llm_response spans = %d, want none: %s", len(got), why)
	}
}

// oneToolCall is the single llm_tool_call span that ended.
func (s *realtimeStub) oneToolCall() sdktrace.ReadOnlySpan {
	s.t.Helper()
	got := s.spans("llm_tool_call")
	if len(got) != 1 {
		s.t.Fatalf("llm_tool_call spans = %d, want one", len(got))
	}
	return got[0]
}

// turnContext opens a turn span, makes it current, and returns it.
func (s *realtimeStub) turnContext() trace.Span {
	_, span := tracing.Tracer().Start(context.Background(), "turn")
	s.tc.SetTurnContext(span.SpanContext())
	return span
}

// spanAttrs are the attributes of a span by key.
func spanAttrs(sp sdktrace.ReadOnlySpan) map[string]attribute.Value {
	out := map[string]attribute.Value{}
	for _, kv := range sp.Attributes() {
		out[string(kv.Key)] = kv.Value
	}
	return out
}

// wantString checks a string attribute.
func wantString(t *testing.T, attrs map[string]attribute.Value, key, want string) {
	t.Helper()
	v, ok := attrs[key]
	if !ok || v.AsString() != want {
		t.Errorf("%s = %q (set: %v), want %q", key, v.AsString(), ok, want)
	}
}

// wantInt checks an integer attribute.
func wantInt(t *testing.T, attrs map[string]attribute.Value, key string, want int64) {
	t.Helper()
	v, ok := attrs[key]
	if !ok || v.AsInt64() != want {
		t.Errorf("%s = %d (set: %v), want %d", key, v.AsInt64(), ok, want)
	}
}

// wantBool checks a boolean attribute.
func wantBool(t *testing.T, attrs map[string]attribute.Value, key string, want bool) {
	t.Helper()
	v, ok := attrs[key]
	if !ok || v.AsBool() != want {
		t.Errorf("%s = %v (set: %v), want %v", key, v.AsBool(), ok, want)
	}
}

// wantAbsent checks that an attribute was left off.
func wantAbsent(t *testing.T, attrs map[string]attribute.Value, key string) {
	t.Helper()
	if v, ok := attrs[key]; ok {
		t.Errorf("%s = %v, want it left off", key, v.String())
	}
}

// tokenUsage is a usage report of prompt and completion tokens.
func tokenUsage(prompt, completion int64) frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens: prompt, CompletionTokens: completion, TotalTokens: prompt + completion,
	}
}

// someUsage is the usage report most tests make.
func someUsage() frames.LLMTokenUsage { return tokenUsage(10, 5) }

// ttfbFrame is a time to first byte reported by the named processor.
func ttfbFrame(name string, value time.Duration) *frames.MetricsFrame {
	var data frames.TTFBMetricsData
	data.Processor, data.Value = name, value
	return frames.NewMetricsFrame(data)
}

func audioFrame() *frames.TTSAudioRawFrame {
	return frames.NewTTSAudioRawFrame([]byte{0, 0}, 16000, 1)
}

func transcript(text string) *frames.TranscriptionFrame {
	return frames.NewTranscriptionFrame(text, "", "")
}

func vadStopped() *frames.VADUserStoppedSpeakingFrame {
	return frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now())
}

func vadStoppedAt(speechEnd time.Time) *frames.VADUserStoppedSpeakingFrame {
	return frames.NewVADUserStoppedSpeakingFrame(0.2, speechEnd.Add(200*time.Millisecond))
}

func llmStart() *frames.LLMFullResponseStartFrame { return frames.NewLLMFullResponseStartFrame() }

func llmEnd() *frames.LLMFullResponseEndFrame { return frames.NewLLMFullResponseEndFrame() }

func llmText(text string) *frames.LLMTextFrame { return frames.NewLLMTextFrame(text) }

func ttsSentence(text string) *frames.TTSTextFrame {
	return frames.NewTTSTextFrame(text, frames.TextTypeSentence)
}

func inProgress(id, name, args string) *frames.FunctionCallInProgressFrame {
	return frames.NewFunctionCallInProgressFrame(id, name, json.RawMessage(args), true, "")
}

func callResult(id, name, args, result string) *frames.FunctionCallResultFrame {
	return frames.NewFunctionCallResultFrame(id, name, json.RawMessage(args), result)
}

// normalReply is the user stopping, the model replying, the end, then usage.
func (s *realtimeStub) normalReply(speechEnd time.Time) {
	s.t.Helper()
	s.push(vadStoppedAt(speechEnd))
	s.push(llmStart())
	s.push(llmText("Hi there."))
	s.push(audioFrame())
	s.push(llmEnd())
	s.usage(someUsage())
	settle()
}

// ---------------------------------------------------------------------------
// Reply spans
// ---------------------------------------------------------------------------

func TestRealtimeNormalReplyStartsAtSpeechEndWithAttributes(t *testing.T) {
	s := newRealtimeStub(t, true)
	speechEnd := time.Now().Add(-time.Second)
	s.pushUp(transcript("What time is it?"))
	beforeEnd := time.Now()
	s.normalReply(speechEnd)

	span := s.oneReply()
	if !span.StartTime().Equal(speechEnd) {
		t.Errorf("start = %v, want the speech end %v", span.StartTime(), speechEnd)
	}
	if span.EndTime().Before(beforeEnd) {
		t.Errorf("end = %v, want it after %v", span.EndTime(), beforeEnd)
	}
	attrs := spanAttrs(span)
	wantString(t, attrs, "gen_ai.provider.name", "stubrealtime")
	wantString(t, attrs, "gen_ai.request.model", "rt-model")
	wantString(t, attrs, "gen_ai.operation.name", "chat")
	wantString(t, attrs, "gen_ai.output.type", "speech")
	wantString(t, attrs, "input", "What time is it?")
	wantString(t, attrs, "input_modality", "AUDIO")
	wantString(t, attrs, "output", "Hi there.")
	wantString(t, attrs, "output_modality", "AUDIO")
	wantInt(t, attrs, "gen_ai.usage.input_tokens", 10)
	wantInt(t, attrs, "gen_ai.usage.output_tokens", 5)
	wantBool(t, attrs, "llm.interrupted", false)
}

func TestRealtimeUsageAndTranscriptAfterEndAreWaitedFor(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(llmStart())
	s.push(llmText("Sure."))
	endBefore := time.Now()
	s.push(llmEnd())
	endAfter := time.Now()
	s.noReplies("the reply waits for its late data")

	s.usage(someUsage())
	s.noReplies("still waiting for the transcript")

	s.pushUp(transcript("Hello."))
	s.noReplies("the transcript may come in several pieces")

	settle()
	span := s.oneReply()
	attrs := spanAttrs(span)
	wantString(t, attrs, "input", "Hello.")
	wantInt(t, attrs, "gen_ai.usage.input_tokens", 10)
	// The span ends when the end frame passed, not when the late data arrived.
	if span.EndTime().Before(endBefore) || span.EndTime().After(endAfter) {
		t.Errorf("end = %v, want it between %v and %v", span.EndTime(), endBefore, endAfter)
	}
}

func TestRealtimeTranscriptSentencesAfterUsageAllBelongToTheReply(t *testing.T) {
	// Some services push one TranscriptionFrame per sentence, often after the
	// reply has ended and its usage has arrived.
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(llmStart())
	s.push(llmText("Sunny, 21 degrees."))
	s.push(llmEnd())
	s.usage(someUsage())
	s.pushUp(transcript("What's the weather in Rome?"))
	s.pushUp(transcript(" Use Celsius."))
	s.noReplies("the reply answers speech, so it waits")

	// The next reply closes the pending one without waiting for the grace
	// period.
	s.push(vadStopped())
	s.push(llmStart())
	s.pushUp(transcript("Thanks."))
	s.push(llmEnd())
	s.usage(someUsage())
	settle()

	got := s.replies()
	if len(got) != 2 {
		t.Fatalf("llm_response spans = %d, want two", len(got))
	}
	wantString(t, spanAttrs(got[0]), "input", "What's the weather in Rome? Use Celsius.")
	wantString(t, spanAttrs(got[1]), "input", "Thanks.")
}

func TestRealtimeReplyWithoutSpeechExportsAsSoonAsUsageArrives(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(llmText("Hello! How can I help?"))
	s.push(llmEnd())
	s.noReplies("the reply waits for its usage")
	s.usage(someUsage())
	wantAbsent(t, spanAttrs(s.oneReply()), "input")
}

func TestRealtimeGracePeriodExportsAnIncompleteReply(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(llmStart())
	s.push(llmText("Hi there."))
	s.push(llmEnd())
	s.noReplies("the grace period has not run out")

	settle()
	attrs := spanAttrs(s.oneReply())
	wantAbsent(t, attrs, "gen_ai.usage.input_tokens")
	wantString(t, attrs, "output", "Hi there.")
}

func TestRealtimeInterruptionWithLateUsageAndDuplicateEnd(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(llmStart())
	s.pushUp(transcript("Tell me a story"))
	s.push(llmText("Let me tell you a long"))
	onset := time.Now()
	s.push(frames.NewVADUserStartedSpeakingFrame(0.1, onset.Add(100*time.Millisecond)))
	s.push(frames.NewInterruptionFrame())
	// The canceled response finishes: the end again, then its usage.
	s.push(llmEnd())
	s.noReplies("the interrupted reply waits for its late data")
	s.usage(tokenUsage(3, 1))

	settle()
	span := s.oneReply()
	attrs := spanAttrs(span)
	wantString(t, attrs, "input", "Tell me a story")
	wantBool(t, attrs, "llm.interrupted", true)
	wantInt(t, attrs, "gen_ai.usage.input_tokens", 3)
	if !span.EndTime().Equal(onset) {
		t.Errorf("end = %v, want the speech onset %v", span.EndTime(), onset)
	}
}

func TestRealtimeBargeInTranscriptGoesToTheNextReply(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.pushUp(transcript("First question"))
	s.push(llmStart())
	s.push(llmText("Answer one"))
	s.push(frames.NewVADUserStartedSpeakingFrame(0.1, time.Now()))
	// The user is speaking again, so this transcript is theirs, not the first
	// reply's.
	s.pushUp(transcript("Second question"))
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())
	s.push(vadStopped())
	s.push(llmStart())
	s.push(llmText("Answer two"))
	s.push(llmEnd())
	s.usage(someUsage())

	settle()
	got := s.replies()
	if len(got) != 2 {
		t.Fatalf("llm_response spans = %d, want two", len(got))
	}
	first, second := spanAttrs(got[0]), spanAttrs(got[1])
	wantString(t, first, "input", "First question")
	wantString(t, first, "output", "Answer one")
	wantString(t, second, "input", "Second question")
	wantString(t, second, "output", "Answer two")
}

func TestRealtimeDuplicateStartsMakeOneSpanAndABareEndMakesNone(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmEnd())
	s.noReplies("an end with no reply open")

	s.push(llmStart())
	s.push(llmStart())
	s.push(llmText("One reply"))
	s.push(llmEnd())
	s.usage(someUsage())
	s.push(llmEnd())

	wantString(t, spanAttrs(s.oneReply()), "output", "One reply")
}

func TestRealtimeUpstreamCopiesOfBroadcastFramesAreIgnored(t *testing.T) {
	s := newRealtimeStub(t, true)
	longAgo := time.Now().Add(-time.Minute)
	s.pushUp(frames.NewVADUserStoppedSpeakingFrame(0.2, longAgo))
	before := time.Now()
	s.push(llmStart())
	s.push(llmEnd())
	s.usage(someUsage())

	if span := s.oneReply(); span.StartTime().Before(before) {
		t.Errorf("start = %v, want it after %v", span.StartTime(), before)
	}
}

func TestRealtimeUserStoppedSpeakingIsTheFallbackStartAndALateOneIsIgnored(t *testing.T) {
	s := newRealtimeStub(t, true)
	before := time.Now()
	s.push(frames.NewUserStoppedSpeakingFrame())
	after := time.Now()
	s.pushUp(transcript("Hi"))
	time.Sleep(10 * time.Millisecond)
	s.push(llmStart())
	// A turn strategy ends the user turn after the service already started
	// replying.
	s.push(frames.NewUserStoppedSpeakingFrame())
	s.push(llmEnd())
	s.usage(someUsage())

	settle()
	span := s.oneReply()
	if span.StartTime().Before(before) || span.StartTime().After(after) {
		t.Errorf("start = %v, want it between %v and %v", span.StartTime(), before, after)
	}
	wantString(t, spanAttrs(span), "input", "Hi")

	// The late stop must not become the next reply's start either.
	s.push(llmStart())
	s.push(llmEnd())
	s.usage(someUsage())
	settle()
	got := s.replies()
	if len(got) != 2 {
		t.Fatalf("llm_response spans = %d, want two", len(got))
	}
	if !got[1].StartTime().After(after) {
		t.Errorf("second start = %v, want it after %v", got[1].StartTime(), after)
	}
}

func TestRealtimeBargeInAfterALateUserStopEndsTheReplyAtTheNewOnset(t *testing.T) {
	// The service's own VAD starts the reply before the turn strategy's stop.
	s := newRealtimeStub(t, true)
	s.push(frames.NewProposedUserStartedSpeakingFrame())
	s.push(frames.NewUserStartedSpeakingFrame())
	time.Sleep(10 * time.Millisecond)
	s.push(llmStart())
	s.push(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(100 * time.Millisecond)
	beforeBargeIn := time.Now()
	s.push(frames.NewProposedUserStartedSpeakingFrame())
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())

	settle()
	span := s.oneReply()
	wantBool(t, spanAttrs(span), "llm.interrupted", true)
	if span.EndTime().Before(beforeBargeIn) {
		t.Errorf("end = %v, want it after the barge-in %v", span.EndTime(), beforeBargeIn)
	}
}

func TestRealtimeVADSpeechEndWinsOverUserStoppedSpeaking(t *testing.T) {
	s := newRealtimeStub(t, true)
	speechEnd := time.Now().Add(-2 * time.Second)
	s.push(vadStoppedAt(speechEnd))
	s.push(frames.NewUserStoppedSpeakingFrame())
	s.pushUp(transcript("Hi"))
	s.push(llmStart())
	s.push(llmEnd())
	s.usage(someUsage())

	settle()
	if span := s.oneReply(); !span.StartTime().Equal(speechEnd) {
		t.Errorf("start = %v, want the speech end %v", span.StartTime(), speechEnd)
	}
}

func TestRealtimeReplyWithNoSpeechBeforeItHasNoInput(t *testing.T) {
	s := newRealtimeStub(t, true)
	before := time.Now()
	s.push(llmStart())
	s.push(llmText("Hello! I'm your assistant."))
	s.push(llmEnd())
	s.usage(someUsage())

	span := s.oneReply()
	if span.StartTime().Before(before) {
		t.Errorf("start = %v, want it after %v", span.StartTime(), before)
	}
	attrs := spanAttrs(span)
	wantAbsent(t, attrs, "input")
	wantAbsent(t, attrs, "input_modality")
}

func TestRealtimeReplyIsParentedToTheTurnOfItsSpeechEnd(t *testing.T) {
	s := newRealtimeStub(t, true)
	turnA := s.turnContext()
	s.push(vadStopped())
	s.pushUp(transcript("Hi"))
	turnA.End()
	turnB := s.turnContext()
	s.push(llmStart())
	s.push(llmEnd())
	s.usage(someUsage())
	turnB.End()

	settle()
	if got, want := s.oneReply().Parent().SpanID(), turnA.SpanContext().SpanID(); got != want {
		t.Errorf("parent = %v, want the turn the speech ended in %v", got, want)
	}
}

func TestRealtimeReplyWithoutSpeechEndIsParentedToTheCurrentTurn(t *testing.T) {
	s := newRealtimeStub(t, true)
	turn := s.turnContext()
	s.push(llmStart())
	s.push(llmEnd())
	s.usage(someUsage())
	turn.End()

	if got, want := s.oneReply().Parent().SpanID(), turn.SpanContext().SpanID(); got != want {
		t.Errorf("parent = %v, want the current turn %v", got, want)
	}
}

func TestRealtimePipelineEndClosesAnOpenReply(t *testing.T) {
	closers := map[string]func(*realtimeStub){
		"end_frame":    func(s *realtimeStub) { s.push(frames.NewEndFrame()) },
		"cancel_frame": func(s *realtimeStub) { s.push(frames.NewCancelFrame()) },
		"cleanup":      func(s *realtimeStub) { _ = s.Cleanup(context.Background()) },
	}
	for name, closer := range closers {
		t.Run(name, func(t *testing.T) {
			s := newRealtimeStub(t, true)
			s.push(llmStart())
			s.push(llmText("Bye"))
			closer(s)

			attrs := spanAttrs(s.oneReply())
			wantString(t, attrs, "output", "Bye")
			wantBool(t, attrs, "llm.interrupted", true)
		})
	}
}

func TestRealtimePipelineEndExportsAPendingReply(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(llmEnd())
	s.noReplies("the reply waits for its usage")
	s.push(frames.NewEndFrame())
	wantBool(t, spanAttrs(s.oneReply()), "llm.interrupted", false)
}

func TestRealtimeExportingAReplyCancelsItsGraceTimer(t *testing.T) {
	closers := map[string]func(*realtimeStub){
		"end_frame":  func(s *realtimeStub) { s.push(frames.NewEndFrame()) },
		"next_reply": func(s *realtimeStub) { s.push(llmStart()) },
	}
	for name, closer := range closers {
		t.Run(name, func(t *testing.T) {
			s := newRealtimeStub(t, true)
			s.realtime.grace = 10 * time.Second
			s.push(vadStopped())
			s.push(llmStart())
			s.push(llmEnd())
			s.usage(someUsage())
			s.realtime.mu.Lock()
			pending := s.realtime.pending
			scheduled := pending != nil && pending.graceTimer != nil
			s.realtime.mu.Unlock()
			if !scheduled {
				t.Fatal("no grace timer was scheduled for the pending reply")
			}
			closer(s)

			if got := len(s.replies()); got != 1 {
				t.Errorf("llm_response spans = %d, want one", got)
			}
			s.realtime.mu.Lock()
			defer s.realtime.mu.Unlock()
			if !pending.exported || pending.graceTimer != nil {
				t.Errorf("exported = %v, timer left = %v, want the reply exported and its timer stopped",
					pending.exported, pending.graceTimer != nil)
			}
		})
	}
}

func TestRealtimePipelineEndMidReplyLeavesNoGraceTimer(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(llmStart())
	s.realtime.mu.Lock()
	reply := s.realtime.reply
	s.realtime.mu.Unlock()
	s.push(frames.NewEndFrame())
	_ = s.Cleanup(context.Background())

	wantBool(t, spanAttrs(s.oneReply()), "llm.interrupted", true)
	s.realtime.mu.Lock()
	defer s.realtime.mu.Unlock()
	if reply == nil || reply.graceTimer != nil {
		t.Errorf("a grace timer was scheduled for a reply closed with the pipeline")
	}
}

func TestRealtimeTTSTextIsTheFallbackOutputIncludingTextBeforeStart(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(ttsSentence("Weather"))
	s.push(llmStart())
	s.push(ttsSentence("today"))
	s.push(llmEnd())
	s.usage(someUsage())

	attrs := spanAttrs(s.oneReply())
	wantString(t, attrs, "output", "Weather today")
	wantString(t, attrs, "output_modality", "TEXT")
	wantString(t, attrs, "gen_ai.output.type", "text")
}

func TestRealtimeLLMTextIsPreferredOverTTSText(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(llmText("Hello "))
	s.push(ttsSentence("Hello"))
	s.push(llmText("world."))
	s.push(ttsSentence("world."))
	s.push(llmEnd())
	s.usage(someUsage())

	wantString(t, spanAttrs(s.oneReply()), "output", "Hello world.")
}

func TestRealtimeTTFBBeforeStartBelongsToTheNextReply(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(vadStopped())
	s.push(ttfbFrame(s.Name(), 420*time.Millisecond))
	s.push(llmStart())
	s.push(ttfbFrame(s.Name(), 990*time.Millisecond))
	s.push(llmEnd())
	s.usage(someUsage())
	s.pushUp(transcript("Hi"))

	settle()
	attrs := spanAttrs(s.oneReply())
	for _, key := range []string{"metrics.ttfb", "gen_ai.response.time_to_first_chunk"} {
		if got := attrs[key].AsFloat64(); got != 0.42 {
			t.Errorf("%s = %v, want 0.42", key, got)
		}
	}
}

func TestRealtimeMetricsFromOtherProcessorsAreIgnored(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(ttfbFrame("some-stt", 310*time.Millisecond))
	s.push(llmEnd())
	s.usage(someUsage())

	wantAbsent(t, spanAttrs(s.oneReply()), "metrics.ttfb")
}

func TestRealtimePartialUsageReportsAreSummed(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	first := tokenUsage(100, 10)
	first.ReasoningTokens = new(int64(4))
	s.usage(first)
	second := tokenUsage(20, 5)
	second.InputAudioTokens = new(int64(7))
	s.usage(second)
	s.push(llmEnd())

	attrs := spanAttrs(s.oneReply())
	wantInt(t, attrs, "gen_ai.usage.input_tokens", 120)
	wantInt(t, attrs, "gen_ai.usage.output_tokens", 15)
	wantInt(t, attrs, "gen_ai.usage.reasoning.output_tokens", 4)
	wantInt(t, attrs, "gen_ai.usage.audio.input_tokens", 7)
}

func TestRealtimeTracingDisabledRecordsNothing(t *testing.T) {
	s := newRealtimeStub(t, false)
	s.normalReply(time.Now())
	s.push(inProgress("c1", "f", "{}"))
	if got := s.rec.Ended(); len(got) != 0 {
		t.Errorf("spans = %d, want none on a pipeline that is not traced", len(got))
	}
	s.realtime.mu.Lock()
	defer s.realtime.mu.Unlock()
	if s.realtime.reply != nil || s.realtime.pending != nil || len(s.realtime.toolSpans) != 0 {
		t.Error("the tracer followed frames on a pipeline that is not traced")
	}
}

// ---------------------------------------------------------------------------
// Interrupted reply end times
// ---------------------------------------------------------------------------

func TestRealtimeInterruptedReplyEndsAtTheVADOnset(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	time.Sleep(10 * time.Millisecond)
	onset := time.Now()
	s.push(frames.NewVADUserStartedSpeakingFrame(0.15, onset.Add(150*time.Millisecond)))
	time.Sleep(10 * time.Millisecond)
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())

	if span := s.oneReply(); !span.EndTime().Equal(onset) {
		t.Errorf("end = %v, want the speech onset %v", span.EndTime(), onset)
	}
}

func TestRealtimeInterruptedReplyEndsAtTheServicesProposedStart(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	time.Sleep(10 * time.Millisecond)
	before := time.Now()
	s.push(frames.NewProposedUserStartedSpeakingFrame())
	after := time.Now()
	time.Sleep(10 * time.Millisecond)
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())

	if span := s.oneReply(); span.EndTime().Before(before) || span.EndTime().After(after) {
		t.Errorf("end = %v, want it between %v and %v", span.EndTime(), before, after)
	}
}

// Some services push the end from their interruption handler, before the
// InterruptionFrame.
func TestRealtimeEndPushedByTheInterruptionHandlerStillCountsAsInterrupted(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(llmText("Once upon a time"))
	time.Sleep(10 * time.Millisecond)
	onset := time.Now()
	s.push(frames.NewProposedUserStartedSpeakingFrame())
	time.Sleep(10 * time.Millisecond)
	s.push(llmEnd())
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())

	span := s.oneReply()
	wantBool(t, spanAttrs(span), "llm.interrupted", true)
	if d := span.EndTime().Sub(onset).Abs(); d >= 5*time.Millisecond {
		t.Errorf("end is %v from the speech onset, want it there", d)
	}
}

func TestRealtimeInterruptionAfterANormallyEndedReplyLeavesItAlone(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(llmStart())
	s.push(llmEnd())
	end := time.Now()
	time.Sleep(10 * time.Millisecond)
	// The user speaks later, during silence; the aggregator still broadcasts an
	// interruption.
	s.push(frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())

	span := s.oneReply()
	wantBool(t, spanAttrs(span), "llm.interrupted", false)
	if span.EndTime().After(end) {
		t.Errorf("end = %v, want it no later than %v", span.EndTime(), end)
	}
}

func TestRealtimeInterruptedReplyNeverEndsBeforeItsStart(t *testing.T) {
	// The user was already speaking when the reply started (a tool follow-up
	// colliding with the next utterance, for one).
	s := newRealtimeStub(t, true)
	s.push(frames.NewVADUserStartedSpeakingFrame(0.1, time.Now()))
	time.Sleep(10 * time.Millisecond)
	s.push(llmStart())
	// The utterance under way is not the follow-up's input, even though it
	// arrives while the follow-up is open.
	s.pushUp(transcript("Actually, wait."))
	s.push(frames.NewInterruptionFrame())
	s.usage(someUsage())
	s.push(vadStopped())
	s.push(llmStart())
	s.push(llmText("Sure, go ahead."))
	s.push(llmEnd())
	s.usage(someUsage())
	settle()

	got := s.replies()
	if len(got) != 2 {
		t.Fatalf("llm_response spans = %d, want two", len(got))
	}
	followUp, answer := got[0], got[1]
	if !followUp.EndTime().Equal(followUp.StartTime()) {
		t.Errorf("follow-up ends at %v, want its start %v", followUp.EndTime(), followUp.StartTime())
	}
	wantAbsent(t, spanAttrs(followUp), "input")
	wantString(t, spanAttrs(answer), "input", "Actually, wait.")
}

// ---------------------------------------------------------------------------
// Tool call spans
// ---------------------------------------------------------------------------

func TestRealtimeToolCallSpanCoversInProgressToFinalResult(t *testing.T) {
	s := newRealtimeStub(t, true)
	turn := s.turnContext()
	s.push(inProgress("c1", "get_weather", `{"city":"Rome"}`))
	if got := s.spans("llm_tool_call"); len(got) != 0 {
		t.Fatalf("llm_tool_call spans = %d, want the call still open", len(got))
	}
	s.push(callResult("c1", "get_weather", `{"city":"Rome"}`, `{"temp_c":21}`))
	turn.End()

	span := s.oneToolCall()
	attrs := spanAttrs(span)
	wantString(t, attrs, "gen_ai.operation.name", "execute_tool")
	wantString(t, attrs, "gen_ai.tool.name", "get_weather")
	wantString(t, attrs, "gen_ai.tool.call.id", "c1")
	wantString(t, attrs, "gen_ai.tool.call.arguments", `{"city":"Rome"}`)
	wantString(t, attrs, "gen_ai.tool.call.result", `{"temp_c":21}`)
	if got, want := span.Parent().SpanID(), turn.SpanContext().SpanID(); got != want {
		t.Errorf("parent = %v, want the turn %v", got, want)
	}
	if got := span.Status().Code; got != codes.Unset {
		t.Errorf("status = %v, want it unset", got)
	}
}

func TestRealtimeToolOnlyResponseWithoutStartFrameOpensAReply(t *testing.T) {
	// Some services push no LLMFullResponseStartFrame for a response that only
	// calls a tool: the function call frames, then usage and the end, then a
	// start for the follow-up response.
	s := newRealtimeStub(t, true)
	speechEnd := time.Now().Add(-500 * time.Millisecond)
	s.push(vadStoppedAt(speechEnd))
	s.pushUp(transcript("Weather in Rome?"))
	s.push(inProgress("c1", "get_weather", `{"city":"Rome"}`))
	s.usage(tokenUsage(7, 2))
	s.push(llmEnd())
	s.push(callResult("c1", "get_weather", `{"city":"Rome"}`, `{"temp_c":21}`))
	s.push(llmStart())
	s.push(llmText("Sunny, 21 degrees."))
	s.push(llmEnd())
	s.usage(tokenUsage(9, 4))

	got := s.replies()
	if len(got) != 2 {
		t.Fatalf("llm_response spans = %d, want two", len(got))
	}
	toolReply, followUp := got[0], got[1]
	if !toolReply.StartTime().Equal(speechEnd) {
		t.Errorf("start = %v, want the speech end %v", toolReply.StartTime(), speechEnd)
	}
	wantString(t, spanAttrs(toolReply), "input", "Weather in Rome?")
	wantInt(t, spanAttrs(toolReply), "gen_ai.usage.input_tokens", 7)
	wantAbsent(t, spanAttrs(toolReply), "output")
	wantString(t, spanAttrs(followUp), "output", "Sunny, 21 degrees.")
	wantInt(t, spanAttrs(followUp), "gen_ai.usage.input_tokens", 9)
	wantString(t, spanAttrs(s.oneToolCall()), "gen_ai.tool.name", "get_weather")
}

func TestRealtimeToolCallErrorSetsErrorStatus(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(inProgress("c2", "boom", "{}"))
	failed := callResult("c2", "boom", "{}", "The tool failed.")
	failed.Error = "ValueError: nope"
	s.push(failed)

	span := s.oneToolCall()
	if got := span.Status().Code; got != codes.Error {
		t.Errorf("status = %v, want an error", got)
	}
	wantString(t, spanAttrs(span), "error.type", "tool_error")
}

func TestRealtimeIntermediateResultKeepsTheToolSpanOpen(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(inProgress("c3", "slow", "{}"))
	working := callResult("c3", "slow", "{}", "working...")
	working.Properties = &frames.FunctionCallResultProperties{IsFinal: new(false)}
	s.push(working)
	if got := s.spans("llm_tool_call"); len(got) != 0 {
		t.Fatalf("llm_tool_call spans = %d, want the call still open", len(got))
	}
	done := callResult("c3", "slow", "{}", "done")
	done.Properties = &frames.FunctionCallResultProperties{IsFinal: new(true)}
	s.push(done)
	wantString(t, spanAttrs(s.oneToolCall()), "gen_ai.tool.call.result", `"done"`)
}

func TestRealtimeCanceledToolCallIsMarked(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(inProgress("c4", "slow", "{}"))
	s.push(frames.NewFunctionCallCancelFrame("c4", "slow"))

	wantBool(t, spanAttrs(s.oneToolCall()), "tool.cancelled", true)
}

func TestRealtimeToolAttributesAreTruncated(t *testing.T) {
	s := newRealtimeStub(t, true)
	big := `{"blob":"` + strings.Repeat("x", 5000) + `"}`
	s.push(inProgress("c5", "big", big))
	s.push(callResult("c5", "big", big, big))

	attrs := spanAttrs(s.oneToolCall())
	arguments := attrs["gen_ai.tool.call.arguments"].AsString()
	if len(arguments) != 2003 || !strings.HasSuffix(arguments, "...") {
		t.Errorf("arguments are %d long, want them cut to 2000 and marked", len(arguments))
	}
	if result := attrs["gen_ai.tool.call.result"].AsString(); len(result) != 2003 {
		t.Errorf("result is %d long, want it cut to 2000 and marked", len(result))
	}
}

func TestRealtimePipelineEndClosesOpenToolCalls(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.push(inProgress("c6", "slow", "{}"))
	s.push(frames.NewEndFrame())

	wantBool(t, spanAttrs(s.oneToolCall()), "tool.cancelled", true)
}

func TestRealtimeUpstreamToolFramesAreIgnored(t *testing.T) {
	s := newRealtimeStub(t, true)
	s.pushUp(inProgress("c7", "f", "{}"))
	s.push(frames.NewEndFrame())
	if got := s.spans("llm_tool_call"); len(got) != 0 {
		t.Errorf("llm_tool_call spans = %d, want none for an upstream copy", len(got))
	}
}

// ---------------------------------------------------------------------------
// Option mechanics
// ---------------------------------------------------------------------------

func TestRealtimeTracingOptionAppliedTwiceTracesOnce(t *testing.T) {
	svc := New("StubRealtime", realtimeStubGen{}, WithContinuousGeneration(), WithRealtimeTracing())
	tracer := svc.realtime
	WithRealtimeTracing()(svc)
	if svc.realtime != tracer {
		t.Error("asking for realtime tracing again replaced the tracer")
	}
}

// (Not a port of an upstream test.) A call's arguments and result are recorded
// as JSON: no arguments are an empty object, a structured result stays as it
// is, and any other result is a JSON string.
func TestRealtimeToolAttributeRendering(t *testing.T) {
	if got := toolArgumentsAttr(nil); got != "{}" {
		t.Errorf("no arguments = %q, want an empty object", got)
	}
	for result, want := range map[string]string{
		`{"temp_c":21}`: `{"temp_c":21}`,
		`[1,2]`:         `[1,2]`,
		`done`:          `"done"`,
		`42`:            `"42"`,
		`{not json`:     `"{not json"`,
	} {
		if got := toolResultAttr(result); got != want {
			t.Errorf("result %q = %q, want %q", result, got, want)
		}
	}
}
