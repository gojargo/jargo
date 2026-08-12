package observers_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// recordSpans installs a recording tracer provider for the test.
func recordSpans(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return rec
}

// spanNamed returns the one ended span with the given name.
func spanNamed(t *testing.T, rec *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	var found sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == name {
			if found != nil {
				t.Fatalf("more than one %q span ended", name)
			}
			found = s
		}
	}
	if found == nil {
		t.Fatalf("no %q span ended, have %v", name, spanNames(rec))
	}
	return found
}

func spanNames(rec *tracetest.SpanRecorder) []string {
	var names []string
	for _, s := range rec.Ended() {
		names = append(names, s.Name())
	}
	return names
}

// attrOf reads one attribute off a span.
func attrOf(s sdktrace.ReadOnlySpan, key string) (attribute.Value, bool) {
	for _, kv := range s.Attributes() {
		if string(kv.Key) == key {
			return kv.Value, true
		}
	}
	return attribute.Value{}, false
}

func startFrame() processor.FramePushed {
	return processor.FramePushed{
		Frame:     frames.NewStartFrame(),
		Direction: processor.Downstream,
	}
}

// TestTurnTraceNestsTurnsUnderConversation checks the shape of the trace: one
// conversation span per session, one turn span beneath it per turn, and the
// tracing context pointing at whichever is current.
func TestTurnTraceNestsTurnsUnderConversation(t *testing.T) {
	rec := recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{
		Tracing:        tc,
		ConversationID: "sess-1",
		Attributes: []attribute.KeyValue{
			attribute.String("langfuse.session.id", "sess-1"),
		},
	})

	o.OnPushFrame(startFrame())
	conversation := tc.ConversationContext()
	if !conversation.IsValid() {
		t.Fatal("the StartFrame should have opened the conversation")
	}
	if id := tc.ConversationID(); id != "sess-1" {
		t.Fatalf("ConversationID() = %q, want sess-1", id)
	}

	o.TurnStarted(1)
	turn := tc.TurnContext()
	if !turn.IsValid() {
		t.Fatal("a started turn should be the current tracing context")
	}
	if turn.TraceID() != conversation.TraceID() {
		t.Fatal("the turn should be in the conversation's trace")
	}

	o.LatencyMeasured(250 * time.Millisecond)
	o.TurnEnded(1, 3*time.Second, false)
	if tc.TurnContext().IsValid() {
		t.Fatal("an ended turn should no longer be the current tracing context")
	}

	o.EndConversation()
	if tc.ConversationContext().IsValid() || tc.ConversationID() != "" {
		t.Fatal("an ended conversation should be cleared from the tracing context")
	}

	turnSpan := spanNamed(t, rec, "turn")
	if parent := turnSpan.Parent(); parent.SpanID() != conversation.SpanID() {
		t.Fatalf("turn parent = %v, want the conversation span", parent.SpanID())
	}
	for key, want := range map[string]any{
		"turn.number":                   int64(1),
		"turn.type":                     "conversation",
		"conversation.id":               "sess-1",
		"turn.duration_seconds":         3.0,
		"turn.was_interrupted":          false,
		"turn.user_bot_latency_seconds": 0.25,
	} {
		v, ok := attrOf(turnSpan, key)
		if !ok {
			t.Fatalf("turn span is missing %s", key)
		}
		if got := v.AsInterface(); got != want {
			t.Fatalf("turn span %s = %v, want %v", key, got, want)
		}
	}

	conversationSpan := spanNamed(t, rec, "conversation")
	for key, want := range map[string]any{
		"conversation.id":     "sess-1",
		"conversation.type":   "voice",
		"langfuse.session.id": "sess-1",
	} {
		v, ok := attrOf(conversationSpan, key)
		if !ok {
			t.Fatalf("conversation span is missing %s", key)
		}
		if got := v.AsInterface(); got != want {
			t.Fatalf("conversation span %s = %v, want %v", key, got, want)
		}
	}
}

// TestTurnTraceEndsOpenTurn checks that a conversation ending mid-turn closes
// the turn, and says that is what closed it.
func TestTurnTraceEndsOpenTurn(t *testing.T) {
	rec := recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc})

	o.OnPushFrame(startFrame())
	o.TurnStarted(1)
	o.EndConversation()

	turnSpan := spanNamed(t, rec, "turn")
	for _, key := range []string{"turn.was_interrupted", "turn.ended_by_conversation_end"} {
		v, ok := attrOf(turnSpan, key)
		if !ok || !v.AsBool() {
			t.Fatalf("turn span %s = %v (present: %v), want true", key, v.AsInterface(), ok)
		}
	}
	if tc.TurnContext().IsValid() {
		t.Fatal("the turn should be cleared from the tracing context")
	}
}

// TestTurnTraceLateTurnEndIgnored checks that the end of a turn that is no
// longer the current one does not close the turn that replaced it.
func TestTurnTraceLateTurnEndIgnored(t *testing.T) {
	rec := recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc})

	o.OnPushFrame(startFrame())
	o.TurnStarted(1)
	o.TurnEnded(1, time.Second, true)
	o.TurnStarted(2)
	o.TurnEnded(1, time.Second, true) // late report of the turn already closed

	if !tc.TurnContext().IsValid() {
		t.Fatal("turn 2 should still be open")
	}
	if got := len(rec.Ended()); got != 1 {
		t.Fatalf("ended spans = %v, want only turn 1", spanNames(rec))
	}
}

// TestTurnTraceGeneratesConversationID checks that a session with no id of its
// own still gets one.
func TestTurnTraceGeneratesConversationID(t *testing.T) {
	recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc})

	o.OnPushFrame(startFrame())
	if tc.ConversationID() == "" {
		t.Fatal("a conversation with no id should have been given one")
	}
	o.EndConversation()
}

// TestTurnTraceStartsConversationOnce checks that only the first StartFrame
// opens the conversation.
func TestTurnTraceStartsConversationOnce(t *testing.T) {
	rec := recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc})

	o.OnPushFrame(startFrame())
	first := tc.ConversationContext()
	o.OnPushFrame(startFrame())
	if got := tc.ConversationContext(); !got.Equal(first) {
		t.Fatal("a second StartFrame should not open another conversation")
	}
	o.EndConversation()
	if got := len(rec.Ended()); got != 1 {
		t.Fatalf("ended spans = %v, want one conversation", spanNames(rec))
	}
}

// TestTurnTraceSpansEveryTurn checks that each turn of a conversation gets its
// own span, numbered, and all of them hang from the one conversation.
func TestTurnTraceSpansEveryTurn(t *testing.T) {
	rec := recordSpans(t)
	tc := tracing.NewTracingContext()
	o := observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc, ConversationID: "sess-1"})

	o.OnPushFrame(startFrame())
	conversation := tc.ConversationContext()
	for turn := 1; turn <= 3; turn++ {
		o.TurnStarted(turn)
		o.TurnEnded(turn, time.Second, false)
	}
	o.EndConversation()

	numbers := map[int64]bool{}
	for _, s := range rec.Ended() {
		if s.Name() != "turn" {
			continue
		}
		if parent := s.Parent(); parent.SpanID() != conversation.SpanID() {
			t.Errorf("turn parent = %v, want the conversation span", parent.SpanID())
		}
		v, ok := attrOf(s, "turn.number")
		if !ok {
			t.Fatal("turn span is missing turn.number")
		}
		numbers[v.AsInt64()] = true
	}
	if len(numbers) != 3 || !numbers[1] || !numbers[2] || !numbers[3] {
		t.Fatalf("turn numbers = %v, want one span each for turns 1, 2 and 3", numbers)
	}
}

// TestTurnTraceConcurrentPipelinesIsolated checks that two conversations running
// at once keep their turns apart: a turn belongs to the conversation whose
// observer opened it, never to the other one's.
func TestTurnTraceConcurrentPipelinesIsolated(t *testing.T) {
	rec := recordSpans(t)
	build := func(id string) (*observers.TurnTrace, *tracing.TracingContext) {
		tc := tracing.NewTracingContext()
		return observers.NewTurnTrace(observers.TurnTraceConfig{Tracing: tc, ConversationID: id}), tc
	}
	a, _ := build("conv-a")
	b, _ := build("conv-b")

	var wg sync.WaitGroup
	for _, o := range []*observers.TurnTrace{a, b} {
		wg.Go(func() {
			o.OnPushFrame(startFrame())
			for turn := 1; turn <= 3; turn++ {
				o.TurnStarted(turn)
				o.LatencyMeasured(100 * time.Millisecond)
				o.TurnEnded(turn, time.Second, false)
			}
			o.EndConversation()
		})
	}
	wg.Wait()

	// Map each conversation span to its id, then check every turn names the
	// conversation it is actually parented under.
	conversations := map[trace.SpanID]string{}
	for _, s := range rec.Ended() {
		if s.Name() != "conversation" {
			continue
		}
		v, _ := attrOf(s, "conversation.id")
		conversations[s.SpanContext().SpanID()] = v.AsString()
	}
	if len(conversations) != 2 {
		t.Fatalf("conversation spans = %d, want one per pipeline", len(conversations))
	}
	var turns int
	for _, s := range rec.Ended() {
		if s.Name() != "turn" {
			continue
		}
		turns++
		named, _ := attrOf(s, "conversation.id")
		parent, ok := conversations[s.Parent().SpanID()]
		if !ok {
			t.Fatalf("turn span is parented outside both conversations")
		}
		if named.AsString() != parent {
			t.Errorf("turn names conversation %q but hangs from %q", named.AsString(), parent)
		}
	}
	if turns != 6 {
		t.Fatalf("turn spans = %d, want three per pipeline", turns)
	}
}

// TestTurnTraceNilObserver checks the shape a task takes when it is not tracing.
func TestTurnTraceNilObserver(t *testing.T) {
	var o *observers.TurnTrace
	o.EndConversation()
}

var _ processor.Observer = (*observers.TurnTrace)(nil)
