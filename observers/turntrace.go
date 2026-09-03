package observers

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// TurnTraceConfig configures a TurnTrace observer.
type TurnTraceConfig struct {
	// Tracing is the pipeline's tracing context, which the observer writes as
	// the conversation and its turns begin and end. Required: it is what the
	// services read to parent their spans to the turn being spoken.
	Tracing *tracing.TracingContext
	// ConversationID names the conversation; empty generates one.
	ConversationID string
	// Attributes are set on the conversation span on top of the ones the
	// observer sets itself, and are where the keys a trace backend reads from
	// the root span belong (a session id, a user id, tags).
	Attributes []attribute.KeyValue
}

// TurnTrace traces a conversation and its turns. The conversation span opens
// with the pipeline and lasts the whole session; each turn opens a span beneath
// it, and the service spans of that turn nest under it in turn, so one session
// is one trace shaped like the conversation it recorded.
//
// It observes only the pipeline start; the turns themselves are reported to it
// by a TurnTracking observer, and the response latency by a UserBotLatency
// observer, through TurnStarted, TurnEnded and LatencyMeasured.
type TurnTrace struct {
	cfg TurnTraceConfig

	mu             sync.Mutex
	dd             deduper
	conversation   trace.Span
	conversationID string
	turn           trace.Span
	turnNumber     int
	// spans is the span context of every turn traced so far, so a service that
	// finishes work belonging to an earlier turn can still parent its span to
	// the turn it belongs to rather than to whichever one is open now.
	spans map[int]trace.SpanContext
}

// NewTurnTrace builds a TurnTrace observer.
func NewTurnTrace(cfg TurnTraceConfig) *TurnTrace {
	return &TurnTrace{cfg: cfg, dd: newDeduper(0), spans: map[int]trace.SpanContext{}}
}

// OnPushFrame implements processor.Observer. The conversation span opens on the
// StartFrame rather than with the first turn, so whatever a bot does before the
// user speaks — a greeting, a flow initializing — is part of the conversation.
func (o *TurnTrace) OnPushFrame(data processor.FramePushed) {
	f, dir := data.Frame, data.Direction
	if skipBroadcastSibling(f, dir) {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.dd.seenBefore(f.ID()) {
		return
	}
	if _, ok := f.(*frames.StartFrame); ok && o.conversation == nil {
		o.startConversation(o.cfg.ConversationID)
	}
}

// StartConversation opens the conversation span under id, generating one when id
// is empty. The task calls it before the pipeline runs, and the StartFrame
// handler above calls it for an observer wired up by hand. Whichever comes
// first opens the span; the other finds it open and returns. A nil observer (a
// task that is not tracing) has nothing to open.
func (o *TurnTrace) StartConversation(id string) {
	if o == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.conversation != nil {
		return
	}
	o.startConversation(id)
}

// startConversation opens the conversation span. The caller holds o.mu.
func (o *TurnTrace) startConversation(id string) {
	if id == "" {
		id = tracing.GenerateConversationID()
	}
	_, span := tracing.Tracer().Start(context.Background(), "conversation")
	span.SetAttributes(
		attribute.String("conversation.id", id),
		attribute.String("conversation.type", "voice"),
	)
	span.SetAttributes(o.cfg.Attributes...)
	o.conversation, o.conversationID = span, id
	o.cfg.Tracing.SetConversationContext(span.SpanContext(), id)
}

// EndConversation closes the conversation span, and any turn still open with it.
// The task calls it when the pipeline has stopped. A nil observer — a task that
// is not tracing — has nothing to close.
func (o *TurnTrace) EndConversation() {
	if o == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.turn != nil {
		// A turn open at the end of a conversation never got its own ending, so
		// it is reported the way an unfinished turn is: cut short, and said to
		// have been cut short by the conversation itself.
		o.turn.SetAttributes(
			attribute.Bool("turn.was_interrupted", true),
			attribute.Bool("turn.ended_by_conversation_end", true),
		)
		o.turn.End()
		o.turn = nil
		o.cfg.Tracing.SetTurnContext(trace.SpanContext{})
	}
	if o.conversation == nil {
		return
	}
	o.conversation.End()
	o.conversation = nil
	o.conversationID = ""
	o.cfg.Tracing.SetConversationContext(trace.SpanContext{}, "")
}

// TurnStarted opens the span for a turn. Wire it to a TurnTracking observer's
// OnTurnStarted.
func (o *TurnTrace) TurnStarted(turn int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.conversation == nil {
		o.startConversation(o.cfg.ConversationID)
	}
	ctx := trace.ContextWithSpanContext(context.Background(), o.conversation.SpanContext())
	_, span := tracing.Tracer().Start(ctx, "turn")
	span.SetAttributes(
		attribute.Int("turn.number", turn),
		attribute.String("turn.type", "conversation"),
	)
	if o.conversationID != "" {
		span.SetAttributes(attribute.String("conversation.id", o.conversationID))
	}
	o.turn, o.turnNumber = span, turn
	o.spans[turn] = span.SpanContext()
	o.cfg.Tracing.SetTurnContext(span.SpanContext())
}

// CurrentTurnContext is the span context of the turn being traced, for a caller
// creating a span beneath it. The zero value means no turn is open.
func (o *TurnTrace) CurrentTurnContext() trace.SpanContext {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.turn == nil {
		return trace.SpanContext{}
	}
	return o.turn.SpanContext()
}

// TurnContext is the span context of the given turn, for a caller creating a
// span beneath a turn that may already have ended. The zero value means the turn
// was never traced.
func (o *TurnTrace) TurnContext(turn int) trace.SpanContext {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.spans[turn]
}

// TurnEnded closes the span for a turn, recording how long it ran and whether it
// was cut short. Wire it to a TurnTracking observer's OnTurnEnded.
func (o *TurnTrace) TurnEnded(turn int, d time.Duration, interrupted bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	// A turn that is not the one being traced is a late report of one already
	// closed; the span it would end is gone.
	if o.turn == nil || turn != o.turnNumber {
		return
	}
	o.turn.SetAttributes(
		attribute.Float64("turn.duration_seconds", d.Seconds()),
		attribute.Bool("turn.was_interrupted", interrupted),
	)
	o.turn.End()
	o.turn = nil
	o.cfg.Tracing.SetTurnContext(trace.SpanContext{})
}

// LatencyMeasured records the user-perceived response latency on the turn it was
// measured in. Wire it to a UserBotLatency observer's OnLatency.
func (o *TurnTrace) LatencyMeasured(d time.Duration) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.turn == nil {
		return
	}
	o.turn.SetAttributes(attribute.Float64("turn.user_bot_latency_seconds", d.Seconds()))
}
