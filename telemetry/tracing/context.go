package tracing

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
)

// TracingContext is the tracing state of one running pipeline: the span the
// whole conversation hangs from, and the span of the turn being spoken right
// now. A task creates one per session and hands it to the processors at setup.
// The turn observer writes it as the conversation and its turns begin and end;
// the services read it to parent their spans, so a span raised from a goroutine
// that no longer holds the frame's context still lands under the turn it
// belongs to.
//
// The zero value is ready to use, and a nil *TracingContext reads as a pipeline
// with no tracing: it reports no conversation, no turn, and leaves a parent
// context alone. Safe for concurrent use.
type TracingContext struct {
	mu             sync.RWMutex
	conversation   trace.SpanContext
	turn           trace.SpanContext
	conversationID string
}

// NewTracingContext builds an empty tracing context.
func NewTracingContext() *TracingContext { return &TracingContext{} }

// SetConversationContext records the conversation span everything else hangs
// from, along with the id naming that conversation. An invalid span context
// clears both, which is how a conversation is closed.
func (c *TracingContext) SetConversationContext(sc trace.SpanContext, id string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.conversation = sc
	c.conversationID = id
}

// ConversationContext is the conversation span, invalid when none is open.
func (c *TracingContext) ConversationContext() trace.SpanContext {
	if c == nil {
		return trace.SpanContext{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conversation
}

// SetTurnContext records the turn being spoken. An invalid span context clears
// it, which is how a turn is closed.
func (c *TracingContext) SetTurnContext(sc trace.SpanContext) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.turn = sc
}

// TurnContext is the span of the turn being spoken, invalid between turns.
func (c *TracingContext) TurnContext() trace.SpanContext {
	if c == nil {
		return trace.SpanContext{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.turn
}

// ConversationID names the conversation being traced, empty when none is open.
func (c *TracingContext) ConversationID() string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conversationID
}

// Parent returns ctx re-parented to the span a service span should hang from:
// the turn being spoken, or the conversation when no turn is open. Everything
// else about ctx — its deadline, its cancellation, its values — is left as it
// was, so a caller can start its span from the returned context and go on using
// it for the work the span covers.
//
// With neither a turn nor a conversation open, ctx is returned unchanged and the
// span lands wherever ctx already pointed.
func (c *TracingContext) Parent(ctx context.Context) context.Context {
	if sc := c.TurnContext(); sc.IsValid() {
		return trace.ContextWithSpanContext(ctx, sc)
	}
	if sc := c.ConversationContext(); sc.IsValid() {
		return trace.ContextWithSpanContext(ctx, sc)
	}
	return ctx
}

// GenerateConversationID returns a fresh conversation id, for a session that
// does not carry an identifier of its own.
func GenerateConversationID() string { return uuid.NewString() }
