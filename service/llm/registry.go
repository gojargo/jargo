package llm

import (
	"context"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/frames"
)

// SyncToolHandlers brings the registry into line with the toolset the
// conversation advertises: it registers the handler any advertised tool carries,
// and drops the ones it registered that way for tools no longer advertised. What
// the model can call and what actually answers therefore stay the same thing,
// without an application having to keep two lists in step.
//
// It runs on every inference, since the conversation carries its current toolset
// each time. Only handlers registered from a toolset are ever dropped: one
// registered by hand is the application's to remove.
func (b *Base) SyncToolHandlers(ctx context.Context, convo *frames.LLMContext) {
	tools := convo.Tools()

	advertised := make(map[string]bool, len(tools))
	for _, t := range tools {
		if t.Handler == nil {
			// Advertise-only. Something must have registered a handler by hand, or
			// the call falls through to the missing-function answer.
			continue
		}
		advertised[t.Name] = true
		b.registerToolHandler(ctx, t)
	}
	b.dropUnadvertisedToolHandlers(advertised)
}

// registerToolHandler registers the handler a tool carries, unless the name is
// already claimed. Registering by hand always wins: it is the more deliberate
// act, and silently replacing it would make which handler runs depend on the
// order two unrelated calls happened in.
func (b *Base) registerToolHandler(ctx context.Context, t frames.Tool) {
	b.handlersMu.Lock()
	if _, claimed := b.handlers[t.Name]; claimed {
		b.handlersMu.Unlock()
		return
	}
	b.handlersMu.Unlock()

	h, ok := t.Handler.(FunctionCallHandler)
	if !ok {
		// A plain function with the right signature converts; anything else is a
		// mistake worth naming, since the tool will otherwise look registered and
		// answer only that it is unavailable.
		fn, isFunc := t.Handler.(func(context.Context, FunctionCallParams) error)
		if !isFunc {
			slog.ErrorContext(ctx, "advertised tool carries a handler of the wrong type",
				"service", b.Name(), "function", t.Name, "handler", t.Handler)
			return
		}
		h = fn
	}

	b.RegisterFunction(t.Name, h, toolCallOptions(t)...)
	b.handlersMu.Lock()
	if item, ok := b.handlers[t.Name]; ok {
		// Marked as the toolset's, so a later sync that stops advertising the tool
		// takes its handler away too.
		item.fromToolset = true
		b.handlers[t.Name] = item
	}
	b.handlersMu.Unlock()

	slog.DebugContext(ctx, "registered the handler an advertised tool carries",
		"service", b.Name(), "function", t.Name)
}

// toolCallOptions turns the call options a tool carries into registration
// options. A tool that sets neither registers with the service's own defaults,
// which is what an advertised tool that says nothing about how it runs should
// get.
func toolCallOptions(t frames.Tool) []RegisterOption {
	var opts []RegisterOption
	if t.CancelOnInterruption != nil {
		opts = append(opts, WithCancelOnInterruption(*t.CancelOnInterruption))
	}
	if t.TimeoutSecs != nil {
		opts = append(opts, WithTimeout(time.Duration(*t.TimeoutSecs*float64(time.Second))))
	}
	return opts
}

// dropUnadvertisedToolHandlers removes the handlers that came from a toolset no
// longer advertising them. Handlers registered by hand, the catch-all and the
// service's own built-in tools are left alone.
func (b *Base) dropUnadvertisedToolHandlers(advertised map[string]bool) {
	b.handlersMu.Lock()
	dropped := false
	for name, item := range b.handlers {
		if item.fromToolset && !advertised[name] {
			delete(b.handlers, name)
			dropped = true
		}
	}
	b.handlersMu.Unlock()

	if dropped {
		// Dropping the last asynchronous tool takes the built-in cancellation tool
		// with it.
		b.syncAsyncToolCancellation()
	}
}
