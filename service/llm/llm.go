// Package llm is the shared base for streaming LLM services. A concrete service
// embeds *Base and implements Generator; the base handles the LLMContextFrame
// lifecycle and brackets the streamed response as an LLMFullResponseStartFrame,
// a stream of LLMTextFrames, and an LLMFullResponseEndFrame.
//
// This keeps every provider down to the part that differs — turning a
// conversation into a stream of text deltas — while the frame contract lives in
// one place.
//
// A service that also supports tool calling implements ToolGenerator. When the
// context carries tools, the base runs a tool loop: it streams text, dispatches
// each requested call to a registered handler, emits the function-call frames,
// and re-generates until the model produces a final answer.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// ErrStopTurn, returned by a ToolHandler, ends the current turn after recording
// the tool result instead of generating a further model response. Use it for
// tools that conclude the interaction, such as ending the session.
var ErrStopTurn = errors.New("stop turn")

// Emit pushes a chunk of generated text downstream as an LLMTextFrame. A
// Generator calls it once per delta received from the provider; empty deltas are
// ignored.
type Emit func(text string) error

// Generator produces a streaming completion for a conversation. The
// implementation streams text deltas to emit until the response completes or ctx
// is canceled (an interruption). A returned error is reported upstream.
type Generator interface {
	Generate(ctx context.Context, convo *frames.LLMContext, emit Emit) error
}

// Sink receives the streamed output of a tool-capable generation: text deltas
// via Text and each requested tool call via Tool.
type Sink interface {
	Text(text string) error
	Tool(call frames.ToolCall) error
}

// ToolGenerator is implemented by services that support tool calling. It streams
// text to sink.Text and reports each tool call the model requests to sink.Tool.
// It returns when the model's turn completes — either a final text answer or a
// request to call tools — or when ctx is canceled.
type ToolGenerator interface {
	GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink Sink) error
}

// ToolHandler runs a tool call and returns the content to feed back to the model
// as the tool result. A handler that does blocking work must honor ctx; if it
// ignores cancellation an interruption is delayed until it returns.
type ToolHandler func(ctx context.Context, args json.RawMessage) (string, error)

// Base is the shared LLM processor. It runs the embedded Generator on each
// LLMContextFrame and surrounds the streamed text with response start/end
// frames. When the context carries tools and the generator supports them, it
// runs the tool loop instead.
type Base struct {
	*processor.Base
	gen Generator

	handlersMu sync.RWMutex
	handlers   map[string]ToolHandler
}

// New builds an LLM Base named name driven by gen. The concrete service passes
// itself as gen and embeds the returned Base.
func New(name string, gen Generator) *Base {
	b := &Base{gen: gen}
	b.Base = processor.New(name, b)
	return b
}

// PushTokenUsage emits a MetricsFrame carrying token usage downstream. A service
// calls it after a generation, gated on UsageMetricsEnabled, so the conversion
// from the provider's usage shape happens only when metrics are collected.
func (b *Base) PushTokenUsage(ctx context.Context, u frames.LLMTokenUsage) error {
	f := frames.NewMetricsFrame(b.Name())
	f.Tokens = &u
	return b.PushFrame(ctx, f, processor.Downstream)
}

// RegisterFunction registers a handler for the named tool. During a tool-capable
// generation, a call to that tool is dispatched to the handler.
func (b *Base) RegisterFunction(name string, h ToolHandler) {
	b.handlersMu.Lock()
	defer b.handlersMu.Unlock()
	if b.handlers == nil {
		b.handlers = make(map[string]ToolHandler)
	}
	b.handlers[name] = h
}

// ProcessFrame runs the generator on each LLMContextFrame and forwards other
// frames untouched.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if cf, ok := f.(*frames.LLMContextFrame); ok {
		return b.run(ctx, cf.Context)
	}
	return b.PushFrame(ctx, f, dir)
}

// run streams a response for the conversation, choosing the tool loop when the
// context carries tools and the generator supports them. It runs under the
// process goroutine's context, so an interruption cancels the in-flight work.
func (b *Base) run(ctx context.Context, convo *frames.LLMContext) error {
	if len(convo.Tools()) > 0 {
		if tg, ok := b.gen.(ToolGenerator); ok {
			return b.runWithTools(ctx, convo, tg)
		}
		slog.Warn("LLM service does not support tools; tools ignored", "processor", b.Name())
	}
	return b.runText(ctx, convo)
}

// runText is the text-only path: brackets the streamed deltas with response
// start/end frames.
func (b *Base) runText(ctx context.Context, convo *frames.LLMContext) error {
	if err := b.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	emit := func(text string) error {
		if text == "" {
			return nil
		}
		return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
	}
	if err := b.gen.Generate(ctx, convo, emit); err != nil && ctx.Err() == nil {
		b.PushError(ctx, "llm generation failed", err, false)
	}
	return b.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// sink adapts per-pass closures to the Sink interface.
type sink struct {
	text func(string) error
	tool func(frames.ToolCall) error
}

func (s sink) Text(t string) error          { return s.text(t) }
func (s sink) Tool(c frames.ToolCall) error { return s.tool(c) }

// runWithTools runs the tool loop: stream text, dispatch any requested calls to
// their handlers, emit the function-call frames, and re-generate until the model
// answers without calling tools. The base writes nothing to the context — the
// assistant aggregator records the tool turn from the emitted frames, keeping
// the context's single writer. An interruption cancels ctx; the loop then stops
// without emitting a partial tool turn.
func (b *Base) runWithTools(ctx context.Context, convo *frames.LLMContext, tg ToolGenerator) error {
	if err := b.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	for {
		var preamble strings.Builder
		var calls []frames.ToolCall
		s := sink{
			text: func(t string) error {
				if t == "" {
					return nil
				}
				preamble.WriteString(t)
				return b.PushFrame(ctx, frames.NewLLMTextFrame(t), processor.Downstream)
			},
			tool: func(c frames.ToolCall) error {
				calls = append(calls, c)
				return nil
			},
		}
		if err := tg.GenerateWithTools(ctx, convo, s); err != nil && ctx.Err() == nil {
			b.PushError(ctx, "llm generation failed", err, false)
			break
		}
		if ctx.Err() != nil {
			break
		}
		if len(calls) == 0 {
			break // final spoken answer; the assistant aggregator commits it
		}
		if err := b.PushFrame(ctx, frames.NewFunctionCallsStartedFrame(preamble.String(), calls), processor.Downstream); err != nil {
			return err
		}
		stop := false
		for _, c := range calls {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if err := b.PushFrame(ctx, frames.NewFunctionCallInProgressFrame(c.ID, c.Name), processor.Downstream); err != nil {
				return err
			}
			result, isErr, stopTurn := b.invoke(ctx, c)
			if stopTurn {
				stop = true
			}
			if err := b.PushFrame(ctx, frames.NewFunctionCallResultFrame(c.ID, c.Name, result, isErr), processor.Downstream); err != nil {
				return err
			}
		}
		if ctx.Err() != nil || stop {
			break
		}
		// Loop: re-read the context (which a handler may have changed) and
		// generate the model's response to the tool results.
	}
	return b.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// invoke dispatches a tool call to its handler, returning the result content,
// whether it was an error, and whether the turn should stop without
// re-generating (a handler that returned ErrStopTurn).
func (b *Base) invoke(ctx context.Context, c frames.ToolCall) (result string, isError, stop bool) {
	b.handlersMu.RLock()
	h := b.handlers[c.Name]
	b.handlersMu.RUnlock()
	if h == nil {
		return fmt.Sprintf("unknown tool %q", c.Name), true, false
	}
	out, err := h(ctx, c.Args)
	if errors.Is(err, ErrStopTurn) {
		return out, false, true
	}
	if err != nil {
		return err.Error(), true, false
	}
	return out, false, false
}
