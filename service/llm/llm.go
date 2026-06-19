// Package llm is the shared base for streaming LLM services. A concrete service
// embeds *Base and implements Generator; the base handles the LLMContextFrame
// lifecycle and brackets the streamed response as an LLMFullResponseStartFrame,
// a stream of LLMTextFrames, and an LLMFullResponseEndFrame.
//
// This keeps every provider down to the part that differs — turning a
// conversation into a stream of text deltas — while the frame contract lives in
// one place.
package llm

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

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

// Base is the shared LLM processor. It runs the embedded Generator on each
// LLMContextFrame and surrounds the streamed text with response start/end
// frames.
type Base struct {
	*processor.Base
	gen Generator
}

// New builds an LLM Base named name driven by gen. The concrete service passes
// itself as gen and embeds the returned Base.
func New(name string, gen Generator) *Base {
	b := &Base{gen: gen}
	b.Base = processor.New(name, b)
	return b
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

// run streams a response for the conversation. It runs under the process
// goroutine's context, so an interruption cancels the in-flight generation.
func (b *Base) run(ctx context.Context, convo *frames.LLMContext) error {
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
