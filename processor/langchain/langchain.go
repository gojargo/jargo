// Package langchain bridges an external "chain" — any streaming text generator,
// such as a LangChain-style runnable or a custom agent — into a jargo pipeline.
// The Processor takes the latest user message from the LLM context, invokes the
// chain, and streams its output downstream as an LLM response, so a chain can
// stand in for a jargo LLM service.
package langchain

import (
	"context"
	"slices"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Chain is a streaming text generator. It receives the latest user input and
// streams the response token by token to emit until the response is complete.
// It is the integration point for an external framework: wrap a LangChain
// runnable, an HTTP agent, or any custom generator. A returned error is reported
// upstream as a pipeline error.
type Chain func(ctx context.Context, input string, emit func(token string) error) error

// Processor runs a Chain in place of an LLM service. On each LLM context frame
// it extracts the latest user message, invokes the chain, and brackets the
// streamed tokens with the LLM response start/end frames so the rest of the
// pipeline — assistant aggregation, TTS — treats it exactly like an LLM.
type Processor struct {
	*processor.Base
	chain Chain
}

// New builds a Processor driven by chain.
func New(chain Chain) *Processor {
	p := &Processor{chain: chain}
	p.Base = processor.New("LangChain", p)
	return p
}

// ProcessFrame invokes the chain on each LLM context frame and forwards other
// frames untouched.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	cf, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return p.PushFrame(ctx, f, dir)
	}
	input := lastUserText(cf.Context)
	if input == "" {
		return p.PushFrame(ctx, f, dir)
	}
	return p.invoke(ctx, input)
}

// invoke streams the chain's response, bracketed by LLM response frames.
func (p *Processor) invoke(ctx context.Context, input string) error {
	if err := p.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	emit := func(token string) error {
		if token == "" {
			return nil
		}
		return p.PushFrame(ctx, frames.NewLLMTextFrame(token), processor.Downstream)
	}
	if err := p.chain(ctx, input, emit); err != nil && ctx.Err() == nil {
		p.PushError(ctx, "langchain invocation failed", err, false)
	}
	return p.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// lastUserText returns the trimmed text of the most recent plain user message in
// the context, or "" when there is none.
func lastUserText(convo *frames.LLMContext) string {
	for _, m := range slices.Backward(convo.Messages()) {
		if m.Role == frames.RoleUser && len(m.ToolResults) == 0 {
			return strings.TrimSpace(m.Text)
		}
	}
	return ""
}
