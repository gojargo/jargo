// Package aggregators assembles the conversation around an LLM. The user
// aggregator collects transcriptions into a user message and triggers the LLM;
// the assistant aggregator collects the streamed response into an assistant
// message. Both share one LLMContext, so the conversation accrues across turns.
//
// Place the user aggregator before the LLM and the assistant aggregator at the
// end of the pipeline:
//
//	pipeline.New(input, stt, agg.User(), llm, tts, output, agg.Assistant())
package aggregators

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Pair is a user and assistant aggregator sharing one conversation context.
type Pair struct {
	context   *frames.LLMContext
	user      *UserAggregator
	assistant *AssistantAggregator
}

// New builds a user/assistant aggregator pair around ctx.
func New(ctx *frames.LLMContext) *Pair {
	return &Pair{
		context:   ctx,
		user:      newUser(ctx),
		assistant: newAssistant(ctx),
	}
}

// User returns the user-side aggregator.
func (p *Pair) User() processor.Processor { return p.user }

// Assistant returns the assistant-side aggregator.
func (p *Pair) Assistant() processor.Processor { return p.assistant }

// Context returns the shared conversation context.
func (p *Pair) Context() *frames.LLMContext { return p.context }

// UserAggregator collects finalized transcriptions into a user message and, when
// the user's turn ends, appends it to the context and triggers the LLM with an
// LLMContextFrame.
type UserAggregator struct {
	*processor.Base
	context     *frames.LLMContext
	aggregation string
}

func newUser(ctx *frames.LLMContext) *UserAggregator {
	u := &UserAggregator{context: ctx}
	u.Base = processor.New("UserContextAggregator", u)
	return u
}

// ProcessFrame collects transcriptions and triggers the LLM.
func (u *UserAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := u.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	tf, ok := f.(*frames.TranscriptionFrame)
	if !ok {
		// Interim transcriptions and everything else just flow through.
		return u.PushFrame(ctx, f, dir)
	}

	if tf.Text != "" {
		if u.aggregation != "" {
			u.aggregation += " "
		}
		u.aggregation += tf.Text
	}

	// Forward the transcription so downstream processors (e.g. RTVI) see it.
	if err := u.PushFrame(ctx, f, dir); err != nil {
		return err
	}

	// Finalized marks the end of the user's turn; commit and run the LLM.
	if tf.Finalized && u.aggregation != "" {
		u.context.AddUserMessage(u.aggregation)
		u.aggregation = ""
		return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)
	}
	return nil
}

// AssistantAggregator collects the LLM's streamed text into a single assistant
// message and appends it to the context when the response completes.
type AssistantAggregator struct {
	*processor.Base
	context     *frames.LLMContext
	aggregation string
	started     bool
}

func newAssistant(ctx *frames.LLMContext) *AssistantAggregator {
	a := &AssistantAggregator{context: ctx}
	a.Base = processor.New("AssistantContextAggregator", a)
	return a
}

// ProcessFrame collects LLM text into an assistant message.
func (a *AssistantAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		a.started = true
		a.aggregation = ""
	case *frames.LLMTextFrame:
		if a.started {
			a.aggregation += fr.Text
		}
	case *frames.LLMFullResponseEndFrame:
		if a.aggregation != "" {
			a.context.AddAssistantMessage(a.aggregation)
		}
		a.started = false
		a.aggregation = ""
	}
	return a.PushFrame(ctx, f, dir)
}
