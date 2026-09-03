package aggregators

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/text"
)

// LLMText turns the tokens a model streams into aggregated text, using the
// aggregator it was built with.
//
// It sits between the LLM and whatever reads its output, and is where the raw
// stream is grouped, categorized, rewritten or filtered before a synthesizer or
// a context aggregator sees it: grouping by sentence for a service that speaks
// better with whole sentences, or pulling code blocks out with a pattern
// aggregator so they are not read aloud.
//
// Each LLMTextFrame is consumed and what the aggregator completes is pushed on
// as an AggregatedTextFrame. Whatever is left over is flushed when the response
// ends, so the last unit is not held back waiting for a boundary that will never
// arrive.
type LLMText struct {
	*processor.Base
	aggregator text.Aggregator
	// aggregatorErr is the failure that left this processor with no aggregator,
	// reported once the pipeline starts and there is somewhere to report it.
	aggregatorErr error
}

// NewLLMText builds a processor grouping the model's output into sentences.
func NewLLMText(name string) *LLMText {
	tok, err := text.NewPunktEnglish()
	p := &LLMText{aggregatorErr: err}
	if err == nil {
		p.aggregator = text.NewSimpleAggregator(frames.AggregationSentence, tok)
	}
	p.Base = processor.New(name, p)
	return p
}

// NewLLMTextWith builds a processor grouping the model's output with aggregator.
func NewLLMTextWith(name string, aggregator text.Aggregator) *LLMText {
	p := &LLMText{aggregator: aggregator}
	p.Base = processor.New(name, p)
	return p
}

// ProcessFrame converts the model's tokens into aggregated text.
func (p *LLMText) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if _, isStart := f.(*frames.StartFrame); isStart && p.aggregatorErr != nil {
		p.PushError(ctx, "llm text processor has no aggregator", p.aggregatorErr, false)
	}

	switch fr := f.(type) {
	case *frames.InterruptionFrame:
		// What was gathered belongs to the turn that was cut off. The aggregator
		// is told it was interrupted rather than reset, because the two are not
		// the same thing to an aggregator that treats them differently.
		if p.aggregator != nil {
			p.aggregator.HandleInterruption()
		}
		return p.PushFrame(ctx, f, dir)
	case *frames.LLMTextFrame:
		return p.handleText(ctx, fr)
	case *frames.LLMFullResponseEndFrame:
		if err := p.flush(ctx, fr.SkipTTS); err != nil {
			return err
		}
		return p.PushFrame(ctx, f, dir)
	case *frames.EndFrame:
		if err := p.flush(ctx, nil); err != nil {
			return err
		}
		return p.PushFrame(ctx, f, dir)
	default:
		return p.PushFrame(ctx, f, dir)
	}
}

// Reset clears what the processor has gathered.
func (p *LLMText) Reset() {
	if p.aggregator != nil {
		p.aggregator.Reset()
	}
}

// handleText folds one token in and pushes whatever it completed.
func (p *LLMText) handleText(ctx context.Context, fr *frames.LLMTextFrame) error {
	if p.aggregator == nil {
		return nil
	}
	for _, unit := range p.aggregator.Aggregate(fr.Text) {
		if err := p.pushAggregation(ctx, unit, fr.SkipTTS); err != nil {
			return err
		}
	}
	return nil
}

// flush pushes whatever the aggregator was still holding, so the last unit of a
// response is not held back waiting for a boundary that is not coming.
func (p *LLMText) flush(ctx context.Context, skipTTS *bool) error {
	if p.aggregator == nil {
		return nil
	}
	unit, ok := p.aggregator.Flush()
	if !ok {
		return nil
	}
	return p.pushAggregation(ctx, unit, skipTTS)
}

// pushAggregation sends one completed unit downstream.
func (p *LLMText) pushAggregation(ctx context.Context, unit text.Aggregation, skipTTS *bool) error {
	out := frames.NewAggregatedTextFrame(unit.Text, unit.Type)
	// The written form the unit was cut from, which is what goes into the
	// conversation when a pattern's delimiters are not meant to be spoken.
	out.RawText = unit.Original()
	out.AppendToContext = true
	out.SkipTTS = skipTTS
	return p.PushFrame(ctx, out, processor.Downstream)
}
