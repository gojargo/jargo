package aggregators

import (
	"context"
	"fmt"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
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
//
// Text transforms can be registered to rewrite the aggregations on their way
// through. The frames feed the TTS service, the context aggregator and client
// output alike, so a transform registered here changes the text for all of
// them, which suits a habit the LLM keeps repeating (emoji, stage directions),
// a name it should not use, or a pattern to redact. To change only what is
// spoken, register the transform on the TTS service instead. A transform whose
// output is only whitespace drops the aggregation, so nothing is spoken or added
// to the context for it.
type LLMText struct {
	*processor.Base
	aggregator text.Aggregator
	transforms []tts.TextTransformer
	// skipTTS is the decision the last token carried, which the text still held
	// when a speech-to-speech service stops its speech goes out with.
	skipTTS *bool
}

// NewLLMText builds a processor grouping the model's output into sentences.
func NewLLMText(name string) *LLMText {
	return NewLLMTextWith(name, text.NewSimpleAggregator(frames.TextTypeSentence, ""))
}

// NewLLMTextWith builds a processor grouping the model's output with aggregator.
func NewLLMTextWith(name string, aggregator text.Aggregator) *LLMText {
	return NewLLMTextFor(name, nil, aggregator)
}

// NewLLMTextFor builds the processor for self, a processor that embeds the
// returned LLMText and handles frames of its own before passing them on to its
// ProcessFrame. A nil self builds a processor of its own, as NewLLMTextWith
// does.
func NewLLMTextFor(name string, self processor.Processor, aggregator text.Aggregator) *LLMText {
	p := &LLMText{aggregator: aggregator}
	if self == nil {
		self = p
	}
	p.Base = processor.New(name, self)
	return p
}

// SetTextTransformers sets the transforms applied to each aggregation, replacing
// any set before. They run in order, each one seeing what the one before it
// produced.
//
// Replacing the list is also how a transform is withdrawn: Go function values
// cannot be compared, so there is no removing one by naming it.
func (p *LLMText) SetTextTransformers(ts ...tts.TextTransformer) {
	p.transforms = make([]tts.TextTransformer, 0, len(ts))
	for _, t := range ts {
		p.AddTextTransformer(t)
	}
}

// AddTextTransformer appends one transform to those already set.
func (p *LLMText) AddTextTransformer(t tts.TextTransformer) {
	if t.TextType == "" {
		t.TextType = t.AggregatedBy //nolint:staticcheck // the deprecated field is honored
	}
	p.transforms = append(p.transforms, t)
}

// SkippingTTS reports whether the last token said its text is not to be spoken.
func (p *LLMText) SkippingTTS() bool { return p.skipTTS != nil && *p.skipTTS }

// ProcessFrame converts the model's tokens into aggregated text.
func (p *LLMText) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
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
	case *frames.TTSStoppedFrame:
		// A speech-to-speech service stops its speech before it ends the
		// response. The text still held was spoken, so it goes out ahead of the
		// stop: text behind it would reach the output transport after the bot
		// stopped speaking.
		if err := p.flush(ctx, p.skipTTS); err != nil {
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
	p.skipTTS = fr.SkipTTS
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

// pushAggregation sends one completed unit downstream, unless a transform
// dropped it.
func (p *LLMText) pushAggregation(ctx context.Context, unit text.Aggregation, skipTTS *bool) error {
	transformed, ok := p.transformText(ctx, unit.Text, unit.Type)
	if !ok {
		return nil
	}
	out := frames.NewAggregatedTextFrame(transformed, unit.Type)
	// The written form, which is what goes into the conversation when a
	// pattern's delimiters are not meant to be spoken.
	out.RawText = rawText(unit, transformed)
	out.AppendToContext = true
	out.SkipTTS = skipTTS
	return p.PushFrame(ctx, out, processor.Downstream)
}

// transformText runs the transforms registered for the unit's type on its text.
// It reports false when the unit is to be dropped: a transform left nothing but
// whitespace, or one failed.
func (p *LLMText) transformText(ctx context.Context, original string, textType frames.TextType) (string, bool) {
	transformed := original
	for _, t := range p.transforms {
		if t.Transform == nil || (t.TextType != textType && t.TextType != frames.AnyTextType) {
			continue
		}
		out, err := t.Transform(ctx, transformed, textType)
		if err != nil {
			// The failure came from the application's transform. Passing the text
			// on untransformed is not safe, as a transform may exist to remove
			// something, so the aggregation is dropped.
			p.PushError(ctx, fmt.Sprintf("Error transforming LLM text [%s]: %v", transformed, err), err, false,
				processor.WithErrorCategory(errs.Application))
			return "", false
		}
		transformed = out
	}
	if transformed != original && strings.TrimSpace(transformed) == "" {
		return "", false
	}
	return transformed, true
}

// rawText is the text the context records for a unit whose text is now
// transformed: a pattern match keeps its delimiters around the new text, and
// any other unit is its text.
func rawText(unit text.Aggregation, transformed string) string {
	if unit.RawText == "" || unit.Type == frames.TextTypeSentence || unit.Type == frames.TextTypeToken {
		return transformed
	}
	if transformed == unit.Text {
		return unit.RawText
	}
	// The rewritten content goes back between the delimiters. The content is the
	// first occurrence of itself in the full match.
	start := strings.Index(unit.RawText, unit.Text)
	if start < 0 {
		return transformed
	}
	return unit.RawText[:start] + transformed + unit.RawText[start+len(unit.Text):]
}
