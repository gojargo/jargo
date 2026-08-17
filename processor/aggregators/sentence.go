package aggregators

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/text"
)

// Sentence gathers text frames into whole sentences, pushing one on only once a
// sentence has ended. It is for a downstream processor that needs coherent
// sentences rather than the fragments a model streams:
//
//	TextFrame("Hello,")  -> nothing
//	TextFrame(" world.") -> TextFrame("Hello, world.")
//
// An interim transcription is dropped: it is a guess that will be revised, so
// folding it into a sentence would repeat what the final one says.
type Sentence struct {
	*processor.Base
	tokenizer   text.SentenceTokenizer
	aggregation string
	// tokenizerErr is the failure that left this aggregator with no tokenizer,
	// reported once the pipeline starts and there is somewhere to report it.
	tokenizerErr error
}

// NewSentence builds a sentence aggregator that finds sentence boundaries the
// way the rest of the framework does.
func NewSentence(name string) *Sentence {
	tok, err := text.NewPunktEnglish()
	s := &Sentence{tokenizer: tok, tokenizerErr: err}
	s.Base = processor.New(name, s)
	return s
}

// NewSentenceWith builds a sentence aggregator finding sentence boundaries with
// tokenizer, for a caller that has one already or wants another language.
func NewSentenceWith(name string, tokenizer text.SentenceTokenizer) *Sentence {
	s := &Sentence{tokenizer: tokenizer}
	s.Base = processor.New(name, s)
	return s
}

// ProcessFrame gathers text and pushes each completed sentence.
func (s *Sentence) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if _, isStart := f.(*frames.StartFrame); isStart && s.tokenizerErr != nil {
		s.PushError(ctx, "sentence aggregator has no tokenizer", s.tokenizerErr, false)
	}

	switch fr := f.(type) {
	case *frames.InterimTranscriptionFrame:
		return nil
	case *frames.TextFrame:
		s.aggregation += fr.Text
		if s.tokenizer == nil || s.tokenizer.MatchEndOfSentence(s.aggregation) == 0 {
			return nil
		}
		said := s.aggregation
		s.aggregation = ""
		return s.PushFrame(ctx, frames.NewTextFrame(said), processor.Downstream)
	case *frames.EndFrame:
		// The session is over, so whatever is held goes out unfinished rather
		// than being lost with the processor.
		if s.aggregation != "" {
			said := s.aggregation
			s.aggregation = ""
			if err := s.PushFrame(ctx, frames.NewTextFrame(said), processor.Downstream); err != nil {
				return err
			}
		}
		return s.PushFrame(ctx, f, processor.Downstream)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}
