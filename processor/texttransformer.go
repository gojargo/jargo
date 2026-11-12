package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// StatelessTextTransformer rewrites the text of every text frame passing
// through it, of any kind (an LLM token, a transcription, an aggregated unit),
// and forwards everything else untouched.
//
// It holds nothing between frames, so it suits a rewrite that depends only on
// the text in hand: changing case, substituting a term, stripping a marker. A
// rewrite that has to see a whole sentence belongs in a text aggregator or a
// filter on the synthesizer instead, where the text has been gathered.
type StatelessTextTransformer struct {
	*Base
	transform func(string) string
}

// NewStatelessTextTransformer builds a transformer applying transform to the
// text of each text frame.
func NewStatelessTextTransformer(name string, transform func(string) string) *StatelessTextTransformer {
	p := &StatelessTextTransformer{transform: transform}
	p.Base = New(name, p, WithDirectMode())
	return p
}

// ProcessFrame implements Processor.
func (p *StatelessTextTransformer) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	// The frame passed on keeps its kind, its other fields and its direction:
	// an LLM token is still one, still bound for whatever speaks it. It is a
	// copy, so an observer still reading the frame that arrived reads what
	// arrived.
	if tf, ok := frames.AsTextFrame(frame); ok {
		if rewritten, ok := frames.WithText(frame, p.transform(tf.Text)); ok {
			frame = rewritten
		}
	}
	return p.PushFrame(ctx, frame, dir)
}
