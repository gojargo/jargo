package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// StatelessTextTransformer rewrites the text of every TextFrame passing through
// it, and forwards everything else untouched.
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
// text of each TextFrame.
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
	tf, ok := frame.(*frames.TextFrame)
	if !ok {
		return p.PushFrame(ctx, frame, dir)
	}
	// The rewritten text is a frame of its own rather than the original with its
	// text changed, so anything else holding the original still reads what
	// arrived. It goes downstream whichever way the original was traveling,
	// because rewritten text is for whoever speaks or records it.
	return p.PushFrame(ctx, frames.NewTextFrame(p.transform(tf.Text)), Downstream)
}
