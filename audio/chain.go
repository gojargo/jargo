package audio

import "context"

// Chain applies several Filters in sequence, composing multiple input-audio
// stages — noise reduction, gating, conditioning — into one Filter that a
// transport can use through Params.AudioInFilter.
type Chain struct {
	filters []Filter
}

// NewChain builds a Chain that applies filters in the given order.
func NewChain(filters ...Filter) *Chain {
	return &Chain{filters: filters}
}

// Start starts every filter in order, stopping at the first error.
func (c *Chain) Start(ctx context.Context, sampleRate int) error {
	for _, f := range c.filters {
		if err := f.Start(ctx, sampleRate); err != nil {
			return err
		}
	}
	return nil
}

// Stop stops every filter, returning the last error if any.
func (c *Chain) Stop(ctx context.Context) error {
	var err error
	for _, f := range c.filters {
		if e := f.Stop(ctx); e != nil {
			err = e
		}
	}
	return err
}

// Filter runs pcm through each filter in order.
func (c *Chain) Filter(ctx context.Context, pcm []byte) ([]byte, error) {
	var err error
	for _, f := range c.filters {
		pcm, err = f.Filter(ctx, pcm)
		if err != nil {
			return nil, err
		}
	}
	return pcm, nil
}

// Compile-time interface check.
var _ Filter = (*Chain)(nil)
