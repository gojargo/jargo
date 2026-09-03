package rtvi_test

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// playback stands in for a transport's output end. It forwards every frame
// unchanged and carries the marker that says a frame it pushes has been through
// playback, which is what the observer keys on to report the bot's output with
// the timing of what the caller actually hears.
//
// A real output transport is the last processor in the pipeline and pushes each
// frame onward as it releases it; this is that shape without the media
// machinery, so the observer sees the same two handovers per frame: one from the
// service that produced it, and one from the end that played it.
type playback struct {
	*processor.Base
}

func newPlayback() *playback {
	p := &playback{}
	p.Base = processor.New("Playback", p)
	return p
}

// PlaysOutput marks this processor as a transport's output end.
func (p *playback) PlaysOutput() {}

// ProcessFrame forwards every frame unchanged.
func (p *playback) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}
