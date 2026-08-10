// Package service holds what every AI service in a pipeline has in common,
// whatever it does: transcription, generation or synthesis.
//
// A concrete service embeds Base rather than a bare frame processor, which is
// what gives it the behavior the rest of the pipeline expects of a service:
// describing itself to the processors around it, and describing itself again
// when asked.
package service

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// MetadataDescriber is implemented by a service that describes itself to the
// rest of the pipeline: the name it answers to, and whatever else downstream
// processors need to know about it, such as the turn-taking a transcriber with
// its own end-of-turn detection recommends.
//
// Each call must return a fresh frame. The metadata goes out in both directions,
// as two frames, so the two halves cannot share one.
type MetadataDescriber interface {
	// ServiceMetadataFrame is the metadata to broadcast, or nil to broadcast
	// nothing.
	ServiceMetadataFrame() frames.ServiceMetadata
}

// Base is the foundation of every AI service. Embed it in a concrete service
// and pass the concrete value as self, the same way a frame processor is built:
//
//	type Synth struct{ *service.Base }
//
//	func NewSynth() *Synth {
//	    s := &Synth{}
//	    s.Base = service.New("Synth", s)
//	    return s
//	}
//
// Implement MetadataDescriber on the concrete service to have it described to
// the pipeline.
type Base struct {
	*processor.Base
}

// New builds a service Base named name on behalf of self, the concrete service
// embedding it.
func New(name string, self processor.Processor, opts ...processor.Option) *Base {
	return &Base{Base: processor.New(name, self, opts...)}
}

// BroadcastServiceMetadata sends this service's metadata both ways through the
// pipeline, so processors on either side of it learn what it is. It does nothing
// for a service that does not describe itself.
func (b *Base) BroadcastServiceMetadata(ctx context.Context) {
	d, ok := b.Self().(MetadataDescriber)
	if !ok {
		return
	}
	if d.ServiceMetadataFrame() == nil {
		return
	}
	_ = b.Broadcast(ctx, func() frames.Frame { return d.ServiceMetadataFrame() })
}

// ProcessFrame handles what every service handles. A concrete service overrides
// it and calls this first.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.ServiceSwitcherRequestMetadataFrame); ok {
		// A switcher has just made this service the active one and wants the
		// pipeline told, since what it knows describes the service replaced.
		b.BroadcastServiceMetadata(ctx)
	}
	return nil
}

// PushFrame forwards a frame, and describes the service to the pipeline once it
// has started.
//
// The metadata goes out after the StartFrame rather than while it is being
// handled, so the processors it reaches have been started themselves and are
// ready to make sense of it.
func (b *Base) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, isStart := f.(*frames.StartFrame); isStart && dir == processor.Downstream {
		b.BroadcastServiceMetadata(ctx)
	}
	return nil
}
