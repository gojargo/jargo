package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// HandlerFunc handles a frame that reaches the edge of a pipeline. A source
// uses it for upstream frames, a sink for downstream frames.
type HandlerFunc func(ctx context.Context, f frames.Frame, dir Direction) error

// source is the entry processor of a pipeline. It forwards downstream frames to
// the next processor and hands upstream frames to an external handler.
type source struct {
	*Base
	upstream HandlerFunc
}

// NewSource returns a pipeline source. Upstream frames reaching it are passed to
// upstream; downstream frames are forwarded along the chain.
func NewSource(name string, upstream HandlerFunc) Processor {
	s := &source{upstream: upstream}
	s.Base = New(name, s, WithDirectMode())
	return s
}

// ProcessFrame routes frames by direction.
func (s *source) ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch dir {
	case Upstream:
		return s.upstream(ctx, f, dir)
	case Downstream:
		return s.PushFrame(ctx, f, dir)
	}
	return nil
}

// sink is the exit processor of a pipeline. It forwards upstream frames to the
// previous processor and hands downstream frames to an external handler.
type sink struct {
	*Base
	downstream HandlerFunc
}

// NewSink returns a pipeline sink. Downstream frames reaching it are passed to
// downstream; upstream frames are forwarded back along the chain.
func NewSink(name string, downstream HandlerFunc) Processor {
	s := &sink{downstream: downstream}
	s.Base = New(name, s, WithDirectMode())
	return s
}

// ProcessFrame routes frames by direction.
func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch dir {
	case Upstream:
		return s.PushFrame(ctx, f, dir)
	case Downstream:
		return s.downstream(ctx, f, dir)
	}
	return nil
}
