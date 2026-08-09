package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// FilterFunc reports whether a frame is allowed to pass through a FunctionFilter.
type FilterFunc func(frames.Frame) bool

// FunctionFilter forwards frames, dropping those a predicate rejects. It runs in
// direct mode, deciding on the caller's goroutine, and is the building block a
// ServiceSwitcher uses to gate a branch on or off.
//
// Two kinds of frame are never dropped, whatever the predicate says. The
// lifecycle frames (start, end and cancel) always pass, because a branch that is
// gated off still has to be started and shut down with the rest of the pipeline.
// Every other system frame passes too, unless the filter was built to decide
// those as well.
type FunctionFilter struct {
	*Base
	// dir is the direction the predicate decides. A nil direction decides both.
	dir                *Direction
	allow              FilterFunc
	filterSystemFrames bool
}

// FilterOption configures a FunctionFilter.
type FilterOption func(*FunctionFilter)

// WithFilterSystemFrames has the predicate decide system frames too, rather than
// passing them through. The lifecycle frames still always pass.
func WithFilterSystemFrames() FilterOption {
	return func(f *FunctionFilter) { f.filterSystemFrames = true }
}

// NewFunctionFilter builds a filter that gates frames using allow. dir is the
// direction the predicate decides; frames traveling the other way pass through
// untouched. A nil dir has the predicate decide both directions.
func NewFunctionFilter(name string, dir *Direction, allow FilterFunc, opts ...FilterOption) *FunctionFilter {
	f := &FunctionFilter{dir: dir, allow: allow}
	for _, opt := range opts {
		opt(f)
	}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// passthrough reports whether frame goes on regardless of what the predicate
// says.
func (f *FunctionFilter) passthrough(frame frames.Frame, dir Direction) bool {
	// A frame traveling the way this filter does not gate is not its business.
	if f.dir != nil && dir != *f.dir {
		return true
	}
	// The lifecycle frames reach every processor, gated or not: one that never
	// starts never runs, and one that never ends never shuts down.
	switch frame.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
		return true
	}
	return !f.filterSystemFrames && isSystemFrame(frame)
}

// isSystemFrame reports whether frame is in the system category.
func isSystemFrame(frame frames.Frame) bool {
	_, ok := frame.(frames.SystemFrame)
	return ok
}

// ProcessFrame drops a frame the predicate rejects and forwards everything else.
// The predicate is consulted for every frame, including the ones that pass
// regardless, because deciding is how a predicate watching the stream keeps up
// with it.
func (f *FunctionFilter) ProcessFrame(ctx context.Context, frame frames.Frame, dir Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	passthrough := f.passthrough(frame, dir)
	allowed := f.allow(frame)
	if !passthrough && !allowed {
		return nil
	}
	return f.PushFrame(ctx, frame, dir)
}
