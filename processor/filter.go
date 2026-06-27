package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
)

// FilterFunc reports whether a frame is allowed to pass through a FunctionFilter.
type FilterFunc func(frames.Frame) bool

// FunctionFilter forwards frames, dropping those that travel in a chosen
// direction and for which a predicate returns false; frames in the other
// direction always pass. It runs in direct mode, deciding on the caller's
// goroutine, and is the building block a ServiceSwitcher uses to gate a branch
// on or off.
type FunctionFilter struct {
	*Base
	dir   Direction
	allow FilterFunc
}

// NewFunctionFilter builds a filter that gates frames moving in dir using allow.
func NewFunctionFilter(name string, dir Direction, allow FilterFunc) *FunctionFilter {
	f := &FunctionFilter{dir: dir, allow: allow}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// ProcessFrame drops a frame moving in the gated direction when the predicate
// rejects it, and forwards everything else.
func (f *FunctionFilter) ProcessFrame(ctx context.Context, frame frames.Frame, dir Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	if dir == f.dir && !f.allow(frame) {
		return nil
	}
	return f.PushFrame(ctx, frame, dir)
}
