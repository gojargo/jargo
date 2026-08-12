package processor

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/notify"
)

// FrameFilter forwards only the frame types it was built for, and drops the
// rest.
//
// The lifecycle and system frames always pass whatever the list says. A
// processor downstream of a filter still has to be started, stopped and told
// about an interruption, so gating those would break the pipeline rather than
// filter it.
type FrameFilter struct {
	*Base
	allowed []func(frames.Frame) bool
}

// NewFrameFilter builds a filter passing only the frames that match one of the
// given types. Build a matcher with FrameIs.
func NewFrameFilter(name string, allowed ...func(frames.Frame) bool) *FrameFilter {
	f := &FrameFilter{allowed: allowed}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// FrameIs builds a matcher for one frame type, for passing to NewFrameFilter.
func FrameIs[T frames.Frame]() func(frames.Frame) bool {
	return func(f frames.Frame) bool {
		_, ok := f.(T)
		return ok
	}
}

// shouldPassthrough reports whether frame is allowed through.
func (f *FrameFilter) shouldPassthrough(frame frames.Frame) bool {
	for _, match := range f.allowed {
		if match(frame) {
			return true
		}
	}
	if _, ok := frame.(*frames.EndFrame); ok {
		return true
	}
	return isSystemFrame(frame)
}

// ProcessFrame implements Processor.
func (f *FrameFilter) ProcessFrame(ctx context.Context, frame frames.Frame, dir Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	if !f.shouldPassthrough(frame) {
		return nil
	}
	return f.PushFrame(ctx, frame, dir)
}

// IdentityFilter forwards every frame unchanged.
//
// It is a processor that does nothing, which is exactly what makes it useful for
// building a branch of a ParallelPipeline that only has to carry frames through:
// the branch is then a place in the pipeline rather than a transformation, and
// no frame should come out of it twice.
type IdentityFilter struct {
	*Base
}

// NewIdentityFilter builds an IdentityFilter.
func NewIdentityFilter(name string) *IdentityFilter {
	f := &IdentityFilter{}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// ProcessFrame implements Processor.
func (f *IdentityFilter) ProcessFrame(ctx context.Context, frame frames.Frame, dir Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	return f.PushFrame(ctx, frame, dir)
}

// NullFilter drops every frame except the ones the pipeline needs to keep
// working: the end of the run and the system frames.
//
// It stops a stretch of the pipeline dead without taking it out, which is what
// makes it useful for holding a branch silent while the pipeline runs.
type NullFilter struct {
	*Base
}

// NewNullFilter builds a NullFilter.
func NewNullFilter(name string) *NullFilter {
	f := &NullFilter{}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// ProcessFrame implements Processor.
func (f *NullFilter) ProcessFrame(ctx context.Context, frame frames.Frame, dir Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	if _, ok := frame.(*frames.EndFrame); !ok && !isSystemFrame(frame) {
		return nil
	}
	return f.PushFrame(ctx, frame, dir)
}

// WakeNotifierFilter forwards every frame, and signals a notifier when a frame
// of one of the types it watches satisfies its predicate.
//
// It is how a condition seen at one point in the pipeline releases something
// held at another. Nothing is filtered out despite the name: what it decides is
// whether to signal, not whether the frame goes on.
type WakeNotifierFilter struct {
	*Base
	notifier notify.Notifier
	types    []func(frames.Frame) bool
	filter   func(frames.Frame) bool
}

// NewWakeNotifierFilter builds a filter signaling notifier when a frame
// matching one of types also satisfies filter. Build a matcher with FrameIs.
func NewWakeNotifierFilter(
	name string,
	notifier notify.Notifier,
	filter func(frames.Frame) bool,
	types ...func(frames.Frame) bool,
) *WakeNotifierFilter {
	f := &WakeNotifierFilter{notifier: notifier, types: types, filter: filter}
	f.Base = New(name, f, WithDirectMode())
	return f
}

// ProcessFrame implements Processor.
func (f *WakeNotifierFilter) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir Direction,
) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	for _, match := range f.types {
		if match(frame) {
			if f.filter(frame) {
				f.notifier.Notify()
			}
			break
		}
	}
	return f.PushFrame(ctx, frame, dir)
}
