package pipeline

import (
	"reflect"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/embeds"
)

// FrameFilter reports whether a frame is one the caller asked to hear about. It
// selects which frames reaching either end of the pipeline are reported to the
// Task's reached handlers.
//
// A nil FrameFilter selects nothing, so a handler registered without one is
// never called. Say what you want to see.
type FrameFilter func(frames.Frame) bool

// FrameTypes returns a FrameFilter selecting frames of the type of one of the
// exemplars, or of a type built on it. An exemplar is read for its type alone
// and never enters a pipeline, so an empty literal is enough:
//
//	pipeline.FrameTypes(&frames.TTSStoppedFrame{}, &frames.BotStoppedSpeakingFrame{})
//
// A frame whose type embeds an exemplar's is selected too: naming TextFrame
// selects LLM tokens and transcriptions. A filter cannot name an interface;
// write the predicate directly for that:
//
//	func(f frames.Frame) bool { _, ok := f.(frames.SystemFrame); return ok }
func FrameTypes(exemplars ...frames.Frame) FrameFilter {
	want := make([]reflect.Type, 0, len(exemplars))
	for _, f := range exemplars {
		want = append(want, reflect.TypeOf(f))
	}
	return func(f frames.Frame) bool {
		got := reflect.TypeOf(f)
		for _, w := range want {
			if embeds.Is(got, w) {
				return true
			}
		}
		return false
	}
}

// AnyFrame is a FrameFilter selecting every frame, for a caller that really does
// want to watch the whole stream. Prefer FrameTypes: a handler that runs on
// every frame runs on the goroutine the frame arrived on, so it is on the path
// of everything the pipeline does.
//
//nolint:gochecknoglobals // a stateless predicate, used as a constant
var AnyFrame FrameFilter = func(frames.Frame) bool { return true }

// Or returns a filter selecting whatever either filter selects. A nil filter
// means nothing was set rather than everything was, so combining with one
// returns the other.
func Or(a, b FrameFilter) FrameFilter {
	switch {
	case a == nil:
		return b
	case b == nil:
		return a
	}
	return func(f frames.Frame) bool { return a(f) || b(f) }
}

// selects reports whether f passes the filter, treating a nil filter as
// selecting nothing.
func (ff FrameFilter) selects(f frames.Frame) bool { return ff != nil && ff(f) }
