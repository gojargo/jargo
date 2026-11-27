package pipeline

import (
	"reflect"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/embeds"
)

// FrameTypes returns a filter selecting frames of the type of one of the
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
func FrameTypes(exemplars ...frames.Frame) func(frames.Frame) bool {
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

// AnyFrame is a filter selecting every frame, for a caller that really does
// want to watch the whole stream. Prefer FrameTypes: a handler that runs on
// every frame runs on the goroutine the frame arrived on, so it is on the path
// of everything the pipeline does.
func AnyFrame(frames.Frame) bool { return true }

// Or returns a filter selecting whatever either filter selects. A nil filter
// means nothing was set rather than everything was, so combining with one
// returns the other.
func Or(a, b func(frames.Frame) bool) func(frames.Frame) bool {
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
func selects(filter func(frames.Frame) bool, f frames.Frame) bool {
	return filter != nil && filter(f)
}
