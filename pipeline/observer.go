package pipeline

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Observer watches frames flowing through a pipeline without modifying them, to
// derive turn, latency or startup metrics, or to log the stream. The task
// reports each frame that reaches the pipeline's downstream end (with dir
// Downstream) or upstream end (with dir Upstream) to every registered observer.
//
// Observers must be safe for concurrent use: the two ends run on separate
// goroutines, so OnFrame may be called from either.
type Observer interface {
	OnFrame(f frames.Frame, dir processor.Direction)
}
